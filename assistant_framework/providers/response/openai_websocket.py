"""
OpenAI WebSocket Realtime API response provider with MCP integration.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import AsyncIterator, List, Dict, Optional, Any

import aiohttp
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

try:
    # Try relative imports first (when used as package)
    from ...interfaces.response import ResponseInterface
    from ...models.data_models import ResponseChunk, ToolCall
    from ...config import PRIMARY_USER
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.response import ResponseInterface
    from models.data_models import ResponseChunk, ToolCall
    try:
        from assistant_framework.config import PRIMARY_USER
    except ImportError:
        PRIMARY_USER = "User"


class OpenAIWebSocketResponseProvider(ResponseInterface):
    """OpenAI WebSocket Realtime API implementation with MCP tools."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI WebSocket provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key
                - model: Model to use (e.g., "gpt-4o-realtime-preview-2024-12-17")
                - max_tokens: Maximum tokens for responses
                - system_prompt: System prompt for the assistant
                - mcp_server_path: Path to MCP server script
                - mcp_venv_python: Path to MCP virtual environment Python
        """
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = config.get('model', 'gpt-4o-realtime-preview-2024-12-17')
        self.max_tokens = config.get('max_tokens', 2000)
        self.temperature = config.get('temperature', 0.8)
        self.recency_bias_prompt = config.get('recency_bias_prompt', '')
        self.system_prompt = config.get('system_prompt', '')
        
        # MCP configuration
        self.mcp_server_path = config.get('mcp_server_path')
        self.mcp_venv_python = config.get('mcp_venv_python')
        
        # Composed tool calling configuration
        self.composed_tool_calling_enabled = config.get('composed_tool_calling_enabled', True)
        self.max_tool_iterations = config.get('max_tool_iterations', 5)
        
        # State management
        self.mcp_session = None
        self.available_tools = {}
        self.openai_functions = []
        self.stdio_client = None
        self.openai_client = None  # Reusable OpenAI client for composition
        
        # Token tracking for composition API calls (separate from main Realtime API)
        self._composition_input_tokens = 0
        self._composition_output_tokens = 0
        
        # Persistent WebSocket state (reduces connection overhead from ~400ms to ~50ms)
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_lock = asyncio.Lock()
        self._last_heartbeat = 0.0
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI WebSocket provider and MCP connection."""
        try:
            # Initialize OpenAI client for composition
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=self.api_key)
            
            # Initialize MCP connection if configured
            if self.mcp_server_path:
                await self._initialize_mcp()
            
            # Pre-establish persistent WebSocket connection for faster first response
            try:
                await self._ensure_ws_connected()
                print("⚡ OpenAI Realtime WebSocket pre-connected on startup")
            except Exception as e:
                # Non-fatal: will connect on first request
                print(f"⚠️  WebSocket pre-connect failed (will retry on first request): {e}")
            
            return True
        except Exception as e:
            print(f"Failed to initialize OpenAI WebSocket provider: {e}")
            return False
    
    async def _initialize_mcp(self):
        """Initialize MCP connection and discover tools."""
        if not self.mcp_server_path:
            return
        
        try:
            # Determine Python executable
            python_cmd = str(self.mcp_venv_python) if self.mcp_venv_python and Path(self.mcp_venv_python).exists() else sys.executable
            
            # Server parameters for stdio transport
            server_params = StdioServerParameters(
                command=python_cmd,
                args=[str(self.mcp_server_path), "--transport", "stdio"],
                env=os.environ.copy()
            )
            
            # Connect to MCP server
            self.stdio_client = stdio_client(server_params)
            read, write = await self.stdio_client.__aenter__()
            self.mcp_session = ClientSession(read, write)
            await self.mcp_session.__aenter__()
            
            # Initialize MCP session
            await self.mcp_session.initialize()
            
            # Discover available tools
            await self._discover_mcp_tools()
        except Exception as e:
            print(f"Warning: MCP initialization failed: {e}")
            # Clean up partial initialization
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                except e:
                    pass
                self.mcp_session = None
            if self.stdio_client:
                try:
                    await self.stdio_client.__aexit__(None, None, None)
                except e:
                    pass
                self.stdio_client = None
            # Don't fail the entire initialization if MCP fails
            print("Continuing without MCP tools...")
    
    async def _ensure_ws_connected(self) -> aiohttp.ClientWebSocketResponse:
        """
        Ensure persistent WebSocket is connected and return it.
        
        This reuses the same WebSocket connection across requests,
        saving ~300-500ms of connection overhead per request.
        """
        import time
        
        async with self._ws_lock:
            # Check if connection is alive
            if self._ws and not self._ws.closed:
                # Check heartbeat health
                if time.time() - self._last_heartbeat > 25:
                    try:
                        await self._ws.ping()
                        self._last_heartbeat = time.time()
                    except Exception:
                        # Connection dead, will reconnect
                        self._ws = None
                else:
                    return self._ws
            
            # Create session if needed
            if not self._ws_session or self._ws_session.closed:
                self._ws_session = aiohttp.ClientSession()
            
            # Connect WebSocket
            url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            try:
                self._ws = await self._ws_session.ws_connect(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "realtime=v1",
                    },
                    heartbeat=30,
                    protocols=["openai-realtime-v1"],
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                self._last_heartbeat = time.time()
                print("⚡ Persistent OpenAI WebSocket connected")
            except Exception as e:
                print(f"❌ WebSocket connection failed: {e}")
                raise
            
            return self._ws
    
    async def ensure_ws_warm(self) -> bool:
        """
        Ensure WebSocket connection is warm (pre-connected and healthy).
        
        Call this speculatively during idle/wake word to reduce first-response latency.
        Non-blocking if already connected.
        
        Returns:
            True if connection is warm/healthy, False if reconnection needed
        """
        import time
        
        try:
            # Quick check without lock first
            if self._ws and not self._ws.closed:
                if time.time() - self._last_heartbeat < 20:
                    return True  # Already warm
                
            # Need to connect or refresh
            await self._ensure_ws_connected()
            return True
            
        except Exception as e:
            print(f"⚠️  WebSocket warm-up failed (will retry on request): {e}")
            return False
    
    async def _discover_mcp_tools(self):
        """Discover and catalog all available MCP tools."""
        if not self.mcp_session:
            return
        
        try:
            tools_result = await self.mcp_session.list_tools()
            
            for tool in tools_result.tools:
                self.available_tools[tool.name] = tool
                
                # Convert MCP tool to OpenAI function format
                openai_function = self._mcp_tool_to_openai_function(tool)
                self.openai_functions.append(openai_function)
                
        except Exception as e:
            print(f"Error discovering MCP tools: {e}")
    
    def _mcp_tool_to_openai_function(self, mcp_tool) -> Dict[str, Any]:
        """Convert MCP tool definition to OpenAI function format."""
        parameters = {"type": "object", "properties": {}, "required": []}
        
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            schema = mcp_tool.inputSchema
            if isinstance(schema, dict):
                if "properties" in schema:
                    # Avoid empty enum arrays which can break downstream validators
                    safe_properties = {}
                    for key, prop in schema["properties"].items():
                        if isinstance(prop, dict) and prop.get("type") == "string" and isinstance(prop.get("enum"), list) and len(prop.get("enum")) == 0:
                            # Drop invalid empty enum; keep as plain string
                            prop = {k: v for k, v in prop.items() if k != "enum"}
                        safe_properties[key] = prop
                    parameters["properties"] = safe_properties
                if "required" in schema:
                    parameters["required"] = schema["required"]
        
        return {
            "type": "function",
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
            "parameters": parameters,
        }
    
    def get_composition_token_usage(self) -> Dict[str, int]:
        """Get token usage from composition/tool API calls (gpt-4o-mini calls)."""
        return {
            "input_tokens": self._composition_input_tokens,
            "output_tokens": self._composition_output_tokens,
            "total": self._composition_input_tokens + self._composition_output_tokens
        }
    
    def reset_composition_tokens(self) -> None:
        """Reset composition token counters (call at start of each response)."""
        self._composition_input_tokens = 0
        self._composition_output_tokens = 0
    
    async def stream_response(self, 
                            message: str, 
                            context: Optional[List[Dict[str, str]]] = None,
                            tool_context: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[ResponseChunk]:
        """Stream a response for the given message with optional context."""
        # Reset composition token counters for this response
        self.reset_composition_tokens()
        
        # Build messages list from context
        messages = list(context) if context else []
        messages.append({"role": "user", "content": message})
        
        # Build structured instructions with clear sections
        effective_instructions = self._build_structured_instructions(messages, tool_context)
        
        # Check if home-related for tool inclusion
        tools = self._should_include_tools(message) if self.openai_functions else None
        
        # Perform WebSocket roundtrip
        async for chunk in self._ws_stream_roundtrip(messages, tools, instructions=effective_instructions):
            yield chunk
    
    def _build_structured_instructions(self, 
                                       messages: List[Dict[str, Any]], 
                                       tool_context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build structured instructions with clear XML-like sections.
        
        Organizes all system content into distinct, labeled sections for the model:
        - <instructions>: Main system prompt (with persistent memory)
        - <recent_context>: Vector memory / past conversation context
        - <tool_guidance>: Compact context for tool selection
        - <response_guidance>: Recency bias and other response instructions
        
        Args:
            messages: Full message list (may contain system messages)
            tool_context: Compact context for tool decisions
            
        Returns:
            Structured instructions string
        """
        sections = []
        
        # Categorize system messages by their purpose
        main_system_prompt = None
        vector_context = None
        other_system = []
        
        for m in messages:
            if m.get("role") != "system":
                continue
            content = m.get("content", "").strip()
            if not content:
                continue
            
            # Detect vector memory context (has specific header)
            if content.startswith("RELEVANT PAST CONVERSATIONS:"):
                vector_context = content
            # First substantial system message is likely the main prompt
            elif main_system_prompt is None and len(content) > 100:
                main_system_prompt = content
            else:
                other_system.append(content)
        
        # Fallback to provider-configured system prompt if no main prompt found
        if not main_system_prompt and self.system_prompt:
            main_system_prompt = self.system_prompt
        
        # Section 1: Main instructions (system prompt with persistent memory)
        if main_system_prompt:
            sections.append(f"<instructions>\n{main_system_prompt}\n</instructions>")
        
        # Section 2: Recent context from vector memory (past conversations)
        if vector_context:
            sections.append(f"<recent_context>\n{vector_context}\n</recent_context>")
        
        # Section 3: Tool guidance (compact recent context for tool selection)
        if tool_context:
            tool_lines = []
            for m in tool_context:
                role = m.get("role", "user")
                content = m.get("content", "")
                # Skip system messages in tool context (already captured above)
                if role == "system" or not content:
                    continue
                tool_lines.append(f"{role}: {content}")
            
            if tool_lines:
                tool_guidance = (
                    "For tool selection, consider this recent context to infer device references and intent. "
                    "Use current request for tool parameters, not past context:\n" + 
                    "\n".join(tool_lines)
                )
                sections.append(f"<tool_guidance>\n{tool_guidance}\n</tool_guidance>")
        
        # Section 4: Response guidance (recency bias, other instructions)
        response_guidance_parts = []
        if self.recency_bias_prompt:
            response_guidance_parts.append(self.recency_bias_prompt)
        response_guidance_parts.extend(other_system)
        
        if response_guidance_parts:
            sections.append(f"<response_guidance>\n{chr(10).join(response_guidance_parts)}\n</response_guidance>")
        
        return "\n\n".join(sections)
    
    def _should_include_tools(self, message: str) -> Optional[List[Dict[str, Any]]]:
        """Expose all discovered tools and let the model decide which to call."""
        if not self.openai_functions:
            return None
        return self.openai_functions
    
    async def _ws_stream_roundtrip(self,
                                  messages: List[Dict[str, Any]],
                                  tools: Optional[List[Dict[str, Any]]] = None,
                                  instructions: str = "") -> AsyncIterator[ResponseChunk]:
        """Perform WebSocket roundtrip with streaming response using persistent connection."""
        # Extract non-system messages
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Prepare realtime tools if provided
        realtime_tools = None
        if tools:
            realtime_tools = [self._flatten_tool_schema(t) for t in tools]
        tools_enabled = realtime_tools is not None
        
        collected_text = []
        function_calls = []
        
        # Use persistent WebSocket connection (saves ~300-500ms)
        try:
            ws = await self._ensure_ws_connected()
        except Exception as e:
            print(f"❌ Failed to get WebSocket connection: {e}")
            raise
        
        try:
            # Configure session (updates config on persistent connection)
            effective_instructions = instructions or self.system_prompt or ""
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    # IMPORTANT: Realtime models use session instructions; we must pass the
                    # context's system prompt here (including persistent memory), not just the
                    # provider's static config.
                    "instructions": effective_instructions,
                    "voice": "alloy",
                    "temperature": self.temperature,
                }
            }
            
            if realtime_tools:
                session_config["session"]["tools"] = realtime_tools
                session_config["session"]["tool_choice"] = "auto"
            
            # Log token breakdown for API request
            self._log_api_tokens(
                instructions=effective_instructions,
                messages=conversation_messages,
                tools=realtime_tools
            )
            
            await ws.send_json(session_config)
            
            # Add conversation context
            for msg in conversation_messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if not content:
                    continue
                
                item_type = "message"
                content_type = "input_text" if role == "user" else "text"
                
                await ws.send_json({
                    "type": "conversation.item.create",
                    "item": {
                        "type": item_type,
                        "role": role if role in ["user", "assistant"] else "assistant",
                        "content": [{"type": content_type, "text": content}]
                    }
                })
            
            # Request response
            await ws.send_json({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "max_output_tokens": self.max_tokens
                }
            })
            
            # Stream response chunks
            response_completed = False
            while not response_completed:
                try:
                    event = await ws.receive(timeout=2)
                except asyncio.TimeoutError:
                    continue
                
                if event.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(event.data)
                        etype = data.get("type")
                        
                        if etype == "response.text.delta":
                            delta = data.get("delta", "")
                            if delta:
                                collected_text.append(delta)
                                if not tools_enabled:
                                    # Only stream deltas when tools are not used
                                    yield ResponseChunk(
                                        content=delta,
                                        is_complete=False
                                    )
                        
                        elif etype == "response.function_call_arguments.delta":
                            # Streaming function call args; accumulate per active call
                            name = data.get("name", "")
                            arguments_delta = data.get("delta", "")
                            if name:
                                if not function_calls or function_calls[-1].get("name") != name:
                                    function_calls.append({"name": name, "arguments": arguments_delta})
                                else:
                                    function_calls[-1]["arguments"] += arguments_delta
                        elif etype == "response.function_call_arguments.done":
                            name = data.get("name", "")
                            arguments = data.get("arguments", "{}")
                            if name:
                                function_calls.append({
                                    "name": name,
                                    "arguments": arguments
                                })
                        
                        elif etype in ("response.done", "response.completed"):
                            response_completed = True
                            
                            # Handle function calls if any
                            if function_calls:
                                # Filter: only ONE calendar_data write per request to prevent duplicate events
                                function_calls = self._filter_duplicate_calendar_writes(function_calls)
                                
                                # Execute all tool calls in parallel for better performance
                                async def execute_one(fc):
                                    try:
                                        args = json.loads(fc["arguments"])
                                    except Exception:
                                        args = {}
                                    tool_call = ToolCall(name=fc["name"], arguments=args)
                                    tool_call.result = await self.execute_tool(fc["name"], args)
                                    return tool_call
                                
                                tool_calls = await asyncio.gather(*[execute_one(fc) for fc in function_calls])
                                
                                # Check if composed tool calling is enabled
                                if self.composed_tool_calling_enabled:
                                    # Use iterative tool execution for multi-step tasks
                                    user_message = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                                    context = [m for m in messages if m.get("role") in ("user", "assistant")]
                                    
                                    final_text, all_tool_calls = await self._iterative_tool_execution(
                                        user_message=user_message,
                                        context=context,
                                        initial_tool_calls=list(tool_calls),
                                        instructions=instructions,
                                    )
                                    
                                    yield ResponseChunk(
                                        content=final_text,
                                        is_complete=True,
                                        tool_calls=all_tool_calls,
                                        finish_reason="stop"
                                    )
                                else:
                                    # Original behavior: compose final answer directly
                                    final_text = await self._compose_final_answer(
                                        user_message=next((m.get("content", "") for m in messages if m.get("role") == "user"), ""),
                                        context=[m for m in messages if m.get("role") in ("user", "assistant")],
                                        tool_calls=tool_calls,
                                        pre_text="",
                                        instructions=instructions,
                                    )

                                    yield ResponseChunk(
                                        content=final_text,
                                        is_complete=True,
                                        tool_calls=tool_calls,
                                        finish_reason="stop"
                                    )
                            else:
                                # No function call emitted; try heuristic fallback for inbox queries
                                final_text = "".join(collected_text)
                                fallback_tool_calls = []
                                try:
                                    heuristic = ("inbox" in (messages[-1].get("content", "").lower()) or
                                                 "email" in (messages[-1].get("content", "").lower()))
                                    if heuristic and self.available_tools.get("get_notifications"):
                                        args = {"user": PRIMARY_USER, "type_filter": "email", "limit": 10}
                                        tool_call = ToolCall(name="get_notifications", arguments=args)
                                        result = await self.execute_tool("get_notifications", args)
                                        tool_call.result = result
                                        fallback_tool_calls.append(tool_call)
                                        # Compose final answer using tool result
                                        final_text = await self._compose_final_answer(
                                            user_message=messages[-1].get("content", ""),
                                            context=[m for m in messages if m.get("role") in ("user", "assistant")],
                                            tool_calls=fallback_tool_calls,
                                            pre_text="",
                                            instructions=instructions,
                                        )
                                except Exception:
                                    pass

                                yield ResponseChunk(
                                    content=final_text,
                                    is_complete=True,
                                    tool_calls=fallback_tool_calls if fallback_tool_calls else None,
                                    finish_reason="stop"
                                )
                            break
                        
                        elif etype == "error":
                            err = data.get("error", {})
                            message = err.get("message", "Unknown error")
                            # On error, mark connection as dead so it will be reconnected
                            self._ws = None
                            raise Exception(f"OpenAI WebSocket error: {message}")
                            
                    except json.JSONDecodeError:
                        continue
                
                elif event.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    # Connection closed/errored, mark as dead
                    self._ws = None
                    break
                    
        except Exception as e:
            # On any exception, mark connection as potentially dead
            self._ws = None
            raise
    
    def _log_api_tokens(
        self,
        instructions: str = "",
        messages: List[Dict[str, Any]] = None,
        tools: List[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Log detailed token breakdown for an API request.
        
        Args:
            instructions: System instructions/prompt sent to the model
            messages: Conversation messages being sent
            tools: Tool definitions being sent
            
        Returns:
            Dictionary with token counts
        """
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            # Fallback if tiktoken not available
            encoder = None
        
        def count(text: str) -> int:
            if not text:
                return 0
            if encoder:
                try:
                    return len(encoder.encode(text))
                except Exception:
                    pass
            return len(text) // 4  # Rough estimate
        
        messages = messages or []
        tools = tools or []
        
        # Count instruction tokens
        instruction_tokens = count(instructions)
        
        # Count message tokens with structure overhead
        message_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            # Add overhead for message structure (~4 tokens per message)
            message_tokens += count(content) + 4
        
        # Count tool definition tokens
        tool_tokens = 0
        for tool in tools:
            tool_tokens += count(json.dumps(tool))
        
        total = instruction_tokens + message_tokens + tool_tokens
        
        # Print detailed breakdown
        print(f"📊 API Input: {total:,} tokens "
              f"(instructions: {instruction_tokens:,}, "
              f"messages: {message_tokens:,}, "
              f"tools: {tool_tokens:,})")
        
        return {
            "instructions": instruction_tokens,
            "messages": message_tokens,
            "tools": tool_tokens,
            "total": total
        }
    
    def _flatten_tool_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten tool schema for Realtime API."""
        if "function" in tool:
            func = tool["function"]
            return {
                "type": "function",
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {}),
            }
        return tool
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools/functions."""
        return self.openai_functions
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool/function call via MCP."""
        if not self.mcp_session:
            return "Error: MCP not initialized"
        
        try:
            print(f"🔧 Executing MCP tool: {tool_name} with args: {arguments}")
            result = await self.mcp_session.call_tool(tool_name, arguments)
            print(f"📥 Tool result type: {type(result)}")
            print(f"📥 Tool result: {result}")
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    else:
                        content_parts.append(str(content_item))
                result_text = '\n'.join(content_parts)
                print(f"📥 Extracted result text: {result_text}")
                
                # Play audio feedback based on result
                self._play_tool_feedback(result_text)
                
                return result_text
            else:
                result_text = str(result)
                print(f"📥 Converted result to string: {result_text}")
                
                # Play audio feedback based on result
                self._play_tool_feedback(result_text)
                
                return result_text
                
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {e}"
            print(f"❌ Tool execution error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Play failure sound for exceptions
            from assistant_framework.utils.audio.tones import beep_tool_failure
            beep_tool_failure()
            
            return error_msg
    
    def _play_tool_feedback(self, result_text: str) -> None:
        """Play audio feedback based on tool execution result."""
        try:
            import json
            from assistant_framework.utils.audio.tones import beep_tool_success, beep_tool_failure
            
            # Try to parse as JSON to check for success field
            try:
                result_data = json.loads(result_text)
                
                # Check common success indicators
                if isinstance(result_data, dict):
                    success = result_data.get('success')
                    error = result_data.get('error')
                    
                    if success is True or (success is None and not error):
                        # Success: either explicit success=true or no error field
                        beep_tool_success()
                    elif success is False or error:
                        # Failure: explicit success=false or error field present
                        beep_tool_failure()
                    else:
                        # Ambiguous result - default to success
                        beep_tool_success()
                else:
                    # Non-dict JSON - assume success
                    beep_tool_success()
                    
            except json.JSONDecodeError:
                # Not JSON - check for error keywords in text
                result_lower = result_text.lower()
                if any(keyword in result_lower for keyword in ['error', 'failed', 'exception', 'not found']):
                    beep_tool_failure()
                else:
                    beep_tool_success()
                    
        except Exception:
            # If feedback fails, don't propagate - it's non-critical
            pass
    
    async def _compose_final_answer(self,
                                    user_message: str,
                                    context: List[Dict[str, Any]],
                                    tool_calls: List[ToolCall],
                                    pre_text: str,
                                    instructions: str = "") -> str:
        """Integrate tool results into a single assistant reply using OpenAI Chat Completions."""
        try:
            # Use the reusable client initialized during setup
            client = self.openai_client
            if not client:
                # Fallback if not initialized
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)

            # Build messages
            messages: List[Dict[str, str]] = []
            effective_system = instructions or self.system_prompt
            if effective_system:
                messages.append({"role": "system", "content": effective_system})
            # Include recent context (truncate to avoid very long prompts)
            for msg in context[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role if role in ("user", "assistant", "system") else "assistant", "content": content})

            messages.append({"role": "user", "content": user_message})

            # Tool results summary fed as system guidance
            tool_summaries = []
            for tc in tool_calls:
                if tc and tc.result:
                    tool_summaries.append(f"{tc.name} result:\n{tc.result}")
            tools_block = "\n\n".join(tool_summaries) if tool_summaries else ""
            
            print("🔍 Composing final answer with tool results:")
            print(f"   Tool calls: {len(tool_calls)}")
            print(f"   Tools block: {tools_block[:500]}")

            guidance = (
                "You have executed tools for the user's request. "
                "Use the tool results below to produce a concise, direct answer for the user. "
                "Do not include raw JSON unless it improves clarity.\n\n" + tools_block
            )
            if pre_text.strip():
                guidance = pre_text + "\n\n" + guidance

            messages.append({"role": "system", "content": guidance})

            # Generate final answer
            result = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.6,
                max_tokens=min(self.max_tokens, 800),
            )
            content = result.choices[0].message.content if result and result.choices else ""
            
            # Track token usage from composition API call
            if result and result.usage:
                usage = result.usage
                self._composition_input_tokens += usage.prompt_tokens or 0
                self._composition_output_tokens += usage.completion_tokens or 0
                print(f"📊 Composition API: +{usage.prompt_tokens or 0} in, +{usage.completion_tokens or 0} out")
            
            return content or ""
        except Exception as e:
            # Fallback: simple concatenation
            fallback = pre_text.strip()
            if tool_calls:
                parts = [fallback] if fallback else []
                for tc in tool_calls:
                    if tc and tc.result:
                        parts.append(str(tc.result))
                fallback = "\n\n".join(parts)
            return fallback or f"(Failed to compose final answer: {e})"

    async def _iterative_tool_execution(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
        initial_tool_calls: List[ToolCall],
        instructions: str = ""
    ) -> tuple:
        """
        Execute tools iteratively, allowing the AI to chain multiple tools for multi-step tasks.
        
        This enables composed tool calling where the AI can:
        1. Execute initial tools
        2. Review results and decide if more tools are needed
        3. Continue until the task is complete or max iterations reached
        
        Args:
            user_message: The user's original message
            context: Conversation context
            initial_tool_calls: First round of tool calls already executed
            instructions: System instructions
            
        Returns:
            Tuple of (final_text, all_tool_calls)
        """
        all_tool_calls = list(initial_tool_calls)
        iteration = 1
        
        print(f"🔄 [Iteration {iteration}/{self.max_tool_iterations}] Initial tools executed: {[tc.name for tc in initial_tool_calls]}")
        
        # Check if we should try for more tools
        while iteration < self.max_tool_iterations:
            # Ask the AI if more tools are needed given the current results
            more_tools = await self._check_for_additional_tools(
                user_message=user_message,
                context=context,
                tool_calls_so_far=all_tool_calls,
                instructions=instructions,
            )
            
            if not more_tools:
                print(f"✅ [Iteration {iteration}] No more tools needed, composing final answer")
                break
            
            iteration += 1
            print(f"🔄 [Iteration {iteration}/{self.max_tool_iterations}] AI requested additional tools: {[t['name'] for t in more_tools]}")
            
            # Execute the new tools
            async def execute_one(fc):
                tool_call = ToolCall(name=fc["name"], arguments=fc.get("arguments", {}))
                tool_call.result = await self.execute_tool(fc["name"], fc.get("arguments", {}))
                return tool_call
            
            new_tool_calls = await asyncio.gather(*[execute_one(fc) for fc in more_tools])
            all_tool_calls.extend(new_tool_calls)
            
            print(f"📊 [Iteration {iteration}] Total tools executed: {len(all_tool_calls)}")
        
        if iteration >= self.max_tool_iterations:
            print(f"⚠️ Max tool iterations ({self.max_tool_iterations}) reached")
        
        # Compose final answer with all accumulated tool results
        final_text = await self._compose_final_answer(
            user_message=user_message,
            context=context,
            tool_calls=all_tool_calls,
            pre_text="",
            instructions=instructions,
        )
        
        return final_text, all_tool_calls
    
    def _get_tool_signature(self, name: str, arguments: Dict[str, Any]) -> str:
        """Generate a unique signature for a tool call to detect duplicates."""
        import hashlib
        args_str = json.dumps(arguments, sort_keys=True) if arguments else "{}"
        return hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()
    
    def _filter_duplicate_calendar_writes(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out duplicate calendar_data write operations.
        Only ONE calendar_data create_event call is allowed per user request to prevent
        duplicate events being created on the calendar.
        
        Args:
            function_calls: List of function calls with 'name' and 'arguments' keys
            
        Returns:
            Filtered list with only ONE calendar write operation allowed
        """
        filtered = []
        calendar_write_seen = False
        
        for fc in function_calls:
            name = fc.get("name", "")
            
            if name == "calendar_data":
                # Parse arguments to check if it's a write operation
                try:
                    args = json.loads(fc.get("arguments", "{}")) if isinstance(fc.get("arguments"), str) else fc.get("arguments", {})
                except json.JSONDecodeError:
                    args = {}
                
                commands = args.get("commands", [])
                is_write = any(
                    cmd.get("read_or_write") in ("write", "create_event") or
                    cmd.get("action") in ("write", "create_event") or
                    cmd.get("write_type") == "create_event"
                    for cmd in commands
                )
                
                if is_write:
                    if calendar_write_seen:
                        print(f"🚫 Blocking duplicate calendar write operation to prevent event duplication")
                        continue
                    calendar_write_seen = True
            
            filtered.append(fc)
        
        return filtered
    
    def _has_calendar_write(self, tool_calls: List[ToolCall]) -> bool:
        """Check if any of the tool calls include a calendar write/create_event operation."""
        for tc in tool_calls:
            if tc and tc.name == "calendar_data":
                args = tc.arguments or {}
                commands = args.get("commands", [])
                for cmd in commands:
                    if cmd.get("read_or_write") in ("write", "create_event") or \
                       cmd.get("action") in ("write", "create_event"):
                        return True
        return False
    
    async def _check_for_additional_tools(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
        tool_calls_so_far: List[ToolCall],
        instructions: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Ask the AI if additional tools are needed to complete the user's request.
        
        Returns:
            List of tool calls if more tools needed, empty list otherwise
        """
        try:
            client = self.openai_client
            if not client:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)
            
            # Build set of already-executed tool signatures to prevent duplicates
            executed_signatures = set()
            for tc in tool_calls_so_far:
                if tc:
                    sig = self._get_tool_signature(tc.name, tc.arguments)
                    executed_signatures.add(sig)
            
            # Build the tool results summary with explicit success indicators
            tool_summaries = []
            successful_tools = []
            for tc in tool_calls_so_far:
                if tc and tc.result:
                    # Check if result indicates success
                    is_success = '"success":true' in tc.result.lower() or '"success": true' in tc.result.lower()
                    status = "✓ SUCCESS" if is_success else "Result"
                    if is_success:
                        successful_tools.append(tc.name)
                    
                    # Truncate long results to avoid context overflow
                    result_preview = tc.result[:500] + "..." if len(tc.result) > 500 else tc.result
                    tool_summaries.append(f"[{status}] Tool '{tc.name}':\n{result_preview}")
            tools_summary = "\n\n".join(tool_summaries)
            
            # Build messages for the AI
            messages: List[Dict[str, Any]] = []
            
            # Build list of successful actions for clearer context
            success_note = ""
            if successful_tools:
                success_note = f"\n\n⚠️ ALREADY COMPLETED: {', '.join(successful_tools)} executed successfully. Do NOT call these again."
            
            # System prompt with tool awareness
            system_content = (
                f"{instructions}\n\n"
                "You are in the middle of fulfilling a user request. "
                "Review the tool results below and the original user request carefully.\n\n"
                "RULES:\n"
                "1. Check if ALL parts of the user's request are fulfilled. Multi-step requests (e.g., 'find X AND send it to Y') require MULTIPLE tools.\n"
                "2. If the user asked to do multiple things (find something AND text/email it, search AND save, etc.), make sure EACH step is done.\n"
                "3. Do NOT call the same tool with the same arguments twice - duplicates are forbidden.\n"
                "4. CALENDAR: Only ONE calendar_data call per request. If calendar_data was already called to create an event, do NOT call it again - the event is already created.\n"
                "5. If ALL parts of the user's request are fulfilled, respond with: DONE\n"
                "6. If there are unfulfilled parts, call the appropriate tool(s) to complete them.\n"
                f"{success_note}\n\n"
                f"Tools already executed:\n{tools_summary}"
            )
            messages.append({"role": "system", "content": system_content})
            
            # Add context
            for msg in context[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role if role in ("user", "assistant") else "assistant", "content": content})
            
            messages.append({"role": "user", "content": user_message})
            
            # Convert our OpenAI functions to the tools format for function calling
            tools = []
            for func in self.openai_functions:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters", {"type": "object", "properties": {}})
                    }
                })
            
            # Ask the AI with function calling enabled
            result = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto",
                temperature=0.3,  # Lower temperature for more deterministic tool selection
                max_tokens=500,
            )
            
            # Track token usage from tool decision API call
            if result and result.usage:
                usage = result.usage
                self._composition_input_tokens += usage.prompt_tokens or 0
                self._composition_output_tokens += usage.completion_tokens or 0
                print(f"📊 Tool decision API: +{usage.prompt_tokens or 0} in, +{usage.completion_tokens or 0} out")
            
            choice = result.choices[0] if result and result.choices else None
            if not choice:
                return []
            
            # Check if the AI wants to call more tools
            if choice.message.tool_calls:
                additional_tools = []
                
                # Check if a calendar write was already executed - block any further calendar writes
                calendar_write_already_done = self._has_calendar_write(tool_calls_so_far)
                
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    # Check if this tool call is a duplicate
                    sig = self._get_tool_signature(tc.function.name, args)
                    if sig in executed_signatures:
                        print(f"🚫 Skipping duplicate tool call: {tc.function.name}")
                        continue
                    
                    # Block additional calendar writes if one was already executed
                    if tc.function.name == "calendar_data" and calendar_write_already_done:
                        commands = args.get("commands", [])
                        is_write = any(
                            cmd.get("read_or_write") in ("write", "create_event") or
                            cmd.get("action") in ("write", "create_event")
                            for cmd in commands
                        )
                        if is_write:
                            print(f"🚫 Blocking calendar write - event already created in this request")
                            continue
                    
                    additional_tools.append({
                        "name": tc.function.name,
                        "arguments": args
                    })
                
                # If all requested tools were duplicates, we're done
                if not additional_tools:
                    print(f"✅ All requested tools were duplicates - task complete")
                    return []
                
                return additional_tools
            
            # Check if the response contains "DONE" or similar
            content = choice.message.content or ""
            if "DONE" in content.upper() or not content.strip():
                return []
            
            # No tool calls and no DONE - treat as done
            return []
            
        except Exception as e:
            print(f"⚠️ Error checking for additional tools: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def cleanup(self) -> None:
        """Clean up resources including persistent WebSocket."""
        # Close persistent WebSocket
        async with self._ws_lock:
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.close()
                    print("✅ Persistent WebSocket closed")
                except Exception as e:
                    print(f"WebSocket close error: {e}")
            self._ws = None
            
            if self._ws_session and not self._ws_session.closed:
                try:
                    await self._ws_session.close()
                except Exception as e:
                    print(f"WebSocket session close error: {e}")
            self._ws_session = None
        
        # Close MCP session
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
            except Exception as e:
                print(f"MCP session cleanup error: {e}")
            finally:
                self.mcp_session = None
        
        # Close stdio client
        if self.stdio_client:
            try:
                await self.stdio_client.__aexit__(None, None, None)
            except Exception as e:
                print(f"STDIO client cleanup error: {e}")
            finally:
                self.stdio_client = None
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'batch': False,
            'tools': True,
            'max_tokens': self.max_tokens,
            'models': [self.model],
            'features': ['mcp_integration', 'function_calling', 'real_time']
        }