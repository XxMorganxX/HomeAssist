"""
OpenAI WebSocket Realtime API response provider with MCP integration.
"""

import asyncio
import json
import sys
import os
import time
from pathlib import Path
from typing import AsyncIterator, List, Dict, Optional, Any

import aiohttp
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

try:
    # Try relative imports first (when used as package)
    from ...interfaces.response import ResponseInterface
    from ...models.data_models import ResponseChunk, ToolCall, HandoffContext
    from ...config import PRIMARY_USER, TOOL_SUBAGENT_CONFIG, TOOL_SIGNAL_MODE
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.response import ResponseInterface
    from models.data_models import ResponseChunk, ToolCall, HandoffContext
    try:
        from assistant_framework.config import PRIMARY_USER, TOOL_SUBAGENT_CONFIG, TOOL_SIGNAL_MODE
    except ImportError:
        PRIMARY_USER = "User"
        TOOL_SUBAGENT_CONFIG = {}
        TOOL_SIGNAL_MODE = True


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
        self.mcp_sse_url = config.get('mcp_sse_url', 'http://127.0.0.1:3000/sse')
        self._mcp_sse_client = None  # Persistent SSE client context
        
        # Composed tool calling configuration
        self.composed_tool_calling_enabled = config.get('composed_tool_calling_enabled', True)
        self.max_tool_iterations = config.get('max_tool_iterations', 5)
        self.tool_subagent_model = config.get('tool_subagent_model', 'o4-mini')
        
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
        
        # TTS provider for tool call announcements (set by orchestrator)
        self._tts_provider = None
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI WebSocket provider and MCP connection."""
        try:
            # Initialize OpenAI client for composition
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=self.api_key)
            
            # Run MCP init and WebSocket pre-connect in PARALLEL for faster boot
            # These are independent operations that can run simultaneously
            async def init_mcp():
                if self.mcp_server_path:
                    await self._initialize_mcp()
            
            async def init_ws():
                try:
                    await self._ensure_ws_connected()
                    print("⚡ OpenAI Realtime WebSocket pre-connected on startup")
                except Exception as e:
                    # Non-fatal: will connect on first request
                    print(f"⚠️  WebSocket pre-connect failed (will retry on first request): {e}")
            
            # Run both in parallel
            await asyncio.gather(init_mcp(), init_ws(), return_exceptions=True)
            
            return True
        except Exception as e:
            print(f"Failed to initialize OpenAI WebSocket provider: {e}")
            return False
    
    def set_tts_provider(self, tts_provider) -> None:
        """
        Store TTS provider reference for tool call announcements.
        
        Called by orchestrator after initialization to enable TTS announcements
        for tool execution success/failure.
        
        Args:
            tts_provider: Initialized TTS provider instance
        """
        self._tts_provider = tts_provider
    
    async def _initialize_mcp(self):
        """Initialize MCP connection and discover tools.
        
        Tries to connect to a persistent SSE server first (fast, ~50ms).
        Falls back to spawning a stdio subprocess if SSE fails (~2s).
        """
        # Try SSE first (persistent background server)
        if await self._try_mcp_sse():
            return
        
        # Fall back to stdio subprocess
        if self.mcp_server_path:
            await self._try_mcp_stdio()
    
    async def _try_mcp_sse(self) -> bool:
        """Try to connect to persistent MCP server via SSE."""
        try:
            import time
            start = time.time()
            
            # Try to connect to SSE server
            self._mcp_sse_client = sse_client(self.mcp_sse_url, timeout=2.0)
            read, write = await self._mcp_sse_client.__aenter__()
            self.mcp_session = ClientSession(read, write)
            await self.mcp_session.__aenter__()
            
            # Initialize MCP session
            await self.mcp_session.initialize()
            
            # Discover available tools
            await self._discover_mcp_tools()
            
            elapsed = time.time() - start
            print(f"⚡ Connected to persistent MCP server via SSE ({elapsed:.2f}s)")
            return True
            
        except Exception as e:
            # Clean up partial SSE initialization
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                except Exception:
                    pass
                self.mcp_session = None
            if self._mcp_sse_client:
                try:
                    await self._mcp_sse_client.__aexit__(None, None, None)
                except Exception:
                    pass
                self._mcp_sse_client = None
            
            # SSE failed, will try stdio fallback
            print(f"⚠️  SSE MCP connection failed: {e}")
            print("   Falling back to subprocess mode...")
            return False
    
    async def _try_mcp_stdio(self) -> bool:
        """Fall back to spawning MCP server as subprocess."""
        if not self.mcp_server_path:
            return False
        
        try:
            import time
            start = time.time()
            
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
            
            elapsed = time.time() - start
            print(f"🔧 Started MCP server subprocess ({elapsed:.1f}s)")
            return True
            
        except Exception as e:
            print(f"Warning: MCP initialization failed: {e}")
            # Clean up partial initialization
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                except Exception:
                    pass
                self.mcp_session = None
            if self.stdio_client:
                try:
                    await self.stdio_client.__aexit__(None, None, None)
                except Exception:
                    pass
                self.stdio_client = None
            # Don't fail the entire initialization if MCP fails
            print("Continuing without MCP tools...")
            return False
    
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
        # tools_enabled is True when tools exist (native or signal mode)
        # This prevents streaming deltas since we need to check for tool signals at the end
        tools_enabled = realtime_tools is not None or TOOL_SIGNAL_MODE
        
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
            
            # When TOOL_SIGNAL_MODE is enabled, append tool awareness to instructions
            # This tells realtime what tools exist so it can signal appropriately
            if TOOL_SIGNAL_MODE and realtime_tools:
                tool_summary = self._generate_tool_signal_summary(realtime_tools)
                if tool_summary:
                    effective_instructions = effective_instructions + "\n" + tool_summary
                    print(f"🎯 Tool Signal Mode: Added {len(realtime_tools)} tools to awareness")
                    print(f"   Tool names: {[t.get('name') for t in realtime_tools]}")
                else:
                    print("⚠️ Tool Signal Mode: No tool summary generated")
                print("🎯 Tool Signal Mode enabled - realtime will output text signals")
            elif TOOL_SIGNAL_MODE and not realtime_tools:
                print("⚠️ Tool Signal Mode enabled but NO tools provided!")
            
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
            
            # When TOOL_SIGNAL_MODE is enabled, don't pass tools to realtime
            # Realtime will output text signals, o4-mini handles actual tool execution
            if realtime_tools and not TOOL_SIGNAL_MODE:
                session_config["session"]["tools"] = realtime_tools
                session_config["session"]["tool_choice"] = "auto"
            
            # Log token breakdown for API request
            effective_tools = realtime_tools if not TOOL_SIGNAL_MODE else None
            self._log_api_tokens(
                instructions=effective_instructions,
                messages=conversation_messages,
                tools=effective_tools
            )
            
            await ws.send_json(session_config)
            
            # Wait for session.updated event to ensure tools are configured
            # This prevents the race condition where messages are sent before tools are ready
            session_updated = False
            wait_start = time.time()
            while not session_updated and (time.time() - wait_start) < 2.0:  # 2 second timeout
                try:
                    event = await ws.receive(timeout=0.5)
                    if event.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(event.data)
                        if data.get("type") == "session.updated":
                            session_updated = True
                        elif data.get("type") == "error":
                            print(f"⚠️  Session update error: {data.get('error', {}).get('message', 'Unknown')}")
                            break
                except asyncio.TimeoutError:
                    continue
            
            if not session_updated:
                print("⚠️  Session update not confirmed, proceeding anyway")
            
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
                                # Filter: only ONE google_search per request to prevent redundant searches
                                function_calls = self._filter_duplicate_searches(function_calls)
                                
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
                                    # Note: tools_complete sound will play at end of iterative execution
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
                                    # Simple tool execution without iteration
                                    # Play tools complete sound since no iterative execution
                                    from assistant_framework.utils.audio.tones import beep_tools_complete
                                    beep_tools_complete()
                                    
                                    # Use fast summary instead of API call for composition
                                    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                                    final_text = self._generate_tool_response_summary(list(tool_calls), user_msg)
                                    print(f"📝 Tool response summary: {final_text}")

                                    yield ResponseChunk(
                                        content=final_text,
                                        is_complete=True,
                                        tool_calls=tool_calls,
                                        finish_reason="stop"
                                    )
                            else:
                                # No native function call emitted
                                final_text = "".join(collected_text)
                                fallback_tool_calls = []
                                
                                # Debug: Log the realtime response
                                print("\n🔍 [Response Handler] Realtime response received:")
                                print(f"   Length: {len(final_text)} chars")
                                print(f"   Preview: {final_text[:150]}...")
                                print(f"   TOOL_SIGNAL_MODE: {TOOL_SIGNAL_MODE}")
                                
                                # Check for tool signal (when TOOL_SIGNAL_MODE is enabled)
                                # Simple detection: "TOOL" means hand off to tool subagent
                                stripped = final_text.strip().upper()
                                is_tool_signal = stripped == "TOOL" or stripped.startswith("TOOL_SIGNAL") or stripped.startswith("TOOL:")
                                
                                # Also check for JSON fallback (model didn't follow instructions)
                                is_json_signal = final_text.strip().startswith("{") and final_text.strip().endswith("}")
                                
                                print(f"   Is tool signal: {is_tool_signal}")
                                print(f"   Is JSON fallback: {is_json_signal}")
                                
                                if TOOL_SIGNAL_MODE and (is_tool_signal or is_json_signal):
                                    print("\n🎯 [Response Handler] Tool handoff detected!")
                                    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                                    context = [m for m in messages if m.get("role") in ("user", "assistant")]
                                    
                                    # Hand off to o4-mini for full tool orchestration
                                    print("🎯 [Response Handler] Handing off to o4-mini with full context...")
                                    final_text, fallback_tool_calls = await self._orchestrate_tools(
                                        user_message=user_msg,
                                        context=context,
                                    )
                                    print(f"🎯 [Response Handler] Orchestration complete, {len(fallback_tool_calls)} tool calls")
                                else:
                                    if TOOL_SIGNAL_MODE:
                                        print("ℹ️ [Response Handler] No tool signal - natural response")
                                    # No tool signal - try heuristic fallback for inbox queries
                                    try:
                                        heuristic = ("inbox" in (messages[-1].get("content", "").lower()) or
                                                     "email" in (messages[-1].get("content", "").lower()))
                                        if heuristic and self.available_tools.get("get_notifications"):
                                            args = {"user": PRIMARY_USER, "type_filter": "email", "limit": 10}
                                            tool_call = ToolCall(name="get_notifications", arguments=args)
                                            result = await self.execute_tool("get_notifications", args)
                                            tool_call.result = result
                                            fallback_tool_calls.append(tool_call)
                                            # Use fast summary instead of API call
                                            user_msg = messages[-1].get("content", "")
                                            final_text = self._generate_tool_response_summary(fallback_tool_calls, user_msg)
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
        import time
        from assistant_framework.utils.logging import format_tool_call, format_tool_result, format_tool_error
        from assistant_framework.utils.audio.tones import beep_tool_call
        
        if not self.mcp_session:
            return "Error: MCP not initialized"
        
        try:
            # Play sound to indicate tool is being called
            beep_tool_call()
            
            # Log the tool call with formatted output
            print(format_tool_call(tool_name, arguments))
            
            # Execute and time the tool
            start_time = time.time()
            result = await self.mcp_session.call_tool(tool_name, arguments)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    else:
                        content_parts.append(str(content_item))
                result_text = '\n'.join(content_parts)
                
                # Log formatted result
                print(format_tool_result(tool_name, result_text, execution_time_ms=execution_time_ms))
                
                # Play audio feedback and TTS announcement based on result
                success = self._determine_tool_success(result_text)
                self._play_tool_feedback(result_text)
                self._announce_tool_result(tool_name, success)
                
                return result_text
            else:
                result_text = str(result)
                
                # Log formatted result
                print(format_tool_result(tool_name, result_text, execution_time_ms=execution_time_ms))
                
                # Play audio feedback and TTS announcement based on result
                success = self._determine_tool_success(result_text)
                self._play_tool_feedback(result_text)
                self._announce_tool_result(tool_name, success)
                
                return result_text
                
        except Exception as e:
            import traceback
            from assistant_framework.utils.logging.console_logger import console_log_async
            from assistant_framework.utils.audio.tones import beep_tool_failure
            
            tb_str = traceback.format_exc()
            
            # Log formatted error to local console
            print(format_tool_error(tool_name, str(e), tb_str))
            
            # Log detailed error to webconsole API
            await console_log_async(f"Tool error [{tool_name}]: {e}", "command", is_positive=False)
            
            # Play failure sound and TTS announcement
            beep_tool_failure()
            self._announce_tool_result(tool_name, success=False)
            
            # Return simple error message to model (details in webconsole)
            return f"Error during {tool_name} tool call"
    
    def _determine_tool_success(self, result_text: str) -> bool:
        """
        Determine if a tool execution was successful based on result text.
        
        Args:
            result_text: The result text from tool execution
            
        Returns:
            True if successful, False if failed
        """
        try:
            import json
            
            # Try to parse as JSON to check for success field
            try:
                result_data = json.loads(result_text)
                
                # Check common success indicators
                if isinstance(result_data, dict):
                    success = result_data.get('success')
                    error = result_data.get('error')
                    
                    if success is True or (success is None and not error):
                        return True
                    elif success is False or error:
                        return False
                    else:
                        # Ambiguous result - default to success
                        return True
                else:
                    # Non-dict JSON - assume success
                    return True
                    
            except json.JSONDecodeError:
                # Not JSON - check for error keywords in text
                result_lower = result_text.lower()
                if any(keyword in result_lower for keyword in ['error', 'failed', 'exception', 'not found']):
                    return False
                else:
                    return True
                    
        except Exception:
            # If determination fails, default to success
            return True
    
    def _play_tool_feedback(self, result_text: str) -> None:
        """Play audio feedback based on tool execution result."""
        try:
            from assistant_framework.utils.audio.tones import beep_tool_success, beep_tool_failure
            
            success = self._determine_tool_success(result_text)
            if success:
                beep_tool_success()
            else:
                beep_tool_failure()
                    
        except Exception:
            # If feedback fails, don't propagate - it's non-critical
            pass
    
    def _announce_tool_result(self, tool_name: str, success: bool) -> None:
        """
        Play TTS announcement for tool execution result.
        
        Runs immediately in a separate thread for instant feedback.
        
        Args:
            tool_name: Name of the tool that was executed
            success: Whether the tool execution succeeded
        """
        try:
            from assistant_framework.config import ENABLE_TTS_ANNOUNCEMENTS
            from assistant_framework.utils.audio.tts_announcements import announce_tool_call
            
            if ENABLE_TTS_ANNOUNCEMENTS and self._tts_provider:
                announce_tool_call(self._tts_provider, tool_name, success)
        except Exception:
            # If announcement fails, don't propagate - it's non-critical
            pass
    
    async def _compose_final_answer(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
        tool_calls: List[ToolCall],
        pre_text: str,
        instructions: str = "",
        handoff: Optional[HandoffContext] = None
    ) -> str:
        """
        Integrate tool results into a single assistant reply using OpenAI Chat Completions.
        
        Uses the configurable composition prompt from TOOL_SUBAGENT_CONFIG to maintain
        voice continuity with the primary assistant.
        
        Args:
            user_message: The user's original message
            context: Conversation context
            tool_calls: Executed tool calls with results
            pre_text: Any prefix text to include
            instructions: System instructions (optional)
            handoff: HandoffContext with style/tone guidance
        """
        try:
            # Use the reusable client initialized during setup
            client = self.openai_client
            if not client:
                # Fallback if not initialized
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)

            # Build messages
            messages: List[Dict[str, str]] = []
            
            # Include recent context (truncate to avoid very long prompts)
            for msg in context[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role if role in ("user", "assistant", "system") else "assistant", "content": content})

            messages.append({"role": "user", "content": user_message})

            # Tool results summary
            tool_summaries = []
            for tc in tool_calls:
                if tc and tc.result:
                    tool_summaries.append(f"{tc.name} result:\n{tc.result}")
            tools_block = "\n\n".join(tool_summaries) if tool_summaries else ""
            
            print("🔍 Composing final answer with tool results:")
            print(f"   Tool calls: {len(tool_calls)}")
            print(f"   Tools block: {tools_block[:500]}")

            # Use configurable composition prompt from TOOL_SUBAGENT_CONFIG
            prompt_template = TOOL_SUBAGENT_CONFIG.get("composition_prompt", "")
            voice_description = TOOL_SUBAGENT_CONFIG.get("voice_description", "")
            
            # Extract handoff context values (with defaults from config)
            response_style = handoff.response_style if handoff else TOOL_SUBAGENT_CONFIG.get("default_response_style", "conversational")
            tone = handoff.tone if handoff else TOOL_SUBAGENT_CONFIG.get("default_tone", "casual")
            max_response_length = handoff.max_response_length if handoff else TOOL_SUBAGENT_CONFIG.get("default_max_length", "1-2 sentences")
            
            # Format the composition prompt
            guidance = prompt_template.format(
                voice_description=voice_description,
                response_style=response_style,
                tone=tone,
                max_response_length=max_response_length,
                tools_block=tools_block,
            )
            
            if pre_text.strip():
                guidance = pre_text + "\n\n" + guidance

            messages.append({"role": "system", "content": guidance})

            # Generate final answer
            # Note: o-series models (o4-mini, etc.) require max_completion_tokens instead of max_tokens
            print(f"🔄 Calling composition API with model: {self.tool_subagent_model}")
            result = await client.chat.completions.create(
                model=self.tool_subagent_model,
                messages=messages,
                max_completion_tokens=min(self.max_tokens, 750),
            )
            
            # Extract content - handle both standard and reasoning model responses
            content = ""
            if result and result.choices:
                choice = result.choices[0]
                # Standard response format
                if hasattr(choice, 'message') and choice.message:
                    content = choice.message.content or ""
                # Some models use 'text' directly
                elif hasattr(choice, 'text'):
                    content = choice.text or ""
                    
            print(f"📝 Composition result: {content[:200] if content else '(empty)'}")
            
            # Track token usage from composition API call
            if result and result.usage:
                usage = result.usage
                self._composition_input_tokens += usage.prompt_tokens or 0
                self._composition_output_tokens += usage.completion_tokens or 0
                print(f"📊 Composition API: +{usage.prompt_tokens or 0} in, +{usage.completion_tokens or 0} out")
            
            # If content is still empty, use a simple summary as fallback
            if not content and tool_calls:
                print("⚠️ Composition returned empty, generating fallback summary")
                content = self._generate_simple_summary(tool_calls)
            
            return content or ""
        except Exception as e:
            # Fallback: simple concatenation
            print(f"❌ Composition API failed: {e}")
            import traceback
            traceback.print_exc()
            fallback = pre_text.strip()
            if tool_calls:
                parts = [fallback] if fallback else []
                for tc in tool_calls:
                    if tc and tc.result:
                        parts.append(str(tc.result))
                fallback = "\n\n".join(parts)
            return fallback or f"(Failed to compose final answer: {e})"
    
    def _generate_simple_summary(self, tool_calls: List[ToolCall]) -> str:
        """Generate a simple summary when composition fails or returns empty."""
        if not tool_calls:
            return "Done."
        
        # Check for success indicators in tool results
        successes = []
        for tc in tool_calls:
            if tc and tc.result:
                result_lower = tc.result.lower()
                if '"success":true' in result_lower or '"success": true' in result_lower:
                    # Extract a brief description based on tool name
                    if tc.name == "calendar_data":
                        successes.append("calendar updated")
                    elif tc.name == "send_sms":
                        successes.append("message sent")
                    elif tc.name == "kasa_lighting":
                        successes.append("lights adjusted")
                    elif tc.name == "spotify_control":
                        successes.append("Spotify updated")
                    else:
                        successes.append(f"{tc.name} completed")
        
        if successes:
            return "Done - " + ", ".join(successes) + "."
        
        return "Done."
    
    def _generate_tool_response_summary(
        self,
        tool_calls: List[ToolCall],
        user_message: str,
        handoff: Optional[HandoffContext] = None
    ) -> str:
        """
        Generate a natural, concise response summary from tool results.
        
        This replaces o4-mini composition with fast, deterministic response generation.
        Optimized for voice UX - brief confirmations for actions, concise info for queries.
        
        Args:
            tool_calls: List of executed tool calls with results
            user_message: Original user message (for context)
            handoff: Optional handoff context
        """
        if not tool_calls:
            return "Done."
        
        responses = []
        
        for tc in tool_calls:
            if not tc or not tc.result:
                continue
            
            try:
                # Try to parse as JSON
                result = json.loads(tc.result) if isinstance(tc.result, str) else tc.result
            except (json.JSONDecodeError, TypeError):
                result = {"raw": tc.result}
            
            # Handle different tool types
            if tc.name == "calendar_data":
                response = self._summarize_calendar_result(result, user_message)
                if response:
                    responses.append(response)
            
            elif tc.name == "weather_data":
                response = self._summarize_weather_result(result)
                if response:
                    responses.append(response)
            
            elif tc.name == "send_sms":
                if result.get("success"):
                    responses.append("Message sent.")
                else:
                    responses.append(f"Couldn't send the message: {result.get('error', 'unknown error')}")
            
            elif tc.name == "kasa_lighting":
                if result.get("success"):
                    action = result.get("action", "adjusted")
                    responses.append(f"Lights {action}.")
                else:
                    responses.append("Couldn't adjust the lights.")
            
            elif tc.name == "spotify_control":
                if result.get("success"):
                    action = result.get("action", "updated")
                    responses.append(f"Spotify {action}.")
                else:
                    responses.append("Couldn't control Spotify.")
            
            elif tc.name == "google_search":
                # For search, include a brief summary
                if result.get("success") and result.get("results"):
                    top_result = result["results"][0] if result["results"] else {}
                    snippet = top_result.get("snippet", "")[:200]
                    if snippet:
                        responses.append(snippet)
                    else:
                        responses.append("Found some results.")
            
            elif tc.name == "get_notifications":
                # Summarize notifications
                items = result.get("notifications", result.get("items", []))
                if items:
                    count = len(items)
                    responses.append(f"You have {count} notification{'s' if count != 1 else ''}.")
                else:
                    responses.append("No new notifications.")
            
            else:
                # Generic handling
                if result.get("success"):
                    responses.append(f"{tc.name.replace('_', ' ').title()} completed.")
                elif result.get("error"):
                    responses.append(f"Error: {result.get('error')}")
        
        if responses:
            return " ".join(responses)
        
        # Fallback to simple summary
        return self._generate_simple_summary(tool_calls)
    
    def _summarize_calendar_result(self, result: dict, user_message: str) -> str:
        """Summarize calendar tool results for voice."""
        if not result.get("success", True):
            error = result.get("error", "Unknown error")
            action_required = result.get("action_required")
            if action_required:
                return action_required
            return f"Calendar error: {error}"
        
        # Check if this was a write operation
        results_list = result.get("results", [])
        for r in results_list:
            if r.get("operation") == "create_event":
                title = r.get("event_title", "event")
                date = r.get("event_date", "")
                time = r.get("event_time", "")
                if date and time:
                    return f"Done, I've scheduled {title} for {date} at {time}."
                elif date:
                    return f"Done, I've scheduled {title} for {date}."
                return f"Done, {title} has been scheduled."
        
        # Read operation - summarize events
        all_events = []
        for r in results_list:
            events = r.get("events", [])
            all_events.extend(events)
        
        if not all_events:
            # Check if asking about specific day
            if "today" in user_message.lower():
                return "You don't have anything on your calendar today."
            elif "tomorrow" in user_message.lower():
                return "You don't have anything on your calendar tomorrow."
            return "Your calendar is clear."
        
        # Summarize events for voice
        if len(all_events) == 1:
            event = all_events[0]
            summary = event.get("summary", event.get("title", "an event"))
            start_time = event.get("start_time", "")
            start_date = event.get("start_date", "")
            
            if start_time and start_time != "All Day":
                return f"You have {summary} at {start_time}."
            elif start_date:
                return f"You have {summary} on {start_date}."
            return f"You have {summary}."
        
        # Multiple events
        count = len(all_events)
        first_event = all_events[0]
        first_summary = first_event.get("summary", first_event.get("title", "an event"))
        first_time = first_event.get("start_time", "")
        
        if first_time and first_time != "All Day":
            return f"You have {count} events. Next up is {first_summary} at {first_time}."
        return f"You have {count} events coming up. First is {first_summary}."
    
    def _summarize_weather_result(self, result: dict) -> str:
        """Summarize weather tool results for voice."""
        if not result.get("success", True):
            return "Couldn't get the weather right now."
        
        # Try to extract current conditions
        current = result.get("current", result.get("data", {}))
        if isinstance(current, dict):
            temp = current.get("temperature", current.get("temp"))
            condition = current.get("condition", current.get("description", ""))
            
            if temp is not None and condition:
                return f"It's currently {temp} degrees and {condition}."
            elif temp is not None:
                return f"It's currently {temp} degrees."
        
        # Check for summary field
        if result.get("summary"):
            return result["summary"][:200]
        
        return "Weather data retrieved."

    # =========================================================================
    # TOOL SIGNAL MODE
    # =========================================================================
    
    def _generate_tool_signal_summary(self, tools: List[Dict[str, Any]]) -> str:
        """
        Generate a brief summary of available tools for signal mode.
        
        This gives the realtime model awareness of what tools exist so it can
        determine when to output a TOOL_SIGNAL vs respond naturally.
        
        Args:
            tools: List of tool definitions (OpenAI function format)
            
        Returns:
            Brief summary string of available tools
        """
        if not tools:
            return ""
        
        tool_summaries = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            
            # Truncate description to first sentence or 80 chars
            if desc:
                first_sentence = desc.split(".")[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:77] + "..."
                tool_summaries.append(f"- {name}: {first_sentence}")
            else:
                tool_summaries.append(f"- {name}")
        
        return "\n".join([
            "\nAVAILABLE TOOLS (output TOOL if any of these are needed):",
            *tool_summaries,
        ])
    
    async def _orchestrate_tools(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> tuple:
        """
        Hand off to o4-mini for full tool orchestration.
        
        The realtime model signaled that tools are needed. o4-mini receives
        the full user message and conversation context to decide what tools
        to call and with what parameters.
        
        Args:
            user_message: Original user message
            context: Conversation context
            
        Returns:
            Tuple of (final_text, tool_calls)
        """
        try:
            print("\n" + "="*60)
            print("🎯 [Tool Orchestration] Starting")
            print("="*60)
            print(f"   User message: {user_message[:100]}...")
            print(f"   Context messages: {len(context)}")
            
            # Get the orchestration prompt template
            orchestration_prompt = TOOL_SUBAGENT_CONFIG.get("tool_orchestration_prompt", "")
            if not orchestration_prompt:
                print("⚠️ [Tool Orchestration] No orchestration prompt found in config!")
            
            # Format the prompt with just the user message
            formatted_prompt = orchestration_prompt.format(user_message=user_message)
            print(f"📝 [Tool Orchestration] Prompt length: {len(formatted_prompt)} chars")
            
            # Use o4-mini to determine and execute tool calls
            client = self.openai_client
            if not client:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)
            
            # Build the tools list for o4-mini
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
            print(f"🔧 [Tool Orchestration] Available tools: {[t['function']['name'] for t in tools]}")
            
            # Build messages with conversation history for full context
            messages = [
                {"role": "system", "content": formatted_prompt},
            ]
            
            # Add recent conversation history
            for msg in context[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            
            # Add the current user message
            messages.append({"role": "user", "content": user_message})
            
            # Use gpt-4o-mini for orchestration - fast and reliable with tool calling
            # (o4-mini reasoning model doesn't work well with tool_choice="required")
            orchestration_model = "gpt-4o-mini"
            
            print(f"🔄 [Tool Orchestration] Calling {orchestration_model} with {len(messages)} messages...")
            print("   Tool choice: required")
            
            # Force tool calling
            result = await client.chat.completions.create(
                model=orchestration_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="required",
                max_tokens=500,
            )
            
            # Track token usage
            if result and result.usage:
                usage = result.usage
                self._composition_input_tokens += usage.prompt_tokens or 0
                self._composition_output_tokens += usage.completion_tokens or 0
                print(f"📊 [Tool Orchestration] API: +{usage.prompt_tokens or 0} in, +{usage.completion_tokens or 0} out")
            
            choice = result.choices[0] if result and result.choices else None
            if not choice:
                print("❌ [Tool Orchestration] No choice in API response")
                return "I couldn't process that request.", []
            
            # Execute the tool calls
            if choice.message.tool_calls:
                print(f"🔧 [Tool Orchestration] Executing {len(choice.message.tool_calls)} tool call(s)")
                
                all_tool_calls = []
                for i, tc in enumerate(choice.message.tool_calls):
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError as e:
                        print(f"⚠️ [Tool Orchestration] JSON parse error for {tc.function.name}: {e}")
                        args = {}
                    
                    print(f"🔧 Tool {i+1}: {tc.function.name}")
                    print(f"   Args: {json.dumps(args, indent=2)[:300]}")
                    
                    tool_call = ToolCall(name=tc.function.name, arguments=args)
                    tool_call.result = await self.execute_tool(tc.function.name, args)
                    
                    result_preview = str(tool_call.result)[:200] if tool_call.result else "(empty)"
                    print(f"   Result: {result_preview}...")
                    
                    all_tool_calls.append(tool_call)
                
                # Extract handoff context for voice/style continuity
                handoff = self._extract_handoff_context(user_message, context)
                
                # Check if we need more tools (iterative execution)
                if self.composed_tool_calling_enabled and len(all_tool_calls) > 0:
                    iteration = 1
                    while iteration < self.max_tool_iterations:
                        more_tools = await self._check_for_additional_tools(
                            user_message=user_message,
                            context=context,
                            tool_calls_so_far=all_tool_calls,
                            instructions="",
                            iteration=iteration,
                            max_iterations=self.max_tool_iterations,
                            handoff=handoff,
                        )
                        
                        if not more_tools:
                            print("✅ [Tool Orchestration] No more tools needed")
                            break
                        
                        iteration += 1
                        print(f"🔄 [Tool Orchestration] Iteration {iteration}: {[t['name'] for t in more_tools]}")
                        
                        for fc in more_tools:
                            tool_call = ToolCall(name=fc["name"], arguments=fc.get("arguments", {}))
                            tool_call.result = await self.execute_tool(fc["name"], fc.get("arguments", {}))
                            all_tool_calls.append(tool_call)
                
                # Play tools complete sound
                from assistant_framework.utils.audio.tones import beep_tools_complete
                beep_tools_complete()
                
                # Send tool results back to realtime model for final response
                print("📝 [Tool Orchestration] Sending tool results to realtime for final response...")
                final_text = await self._realtime_compose_response(
                    user_message=user_message,
                    context=context,
                    tool_calls=all_tool_calls,
                )
                
                print("\n" + "="*60)
                print("✅ [Tool Orchestration] Complete")
                print("="*60)
                print(f"   Final response: {final_text[:150]}...")
                print(f"   Total tool calls: {len(all_tool_calls)}")
                
                return final_text, all_tool_calls
            else:
                # Shouldn't happen with tool_choice="required", but handle it
                content = choice.message.content or "I couldn't determine what tool to use."
                print(f"⚠️ [Tool Orchestration] No tool calls returned: {content[:100]}")
                return content, []
                
        except Exception as e:
            print(f"❌ [Tool Orchestration] Failed: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I couldn't complete that request.", []
    
    async def _realtime_compose_response(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
        tool_calls: List[ToolCall],
    ) -> str:
        """
        Send tool results back to the realtime model for natural response generation.
        
        This keeps the response voice consistent with the rest of the conversation
        by having the same realtime model generate the final answer.
        
        Args:
            user_message: Original user message
            context: Conversation context
            tool_calls: Executed tool calls with results
            
        Returns:
            Final response text from realtime model
        """
        try:
            # Build tool results summary for the realtime model
            tool_results = []
            for tc in tool_calls:
                result_str = json.dumps(tc.result) if isinstance(tc.result, dict) else str(tc.result)
                tool_results.append(f"Tool: {tc.name}\nResult: {result_str}")
            
            tools_context = "\n\n".join(tool_results)
            
            # Create a prompt that gives the realtime model the tool results
            compose_message = (
                f"The user asked: {user_message}\n\n"
                f"I executed the following tools:\n\n{tools_context}\n\n"
                "Based on these results, provide a natural, conversational response to the user. "
                "Be concise and direct."
            )
            
            print(f"📝 [Realtime Compose] Sending {len(tool_calls)} tool results to realtime...")
            
            # Get the WebSocket connection
            ws = await self._ensure_ws_connected()
            
            # Send the compose request to realtime
            # Build conversation with tool results as assistant context
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": compose_message
                        }
                    ]
                }
            }
            
            await ws.send_json(conversation_item)
            
            # Request response
            response_create = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                }
            }
            
            await ws.send_json(response_create)
            
            # Collect the response
            collected_text = []
            timeout = 30.0
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                try:
                    event = await ws.receive(timeout=5.0)
                    if event.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(event.data)
                        etype = data.get("type", "")
                        
                        if etype == "response.text.delta":
                            delta = data.get("delta", "")
                            if delta:
                                collected_text.append(delta)
                        
                        elif etype in ("response.done", "response.completed"):
                            break
                        
                        elif etype == "error":
                            err = data.get("error", {}).get("message", "Unknown error")
                            print(f"⚠️ [Realtime Compose] Error: {err}")
                            break
                            
                except asyncio.TimeoutError:
                    print("⚠️ [Realtime Compose] Timeout waiting for response")
                    break
                except Exception as e:
                    print(f"⚠️ [Realtime Compose] Exception: {e}")
                    break
            
            final_text = "".join(collected_text)
            
            if not final_text:
                # Fallback to simple summary if realtime fails
                print("⚠️ [Realtime Compose] No response, using fallback")
                final_text = self._generate_tool_response_summary(tool_calls, user_message)
            
            print(f"📝 [Realtime Compose] Response: {final_text[:100]}...")
            return final_text
            
        except Exception as e:
            print(f"❌ [Realtime Compose] Failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to deterministic summary
            return self._generate_tool_response_summary(tool_calls, user_message)
    
    def _parse_tool_signal(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse a tool signal from the realtime model's response.
        
        Expected format: TOOL_SIGNAL: <intent> | <key_info>
        Also handles fallback JSON format like {"what": "next appointment"}
        
        Args:
            text: The response text from realtime model
            
        Returns:
            Dict with 'tool' and 'context' keys, or None if invalid
        """
        try:
            text = text.strip()
            print(f"🔍 [Signal Parse] Checking text: {text[:100]}...")
            
            # Primary format: TOOL_SIGNAL: intent | details
            if text.startswith("TOOL_SIGNAL:"):
                pass  # Continue to standard parsing below
            # Fallback: JSON format like {"what": "...", "action": "..."}
            elif text.startswith("{") and text.endswith("}"):
                print("🔍 [Signal Parse] Detected JSON fallback format")
                try:
                    data = json.loads(text)
                    # Extract intent from JSON
                    intent = None
                    context = ""
                    
                    # Common JSON patterns the model might output
                    if "what" in data:
                        context = str(data["what"])
                        # Infer intent from context
                        if any(kw in context.lower() for kw in ["calendar", "appointment", "meeting", "schedule", "event"]):
                            intent = "calendar_read"
                        elif any(kw in context.lower() for kw in ["weather", "temperature", "forecast"]):
                            intent = "weather"
                        elif any(kw in context.lower() for kw in ["light", "lamp"]):
                            intent = "lights"
                    elif "action" in data:
                        action = str(data["action"]).lower()
                        context = str(data.get("details", data.get("query", "")))
                        if "calendar" in action or "schedule" in action:
                            intent = "calendar_write" if "create" in action or "add" in action else "calendar_read"
                        elif "weather" in action:
                            intent = "weather"
                        elif "light" in action:
                            intent = "lights"
                        elif "music" in action or "play" in action:
                            intent = "spotify"
                    elif "tool" in data:
                        intent = str(data["tool"]).lower()
                        context = str(data.get("context", data.get("params", "")))
                    
                    if intent:
                        print(f"🔍 [Signal Parse] Converted JSON to intent: {intent} | {context}")
                        # Convert to TOOL_SIGNAL format and continue
                        text = f"TOOL_SIGNAL: {intent} | {context}"
                    else:
                        print("🔍 [Signal Parse] Could not extract intent from JSON")
                        return None
                except json.JSONDecodeError:
                    print("🔍 [Signal Parse] JSON parse failed")
                    return None
            else:
                print("🔍 [Signal Parse] No TOOL_SIGNAL prefix found")
                return None
            
            # Extract the signal part after "TOOL_SIGNAL:"
            signal_part = text[len("TOOL_SIGNAL:"):].strip()
            print(f"🔍 [Signal Parse] Signal part: {signal_part}")
            
            # Split by | to get intent and key info
            if "|" in signal_part:
                parts = signal_part.split("|", 1)
                tool_intent = parts[0].strip().lower()
                key_info = parts[1].strip() if len(parts) > 1 else ""
            else:
                tool_intent = signal_part.strip().lower()
                key_info = ""
            
            print(f"🔍 [Signal Parse] Intent: {tool_intent}, Key info: {key_info}")
            
            # Map common intent names to actual tool names
            tool_mapping = {
                "calendar_read": "calendar_data",
                "calendar_write": "calendar_data",
                "calendar": "calendar_data",
                "weather": "weather_data",
                "lights": "kasa_lighting",
                "lighting": "kasa_lighting",
                "spotify": "spotify_control",
                "music": "spotify_control",
                "sms": "send_sms",
                "text": "send_sms",
                "message": "send_sms",
                "search": "google_search",
                "notifications": "get_notifications",
                "email": "get_notifications",
                "clipboard": "clipboard",
            }
            
            # Determine actual tool name
            actual_tool = tool_mapping.get(tool_intent, tool_intent)
            print(f"🔍 [Signal Parse] Mapped tool: {tool_intent} -> {actual_tool}")
            
            # Determine if it's a read or write for calendar
            is_calendar_write = tool_intent == "calendar_write" or any(
                kw in key_info.lower() for kw in ["schedule", "create", "add", "set", "book"]
            )
            
            parsed_signal = {
                "tool": actual_tool,
                "intent": tool_intent,
                "context": key_info,
                "is_write": is_calendar_write,
            }
            print(f"✅ [Signal Parse] Parsed signal: {parsed_signal}")
            return parsed_signal
        except Exception as e:
            print(f"❌ [Signal Parse] Failed to parse tool signal: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _orchestrate_from_signal(
        self,
        signal: Dict[str, str],
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> tuple:
        """
        Take a tool signal from realtime and orchestrate full tool execution via o4-mini.
        
        This is the core of tool signal mode - o4-mini determines exact parameters
        from the intent signal and user message, then executes the tool(s).
        
        Args:
            signal: Parsed tool signal with 'tool', 'intent', 'context', 'is_write'
            user_message: Original user message
            context: Conversation context
            
        Returns:
            Tuple of (final_text, tool_calls)
        """
        try:
            print("\n" + "="*60)
            print("🎯 [Signal Orchestration] Starting")
            print("="*60)
            print(f"   Signal tool: {signal['tool']}")
            print(f"   Signal intent: {signal['intent']}")
            print(f"   Signal context: {signal['context']}")
            print(f"   Is write operation: {signal.get('is_write', False)}")
            print(f"   User message: {user_message[:100]}...")
            
            # Get the orchestration prompt template
            orchestration_prompt = TOOL_SUBAGENT_CONFIG.get("tool_orchestration_prompt", "")
            if not orchestration_prompt:
                print("⚠️ [Signal Orchestration] No orchestration prompt found in config!")
            
            # Format the prompt with signal info
            formatted_prompt = orchestration_prompt.format(
                user_message=user_message,
                tool_intent=signal["intent"],
                key_info=signal["context"],
            )
            print(f"📝 [Signal Orchestration] Formatted prompt length: {len(formatted_prompt)} chars")
            
            # Use o4-mini to determine exact tool parameters and execute
            client = self.openai_client
            if not client:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.api_key)
            
            # Build the tools list for o4-mini
            # Prioritize the signaled tool but include all tools for flexibility
            signaled_tool_name = signal["tool"]
            tools = []
            signaled_tool_found = False
            
            for func in self.openai_functions:
                tool_entry = {
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters", {"type": "object", "properties": {}})
                    }
                }
                # Check if this is the signaled tool
                if func.get("name") == signaled_tool_name:
                    signaled_tool_found = True
                tools.append(tool_entry)
            
            print(f"🔧 [Signal Orchestration] Available tools: {[t['function']['name'] for t in tools]}")
            print(f"   Signaled tool '{signaled_tool_name}' found: {signaled_tool_found}")
            
            # Build messages with conversation history for context
            messages = [
                {"role": "system", "content": formatted_prompt},
            ]
            
            # Add recent conversation history (last 5 exchanges max)
            for msg in context[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            
            # Add the current user message
            messages.append({"role": "user", "content": user_message})
            
            print(f"🔄 [Signal Orchestration] Calling {self.tool_subagent_model} with {len(messages)} messages...")
            print("   Tool choice: required (forcing tool call)")
            
            # Force tool calling with tool_choice="required"
            result = await client.chat.completions.create(
                model=self.tool_subagent_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="required",  # FORCE tool calling - don't allow text-only response
                max_completion_tokens=500,
            )
            
            # Track token usage
            if result and result.usage:
                usage = result.usage
                self._composition_input_tokens += usage.prompt_tokens or 0
                self._composition_output_tokens += usage.completion_tokens or 0
                print(f"📊 [Signal Orchestration] API tokens: +{usage.prompt_tokens or 0} in, +{usage.completion_tokens or 0} out")
            
            choice = result.choices[0] if result and result.choices else None
            if not choice:
                print("❌ [Signal Orchestration] No choice in API response")
                return "I couldn't process that request.", []
            
            # Check if o4-mini wants to call tools
            if choice.message.tool_calls:
                print(f"🔧 [Signal Orchestration] o4-mini returned {len(choice.message.tool_calls)} tool call(s)")
                
                # Execute the tool calls
                all_tool_calls = []
                
                for i, tc in enumerate(choice.message.tool_calls):
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError as e:
                        print(f"⚠️ [Signal Orchestration] Failed to parse args for {tc.function.name}: {e}")
                        args = {}
                    
                    print(f"🔧 [Signal Orchestration] Tool {i+1}: {tc.function.name}")
                    print(f"   Arguments: {json.dumps(args, indent=2)[:500]}")
                    
                    tool_call = ToolCall(name=tc.function.name, arguments=args)
                    tool_call.result = await self.execute_tool(tc.function.name, args)
                    
                    # Log result summary
                    result_preview = str(tool_call.result)[:300] if tool_call.result else "(empty)"
                    print(f"   Result preview: {result_preview}...")
                    
                    all_tool_calls.append(tool_call)
                
                # Extract handoff context for voice/style continuity
                handoff = self._extract_handoff_context(user_message, context)
                print(f"📋 [Signal Orchestration] Handoff context: tone={handoff.tone}, style={handoff.response_style}")
                
                # Check if we need more tools (iterative execution)
                if self.composed_tool_calling_enabled and len(all_tool_calls) > 0:
                    print(f"🔄 [Signal Orchestration] Checking for additional tools (max {self.max_tool_iterations} iterations)")
                    # Check for additional tools needed
                    iteration = 1
                    while iteration < self.max_tool_iterations:
                        more_tools = await self._check_for_additional_tools(
                            user_message=user_message,
                            context=context,
                            tool_calls_so_far=all_tool_calls,
                            instructions="",
                            iteration=iteration,
                            max_iterations=self.max_tool_iterations,
                            handoff=handoff,
                        )
                        
                        if not more_tools:
                            print(f"✅ [Signal Orchestration] No more tools needed after iteration {iteration}")
                            break
                        
                        iteration += 1
                        print(f"🔄 [Signal Iteration {iteration}] Additional tools: {[t['name'] for t in more_tools]}")
                        
                        for fc in more_tools:
                            tool_call = ToolCall(name=fc["name"], arguments=fc.get("arguments", {}))
                            tool_call.result = await self.execute_tool(fc["name"], fc.get("arguments", {}))
                            all_tool_calls.append(tool_call)
                
                # Play tools complete sound
                from assistant_framework.utils.audio.tones import beep_tools_complete
                beep_tools_complete()
                
                print(f"📝 [Signal Orchestration] Composing final response from {len(all_tool_calls)} tool call(s)")
                
                # Compose final response using LLM with full tool results
                final_text = await self._compose_final_answer(
                    user_message=user_message,
                    context=context,
                    tool_calls=all_tool_calls,
                    pre_text="",
                    instructions="",
                    handoff=handoff,
                )
                
                print("\n" + "="*60)
                print("✅ [Signal Orchestration] Complete")
                print("="*60)
                print(f"   Final response: {final_text[:200]}...")
                print(f"   Total tool calls: {len(all_tool_calls)}")
                
                return final_text, all_tool_calls
            else:
                # o4-mini decided no tool was needed - return its response
                content = choice.message.content or "Done."
                print(f"ℹ️ [Signal Orchestration] o4-mini chose not to call tools, responding: {content[:100]}")
                return content, []
                
        except Exception as e:
            print(f"❌ Signal orchestration failed: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I couldn't complete that request.", []

    # =========================================================================
    # HANDOFF CONTEXT EXTRACTION
    # =========================================================================
    
    def _extract_handoff_context(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> HandoffContext:
        """
        Extract structured handoff context for tool subagents.
        
        Uses heuristics to determine intent, tone, and style without extra API calls.
        This enables voice continuity between the primary provider and subagents.
        
        Args:
            user_message: The user's original message
            context: Conversation history
            
        Returns:
            HandoffContext with extracted intent, tone, style, and relevant context
        """
        intent = self._extract_intent(user_message)
        tone = self._detect_tone(user_message, context)
        style = self._determine_response_style(user_message)
        relevant = self._extract_relevant_context(context, user_message)
        max_length = self._get_length_preference(user_message)
        
        return HandoffContext(
            user_intent=intent,
            response_style=style,
            tone=tone,
            relevant_context=relevant,
            max_response_length=max_length,
        )
    
    def _extract_intent(self, user_message: str) -> str:
        """
        Extract core intent from user message.
        
        Uses the first sentence or imperative clause as the intent summary.
        E.g., "Schedule a meeting with Alex tomorrow and send him a reminder" 
              -> "Schedule a meeting and send a reminder"
        """
        # Clean up the message
        msg = user_message.strip()
        if not msg:
            return "Fulfill user request"
        
        # Try to get the first sentence as the intent
        # Split on sentence-ending punctuation
        import re
        sentences = re.split(r'[.!?]', msg)
        first_sentence = sentences[0].strip() if sentences else msg
        
        # Truncate if too long
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100] + "..."
        
        return first_sentence if first_sentence else "Fulfill user request"
    
    def _detect_tone(self, user_message: str, context: List[Dict[str, Any]]) -> str:
        """
        Detect conversation tone: casual, formal, or urgent.
        
        Checks for urgency indicators, formality markers, and conversation style.
        """
        msg_lower = user_message.lower()
        
        # Urgency indicators
        urgent_markers = ["asap", "urgent", "immediately", "right now", "quick", "hurry", "emergency"]
        if any(marker in msg_lower for marker in urgent_markers):
            return "urgent"
        
        # Formal indicators
        formal_markers = ["please", "kindly", "would you", "could you", "i would like", "i request"]
        formal_count = sum(1 for marker in formal_markers if marker in msg_lower)
        
        # Casual indicators
        casual_markers = ["hey", "yo", "can you", "just", "real quick", "btw", "fyi"]
        casual_count = sum(1 for marker in casual_markers if marker in msg_lower)
        
        # Check context for tone consistency
        if context:
            recent_user_msgs = [m.get("content", "") for m in context[-3:] if m.get("role") == "user"]
            for msg in recent_user_msgs:
                if any(marker in msg.lower() for marker in casual_markers):
                    casual_count += 1
        
        if formal_count > casual_count:
            return "formal"
        
        return "casual"  # Default to casual
    
    def _determine_response_style(self, user_message: str) -> str:
        """
        Determine expected response style: brief, detailed, or conversational.
        
        Based on question complexity and explicit requests.
        """
        msg_lower = user_message.lower()
        
        # Explicit detail requests
        if any(marker in msg_lower for marker in ["explain", "detail", "tell me more", "elaborate", "in depth"]):
            return "detailed"
        
        # Brief response indicators
        if any(marker in msg_lower for marker in ["quick", "brief", "short", "just tell me", "simple"]):
            return "brief"
        
        # Question complexity heuristics
        word_count = len(user_message.split())
        
        # Simple commands/queries -> brief
        if word_count < 8:
            return "brief"
        
        # Complex multi-part requests -> detailed
        if any(connector in msg_lower for connector in [" and ", " then ", " also ", " after that"]):
            return "detailed"
        
        return "conversational"  # Default
    
    def _extract_relevant_context(self, context: List[Dict[str, Any]], user_message: str) -> List[str]:
        """
        Extract contextually relevant facts from conversation history.
        
        Pulls entity mentions, prior decisions, and relevant tool results.
        """
        relevant = []
        
        if not context:
            return relevant
        
        # Extract key nouns/entities from user message for matching
        msg_lower = user_message.lower()
        
        # Look at recent messages for relevant context
        for msg in context[-6:]:  # Last 6 messages
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if not content:
                continue
            
            # Include assistant responses that might have relevant info
            if role == "assistant" and len(content) > 20:
                # Check if it contains tool results or relevant info
                if any(indicator in content.lower() for indicator in ["scheduled", "created", "found", "sent", "result"]):
                    # Truncate to first 150 chars
                    snippet = content[:150] + "..." if len(content) > 150 else content
                    relevant.append(f"Prior: {snippet}")
            
            # Include user messages that provide context
            elif role == "user" and msg.get("content") != user_message:
                # Only include if it seems related (shares words)
                user_words = set(msg_lower.split())
                content_words = set(content.lower().split())
                if len(user_words & content_words) >= 2:  # At least 2 shared words
                    snippet = content[:100] + "..." if len(content) > 100 else content
                    relevant.append(f"User said: {snippet}")
        
        return relevant[:3]  # Limit to 3 most relevant items
    
    def _get_length_preference(self, user_message: str) -> str:
        """
        Determine preferred response length based on query type.
        """
        msg_lower = user_message.lower()
        
        # Explicit length requests
        if any(marker in msg_lower for marker in ["detailed", "full", "everything", "all about"]):
            return "2-3 paragraphs"
        
        if any(marker in msg_lower for marker in ["brief", "quick", "short", "just"]):
            return "1 sentence"
        
        # Question type heuristics
        if msg_lower.startswith(("what is", "who is", "when is", "where is")):
            return "1-2 sentences"
        
        if msg_lower.startswith(("how do", "how can", "explain", "why")):
            return "paragraph"
        
        # Default for tool operations
        return "1-2 sentences"

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
        1. Extract handoff context (intent, tone, style)
        2. Execute initial tools
        3. Review results and decide if more tools are needed
        4. Continue until the task is complete or max iterations reached
        5. Compose final answer using handoff context for voice continuity
        
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
        
        # Extract handoff context for subagent communication
        # This happens once at the start and is passed to all subagent calls
        handoff = self._extract_handoff_context(user_message, context)
        print(f"📋 Handoff context: intent='{handoff.user_intent[:50]}...', tone={handoff.tone}, style={handoff.response_style}")
        
        print(f"🔄 [Iteration {iteration}/{self.max_tool_iterations}] Initial tools executed: {[tc.name for tc in initial_tool_calls]}")
        
        # Check if we should try for more tools
        while iteration < self.max_tool_iterations:
            # Ask the AI if more tools are needed given the current results
            more_tools = await self._check_for_additional_tools(
                user_message=user_message,
                context=context,
                tool_calls_so_far=all_tool_calls,
                instructions=instructions,
                iteration=iteration,
                max_iterations=self.max_tool_iterations,
                handoff=handoff,
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
        
        # Play tools complete sound - all tool iterations are done
        from assistant_framework.utils.audio.tones import beep_tools_complete
        beep_tools_complete()
        
        # Generate a simple, fast summary instead of calling o4-mini for composition
        # This is better for voice UX - concise confirmations are preferred
        final_text = self._generate_tool_response_summary(all_tool_calls, user_message, handoff)
        print(f"📝 Tool response summary: {final_text}")
        
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
                is_write = any(self._is_calendar_write_command(cmd) for cmd in commands)
                
                if is_write:
                    if calendar_write_seen:
                        print(f"🚫 Blocking duplicate calendar write operation to prevent event duplication")
                        continue
                    calendar_write_seen = True
            
            filtered.append(fc)
        
        return filtered
    
    def _is_calendar_write_command(self, cmd: Dict[str, Any]) -> bool:
        """
        Check if a calendar command is a write/create operation.
        
        Handles various formats the model might use:
        - read_or_write: "write" or "create_event"
        - action: "write" or "create_event"  
        - command: "add", "create", "write"
        - Presence of event_title or nested event.title
        """
        # Explicit write indicators
        if cmd.get("read_or_write") in ("write", "create_event"):
            return True
        if cmd.get("action") in ("write", "create_event", "add", "create"):
            return True
        if cmd.get("command") in ("add", "create", "write", "create_event"):
            return True
        
        # Presence of event creation fields
        if cmd.get("event_title"):
            return True
        
        # Nested event object with title
        event_obj = cmd.get("event")
        if isinstance(event_obj, dict) and event_obj.get("title"):
            return True
        
        return False
    
    def _has_calendar_write(self, tool_calls: List[ToolCall]) -> bool:
        """Check if any of the tool calls include a calendar write/create_event operation."""
        for tc in tool_calls:
            if tc and tc.name == "calendar_data":
                args = tc.arguments or {}
                commands = args.get("commands", [])
                for cmd in commands:
                    if self._is_calendar_write_command(cmd):
                        return True
        return False
    
    def _filter_duplicate_searches(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out duplicate google_search calls.
        Only ONE google_search call is allowed per user request.
        The first search is kept; subsequent searches are blocked.
        
        Args:
            function_calls: List of function calls with 'name' and 'arguments' keys
            
        Returns:
            Filtered list with only ONE google_search allowed
        """
        filtered = []
        search_seen = False
        
        for fc in function_calls:
            name = fc.get("name", "")
            
            if name == "google_search":
                if search_seen:
                    print(f"🚫 Blocking duplicate google_search - only one search per request allowed")
                    continue
                search_seen = True
            
            filtered.append(fc)
        
        return filtered
    
    def _has_google_search(self, tool_calls: List[ToolCall]) -> bool:
        """Check if any of the tool calls include a google_search operation."""
        for tc in tool_calls:
            if tc and tc.name == "google_search":
                return True
        return False
    
    async def _check_for_additional_tools(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
        tool_calls_so_far: List[ToolCall],
        instructions: str = "",
        iteration: int = 1,
        max_iterations: int = 5,
        handoff: Optional[HandoffContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Ask the AI if additional tools are needed to complete the user's request.
        
        Args:
            user_message: The user's original message
            context: Conversation context
            tool_calls_so_far: Tools already executed
            instructions: System instructions (optional)
            iteration: Current iteration number (1-indexed)
            max_iterations: Maximum allowed iterations
            handoff: HandoffContext with extracted intent and relevant context
        
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
            
            # Build iteration status message
            iteration_note = f"ITERATION: {iteration}/{max_iterations}"
            if iteration >= max_iterations - 1:
                iteration_note += " - APPROACHING LIMIT, prioritize completing the task now"
            
            # Use configurable prompt from TOOL_SUBAGENT_CONFIG
            prompt_template = TOOL_SUBAGENT_CONFIG.get("tool_decision_prompt", "")
            
            # Extract handoff context values (with defaults)
            user_intent = handoff.user_intent if handoff else user_message[:100]
            relevant_context = "\n".join(handoff.relevant_context) if handoff and handoff.relevant_context else "None"
            
            # Format the prompt with placeholders
            system_content = prompt_template.format(
                user_intent=user_intent,
                relevant_context=relevant_context,
                iteration_note=iteration_note,
                success_note=success_note,
                tools_summary=tools_summary,
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
            # Note: o-series models (o4-mini, etc.) require max_completion_tokens instead of max_tokens
            result = await client.chat.completions.create(
                model=self.tool_subagent_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto",
                max_completion_tokens=2480,
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
                # Check if a google_search was already executed - block any further searches
                search_already_done = self._has_google_search(tool_calls_so_far)
                
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
                        is_write = any(self._is_calendar_write_command(cmd) for cmd in commands)
                        if is_write:
                            print(f"🚫 Blocking calendar write - event already created in this request")
                            continue
                    
                    # Block additional google_search if one was already executed
                    if tc.function.name == "google_search" and search_already_done:
                        print(f"🚫 Blocking google_search - search already completed in this request")
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