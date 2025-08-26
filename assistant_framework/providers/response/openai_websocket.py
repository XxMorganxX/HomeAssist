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
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.response import ResponseInterface
    from models.data_models import ResponseChunk, ToolCall


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
        self.system_prompt = config.get('system_prompt', '')
        
        # MCP configuration
        self.mcp_server_path = config.get('mcp_server_path')
        self.mcp_venv_python = config.get('mcp_venv_python')
        
        # State management
        self.mcp_session = None
        self.available_tools = {}
        self.openai_functions = []
        self.stdio_client = None
        
    async def initialize(self) -> bool:
        """Initialize the OpenAI WebSocket provider and MCP connection."""
        try:
            # Initialize MCP if configured
            if self.mcp_server_path:
                await self._initialize_mcp()
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
                except:
                    pass
                self.mcp_session = None
            if self.stdio_client:
                try:
                    await self.stdio_client.__aexit__(None, None, None)
                except:
                    pass
                self.stdio_client = None
            # Don't fail the entire initialization if MCP fails
            print("Continuing without MCP tools...")
    
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
                    parameters["properties"] = schema["properties"]
                if "required" in schema:
                    parameters["required"] = schema["required"]
        
        return {
            "type": "function",
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
            "parameters": parameters,
        }
    
    async def stream_response(self, 
                            message: str, 
                            context: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[ResponseChunk]:
        """Stream a response for the given message with optional context."""
        # Prepare messages
        messages = context if context else []
        messages.append({"role": "user", "content": message})
        
        # Check if home-related for tool inclusion
        tools = self._should_include_tools(message) if self.openai_functions else None
        
        # Perform WebSocket roundtrip
        async for chunk in self._ws_stream_roundtrip(messages, tools):
            yield chunk
    
    def _should_include_tools(self, message: str) -> Optional[List[Dict[str, Any]]]:
        """Determine if tools should be included based on message content."""
        home_keywords = [
            'light', 'lights', 'thermostat', 'temperature', 'calendar', 'notification',
            'spotify', 'music', 'home', 'house', 'smart', 'device', 'weather',
            'turn on', 'turn off', 'set', 'check', 'show', 'play', 'stop'
        ]
        
        message_lower = message.lower()
        is_home_related = any(keyword in message_lower for keyword in home_keywords)
        
        return self.openai_functions if is_home_related else None
    
    async def _ws_stream_roundtrip(self,
                                  messages: List[Dict[str, Any]],
                                  tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[ResponseChunk]:
        """Perform WebSocket roundtrip with streaming response."""
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        
        # Extract non-system messages
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Prepare realtime tools if provided
        realtime_tools = None
        if tools:
            realtime_tools = [self._flatten_tool_schema(t) for t in tools]
        tools_enabled = realtime_tools is not None
        
        collected_text = []
        function_calls = []
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url,
                headers=headers,
                heartbeat=20,
                protocols=["openai-realtime-v1"],
                timeout=aiohttp.ClientTimeout(total=30),
            ) as ws:
                # Configure session
                session_config = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "instructions": self.system_prompt,
                        "voice": "alloy",
                        "temperature": 0.8,
                    }
                }
                
                if realtime_tools:
                    session_config["session"]["tools"] = realtime_tools
                    session_config["session"]["tool_choice"] = "auto"
                
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
                                    tool_calls = []
                                    for fc in function_calls:
                                        try:
                                            args = json.loads(fc["arguments"])
                                        except Exception:
                                            args = {}
                                        tool_call = ToolCall(name=fc["name"], arguments=args)
                                        result = await self.execute_tool(fc["name"], args)
                                        tool_call.result = result
                                        tool_calls.append(tool_call)

                                    # Compose final user-facing answer that integrates tool results
                                    final_text = await self._compose_final_answer(
                                        user_message=next((m.get("content", "") for m in messages if m.get("role") == "user"), ""),
                                        context=[m for m in messages if m.get("role") in ("user", "assistant")],
                                        tool_calls=tool_calls,
                                        pre_text=""  # ignore pre-text to avoid duplication
                                    )

                                    yield ResponseChunk(
                                        content=final_text,
                                        is_complete=True,
                                        tool_calls=tool_calls,
                                        finish_reason="stop"
                                    )
                                else:
                                    # Yield final chunk
                                    yield ResponseChunk(
                                        content="".join(collected_text),
                                        is_complete=True,
                                        finish_reason="stop"
                                    )
                                break
                            
                            elif etype == "error":
                                err = data.get("error", {})
                                message = err.get("message", "Unknown error")
                                raise Exception(f"OpenAI WebSocket error: {message}")
                                
                        except json.JSONDecodeError:
                            continue
                    
                    elif event.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
    
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
            result = await self.mcp_session.call_tool(tool_name, arguments)
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    else:
                        content_parts.append(str(content_item))
                return '\n'.join(content_parts)
            else:
                return str(result)
                
        except Exception as e:
            return f"Error calling tool {tool_name}: {e}"
    
    async def _compose_final_answer(self,
                                    user_message: str,
                                    context: List[Dict[str, Any]],
                                    tool_calls: List[ToolCall],
                                    pre_text: str) -> str:
        """Integrate tool results into a single assistant reply using OpenAI Chat Completions."""
        try:
            # Prefer lightweight model for composition
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)

            # Build messages
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
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
                temperature=0.5,
                max_tokens=min(self.max_tokens, 800),
            )
            content = result.choices[0].message.content if result and result.choices else ""
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

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up MCP session first
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
            except Exception as e:
                print(f"MCP session cleanup error: {e}")
            finally:
                self.mcp_session = None
        
        # Then clean up stdio client
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