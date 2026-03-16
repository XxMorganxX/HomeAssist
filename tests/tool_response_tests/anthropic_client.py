"""
Anthropic (Claude) client for tool testing.

This is a standalone test client that uses Claude Haiku 4 for tool orchestration.
It is isolated to the test suite and does not affect the main assistant.

The client:
- Accepts the same context format as the OpenAI tests
- Calls Claude Haiku 4 with MCP tool definitions (converted from OpenAI format)
- Executes tools via the provided execute_tool function
- Returns the same output format (response text and ToolCall list)
"""

import os
import json
import time
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass

# Anthropic SDK
try:
    import anthropic
except ImportError:
    raise ImportError("anthropic package required. Install with: pip install anthropic>=0.40.0")


@dataclass
class ToolCall:
    """Represents a tool/function call (matches data_models.ToolCall)."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'arguments': self.arguments,
            'call_id': self.call_id,
            'result': self.result
        }


class AnthropicToolClient:
    """
    A test-only client for running tool orchestration through Claude Haiku 4.
    
    This client is designed to match the same input/output interface as the
    OpenAI-based tool orchestration in the main assistant, allowing for
    side-by-side comparison in the test UI.
    """
    
    # Claude Haiku 4 model identifier
    MODEL = "claude-3-5-haiku-latest"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Token tracking
        self.input_tokens = 0
        self.output_tokens = 0
    
    def convert_openai_tools_to_anthropic(self, openai_tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI tool format to Anthropic tool format.
        
        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "...",
                "parameters": {...}
            }
        }
        
        Anthropic format:
        {
            "name": "weather",
            "description": "...",
            "input_schema": {...}
        }
        """
        anthropic_tools = []
        
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tool = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                }
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def _build_system_prompt(self, base_system: str = "") -> str:
        """Build the system prompt for tool orchestration."""
        default_system = """You are a helpful assistant that can use tools to help users.
When a user makes a request that requires tools, analyze which tool(s) are needed and call them.
After executing tools, synthesize the results into a natural, conversational response.

IMPORTANT:
- Call tools when needed to fulfill the user's request
- You may call multiple tools if the request requires it
- After tool results are available, provide a helpful response based on those results
- Be concise and natural in your responses"""
        
        if base_system:
            return f"{base_system}\n\n{default_system}"
        return default_system
    
    async def run_prompt(
        self,
        prompt: str,
        tools: List[Dict],
        execute_tool_fn: Callable,
        context: List[Dict] = None,
        system_prompt: str = "",
        max_iterations: int = 5,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Run a prompt through Claude with tool support.
        
        Args:
            prompt: The user's prompt/request
            tools: List of tools in OpenAI format (will be converted to Anthropic format)
            execute_tool_fn: Async function to execute tools: (name, args) -> result_str
            context: Optional conversation context (list of messages)
            system_prompt: Optional system prompt to prepend
            max_iterations: Maximum tool call iterations
        
        Returns:
            Tuple of (final_response_text, list_of_tool_calls)
        """
        # Convert tools to Anthropic format
        anthropic_tools = self.convert_openai_tools_to_anthropic(tools)
        
        # Build messages
        messages = []
        
        # Add context if provided
        if context:
            for msg in context[-10:]:  # Last 10 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
        
        # Add the current user message
        messages.append({"role": "user", "content": prompt})
        
        # Build system prompt
        system = self._build_system_prompt(system_prompt)
        
        all_tool_calls = []
        iteration = 0
        
        print(f"🔄 [Anthropic] Starting orchestration with {self.MODEL}")
        print(f"   Available tools: {[t['name'] for t in anthropic_tools]}")
        
        while iteration < max_iterations:
            iteration += 1
            start_time = time.time()
            
            try:
                # Call Claude API
                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=1024,
                    system=system,
                    tools=anthropic_tools if anthropic_tools else None,
                    messages=messages,
                )
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                # Track tokens
                if response.usage:
                    self.input_tokens += response.usage.input_tokens
                    self.output_tokens += response.usage.output_tokens
                    print(f"📊 [Anthropic] API tokens: +{response.usage.input_tokens} in, +{response.usage.output_tokens} out ({elapsed_ms}ms)")
                
                # Process response content blocks
                text_content = ""
                tool_use_blocks = []
                
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)
                
                # If no tool calls, we're done
                if not tool_use_blocks:
                    print(f"✅ [Anthropic] Response complete (no tool calls)")
                    return text_content, all_tool_calls
                
                # Process tool calls
                print(f"🔧 [Anthropic] Iteration {iteration}: {len(tool_use_blocks)} tool call(s)")
                
                tool_results = []
                
                for tool_block in tool_use_blocks:
                    tool_name = tool_block.name
                    tool_input = tool_block.input
                    tool_id = tool_block.id
                    
                    print(f"🔧 [Anthropic] Tool: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_input, indent=2)[:500]}")
                    
                    # Execute the tool
                    try:
                        tool_result = await execute_tool_fn(tool_name, tool_input)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    
                    # Create ToolCall record
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=tool_input,
                        call_id=tool_id,
                        result=tool_result
                    )
                    all_tool_calls.append(tool_call)
                    
                    # Log result preview
                    result_preview = str(tool_result)[:300] if tool_result else "(empty)"
                    print(f"   Result preview: {result_preview}...")
                    
                    # Build tool result for Claude
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": str(tool_result) if tool_result else ""
                    })
                
                # Add assistant message with tool use
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool results
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Claude decided to stop after this
                    if text_content:
                        return text_content, all_tool_calls
                
            except anthropic.APIError as e:
                print(f"❌ [Anthropic] API error: {e}")
                return f"Error: {str(e)}", all_tool_calls
        
        # Max iterations reached - compose final response
        print(f"⚠️ [Anthropic] Max iterations ({max_iterations}) reached")
        
        # Make one final call to get the response
        try:
            final_response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            
            if final_response.usage:
                self.input_tokens += final_response.usage.input_tokens
                self.output_tokens += final_response.usage.output_tokens
            
            final_text = ""
            for block in final_response.content:
                if block.type == "text":
                    final_text += block.text
            
            return final_text, all_tool_calls
            
        except Exception as e:
            print(f"❌ [Anthropic] Final response error: {e}")
            return "I completed the requested actions.", all_tool_calls
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage for this session."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens
        }
    
    def reset_token_usage(self):
        """Reset token counters."""
        self.input_tokens = 0
        self.output_tokens = 0
