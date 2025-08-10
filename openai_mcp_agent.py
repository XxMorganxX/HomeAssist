#!/usr/bin/env python3
"""
OpenAI-MCP Integration Agent

A single file that allows OpenAI to interact with your MCP server tools
through function calling. Connects to your stdio MCP server and provides
natural language interface to all smart home tools.

Usage: python openai_mcp_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from dotenv import load_dotenv
from context import Context
import config

# Load environment variables
load_dotenv()

# Initialize OpenAI client
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# MCP Server configuration
MCP_SERVER_PATH = Path(__file__).parent.parent / "RasPi_Smart_Home" / "mcp_server"
MCP_SERVER_SCRIPT = MCP_SERVER_PATH / "server.py"
MCP_VENV_PYTHON = Path(__file__).parent.parent / "RasPi_Smart_Home" / "venv" / "bin" / "python"

class OpenAIMCPAgent:
    """Agent that connects OpenAI to MCP server tools via function calling."""
    
    def __init__(self):
        self.mcp_session = None
        self.available_tools = {}
        self.openai_functions = []
        
    async def initialize(self):
        """Initialize MCP connection and discover tools."""
        # Determine Python executable
        python_cmd = str(MCP_VENV_PYTHON) if MCP_VENV_PYTHON.exists() else sys.executable
        
        # Server parameters for stdio transport
        server_params = StdioServerParameters(
            command=python_cmd,
            args=[str(MCP_SERVER_SCRIPT), "--transport", "stdio"],
            env=os.environ.copy()
        )
        
        try:
            # Connect to MCP server
            self.stdio_client = stdio_client(server_params)
            self.read, self.write = await self.stdio_client.__aenter__()
            self.mcp_session = ClientSession(self.read, self.write)
            await self.mcp_session.__aenter__()
            
            # Initialize MCP session
            await self.mcp_session.initialize()
            
            # Discover available tools
            await self.discover_tools()
            
            print("✅ Connected to MCP server")
            print(f"📚 Discovered {len(self.available_tools)} tools: {list(self.available_tools.keys())}")
            
            # Debug: Print first tool schema to check format
            if self.openai_functions:
                print(f"[DEBUG] First tool schema: {json.dumps(self.openai_functions[0], indent=2)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            return False
    
    async def discover_tools(self):
        """Discover and catalog all available MCP tools."""
        try:
            tools_result = await self.mcp_session.list_tools()
            
            for tool in tools_result.tools:
                self.available_tools[tool.name] = tool
                
                # Convert MCP tool to OpenAI function format
                openai_function = self.mcp_tool_to_openai_function(tool)
                self.openai_functions.append(openai_function)
                
        except Exception as e:
            print(f"Error discovering tools: {e}")
    
    def mcp_tool_to_openai_function(self, mcp_tool) -> Dict[str, Any]:
        """Convert MCP tool definition to OpenAI function calling format."""
        
        # Extract parameters from MCP tool schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Parse MCP tool schema if available
        if hasattr(mcp_tool, 'inputSchema') and mcp_tool.inputSchema:
            schema = mcp_tool.inputSchema
            if isinstance(schema, dict):
                if "properties" in schema:
                    parameters["properties"] = schema["properties"]
                if "required" in schema:
                    parameters["required"] = schema["required"]
        
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
                "parameters": parameters
            }
        }
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result."""
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
    
    async def chat(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Process user message with OpenAI and execute any tool calls."""
        
        if conversation_history is None:
            conversation_history = []
        
        # Add user message to conversation
        messages = conversation_history + [
            {
                "role": "user", 
                "content": user_message
            }
        ]
        
        if config.DEBUG_MESSAGE_BEING_SENT:
            print(f"[DEBUG] Messages being sent: {json.dumps(messages, indent=2)}")
        
        try:
            # Build request parameters
            # Increased max_completion_tokens to account for reasoning tokens
            request_params = {
                "model": config.RESPONSE_MODEL,
                "messages": messages,
                "max_completion_tokens": config.MAX_COMPLETION_TOKENS  # Increased from 700
            }
            
            # Only add tools if we have any
            if self.openai_functions:
                request_params["tools"] = self.openai_functions
                request_params["tool_choice"] = "auto"
                print(f"[DEBUG] Sending request with {len(self.openai_functions)} tools")
            else:
                print(f"[DEBUG] No tools available, sending basic request")
            
            # Call OpenAI
            if config.DEBUG_MESSAGE_BEING_SENT:
                print(f"[DEBUG] Request params: {json.dumps(request_params, indent=2, default=str)}")
            response = client.chat.completions.create(**request_params)
            if config.DEBUG_RAW_RESPONSE:
                print(f"[DEBUG] Raw response object: {response}")
            
            if config.DEBUG_MESSAGE_CHOICES:
                print(f"[DEBUG] Response choices: {response.choices}")
            
            response_message = response.choices[0].message
            
            # Debug: Check what we got back
            if not response_message.content and not response_message.tool_calls:
                print(f"[DEBUG] Empty response from OpenAI. Message: {response_message}")
                print("[DEBUG] Retrying without tools...")
                
                # Retry without tools
                simple_params = {
                    "model": config.RESPONSE_MODEL,
                    "messages": messages,
                    "max_completion_tokens": config.MAX_COMPLETION_TOKENS  # Increased from 700
                }
                retry_response = client.chat.completions.create(**simple_params)
                response_message = retry_response.choices[0].message
                print(f"[DEBUG] Retry response: {response_message}")
                
            # Check if OpenAI wants to call any tools
            if response_message.tool_calls:
                # Add OpenAI's message with tool calls to conversation
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in response_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Calling {function_name} with args: {function_args}")
                    
                    # Call the MCP tool
                    tool_result = await self.call_mcp_tool(function_name, function_args)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get OpenAI's final response after tool execution
                final_response = client.chat.completions.create(
                    model=config.RESPONSE_MODEL,
                    messages=messages,
                    max_completion_tokens=config.MAX_COMPLETION_TOKENS
                )
                
                return final_response.choices[0].message.content
            
            else:
                # No tools needed, return OpenAI's direct response
                return response_message.content or "I apologize, but I didn't generate a response. Please try again."
                
        except Exception as e:
            return f"Error processing request: {e}"
    
    async def cleanup(self):
        """Clean up MCP connection."""
        try:
            if self.mcp_session:
                await self.mcp_session.__aexit__(None, None, None)
            if hasattr(self, 'stdio_client'):
                await self.stdio_client.__aexit__(None, None, None)
        except Exception as e:
            print(f"Cleanup error: {e}")

async def main():
    """Main interactive loop."""
    print("🤖 OpenAI-MCP Smart Home Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = OpenAIMCPAgent()
    if not await agent.initialize():
        return
    
    print("\n💬 Chat with your smart home! Type 'exit' to quit.")
    print("Examples:")
    print("  - Check my notifications")
    print("  - Turn on the living room lights")
    print("  - What's on my calendar today?")
    print("  - Play music on Spotify")
    print()
    
    # Initialize context manager - it handles system prompt and conversation history
    contextManager = Context(user_input="", conversation_history=[])
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            if not user_input:
                continue
            
            print("🤔 Thinking...")
            
            # Update context with current user input
            contextManager.user_input = user_input
            
            # Get response from agent using context-managed history
            response = await agent.chat(user_input, contextManager.get_conversation_history())
            
            print(f"Assistant: {response}\n")
            
            # Update conversation history through context manager
            contextManager.add_to_conversation_history([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
            
            # Trim history to prevent context window issues
            history = contextManager.get_conversation_history()
            print("--------------------------------")
            print(f"History: {history}")
            print("--------------------------------")
            if len(history) > 21:  # system + 10 exchanges
                # Keep system message and last 20 messages
                contextManager.conversation_history = [history[0]] + history[-20:]
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")