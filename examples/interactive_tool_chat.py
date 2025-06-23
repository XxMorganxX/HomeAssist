"""
Simple ChatGPT chatbot with automatic access to all MCP tools.
Uses OpenAI's function calling to handle everything automatically.
"""

import warnings
# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import sys
import os
import json
import logging
from pathlib import Path

# ============ CONFIGURATION ============
VERBOSE_LOGGING = False  # Set to True to see API calls and detailed logs
# =======================================

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from mcp_server.server import MCPServer
from dotenv import load_dotenv

load_dotenv()

# Configure logging based on VERBOSE_LOGGING setting
if not VERBOSE_LOGGING:
    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp_server").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)
else:
    # Keep all logs visible
    logging.basicConfig(level=logging.INFO)


def main():
    
    # Initialize
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    mcp_server = MCPServer()
    
    # Convert MCP tools to OpenAI functions
    functions = []
    for tool in mcp_server.list_tools():
        functions.append({
            "name": tool['name'],
            "description": mcp_server.get_tool_info(tool['name'])['description'],
            "parameters": tool['schema']
        })
    
    # Chat history
    messages = [{
        "role": "system", 
        "content": "You are a helpful assistant with access to various tools. Use them when appropriate to help the user."
    }]
    
    print("="*80)
    print("ChatGPT with Tools")
    print(f"Available: {', '.join([f['name'] for f in functions])}")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Call ChatGPT with functions
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        
        message = response.choices[0].message
        
        # If ChatGPT wants to use a tool
        if message.function_call:
            func_name = message.function_call.name
            func_args = json.loads(message.function_call.arguments)
            
            print(f"[Using tool: {func_name}]")
            
            # Execute tool
            result = mcp_server.execute_tool(func_name, func_args)
            
            # Add to conversation
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": {"name": func_name, "arguments": message.function_call.arguments}
            })
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps(result)
            })
            
            # Get final response
            final = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            response_text = final.choices[0].message.content
            messages.append({"role": "assistant", "content": response_text})
            print(f"Assistant: {response_text}\n")
        else:
            # Regular response
            messages.append({"role": "assistant", "content": message.content})
            print(f"Assistant: {message.content}\n")


if __name__ == "__main__":
    main()