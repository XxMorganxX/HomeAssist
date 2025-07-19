#!/usr/bin/env python3
"""
Standalone MCP Server with Terminal Client
Minimal implementation for running MCP tools via terminal interface.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from mcp_server.server import MCPServer
    from core.streaming_chatbot import ToolEnabledStreamingChatbot
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class StandaloneMCPServer:
    """Standalone MCP server with terminal client interface."""
    
    def __init__(self):
        """Initialize the standalone MCP server."""
        self.mcp_server = None
        self.chatbot = None
        
    def initialize(self) -> bool:
        """Initialize the MCP server and chatbot."""
        try:
            print("🔧 Initializing MCP server...")
            self.mcp_server = MCPServer()
            
            # Print MCP server status
            status = self.mcp_server.get_server_status()
            print(f"   Tools available: {status['tools_available']}")
            print(f"   Tool names: {', '.join(status['tool_names'])}")
            
            print("🔧 Initializing chatbot...")
            self.chatbot = ToolEnabledStreamingChatbot()
            
            print("✅ MCP server and chatbot initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def run_terminal_mode(self):
        """Run the terminal client interface."""
        if not self.mcp_server or not self.chatbot:
            print("❌ MCP server or chatbot not available")
            return
        
        print("\n🧪 Standalone MCP Server - Terminal Client")
        print("=" * 50)
        
        # Get available tools
        status = self.mcp_server.get_server_status()
        tools = status['tool_names']
        
        print(f"Available tools: {', '.join(tools)}")
        print("\nExamples:")
        print("  - 'turn on the living room lights'")
        print("  - 'play some music on spotify'")
        print("  - 'set the lighting scene to movie'")
        print("  - 'what's the current spotify user?'")
        print("  - 'pause the music'")
        print("\nType 'quit', 'exit', or 'q' to exit")
        print("Type 'tools' to see available tools again")
        print("Type 'status' to see system status")
        print("Press Ctrl+C to exit at any time")
        print("-" * 50)
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Exiting terminal mode")
                        break
                    elif user_input.lower() == 'tools':
                        print(f"Available tools: {', '.join(tools)}")
                        continue
                    elif user_input.lower() == 'status':
                        status = self.mcp_server.get_server_status()
                        print(f"🔧 Server Status:")
                        print(f"   Tools available: {status['tools_available']}")
                        print(f"   Tool names: {', '.join(status['tool_names'])}")
                        continue
                    
                    # Process the input through the chatbot
                    print("🤖 Assistant: ", end="", flush=True)
                    
                    response_text = self.chatbot.process_text_input(user_input)
                    print(response_text)
                    
                except EOFError:
                    # Handle Ctrl+D gracefully
                    print("\n👋 Exiting terminal mode")
                    break
                except Exception as e:
                    print(f"\n❌ Error processing input: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n👋 Exiting terminal mode")
            return


def main():
    """Main entry point for standalone MCP server."""
    try:
        print("🏠 Standalone MCP Server with Terminal Client")
        print("=" * 50)
        
        # Create and initialize the server
        server = StandaloneMCPServer()
        
        if not server.initialize():
            print("❌ Failed to initialize MCP server")
            sys.exit(1)
        
        # Run terminal mode
        server.run_terminal_mode()
        
    except Exception as e:
        logger.error(f"Failed to start standalone MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 