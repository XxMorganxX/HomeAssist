"""
MCP Server bootstrap and main server implementation.
Coordinates tool discovery and MCP protocol handling.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug Python environment
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path[:3]}...")

# Test kasa import
try:
    import kasa
    logger.info(f"Kasa module found at: {kasa.__file__}")
except ImportError as e:
    logger.error(f"Cannot import kasa: {e}")
    logger.info(f"Site packages: {[p for p in sys.path if 'site-packages' in p]}")

from mcp_server.tool_registry import ToolRegistry


class MCPServer:
    """
    Main MCP Server that coordinates tools and provides MCP protocol handling.
    Provides a framework for extensible tool creation.
    """
    
    def __init__(self):
        """Initialize MCP server."""
        self.tool_registry = ToolRegistry()
        self.available_tools: Dict[str, Any] = {}
        
        # Discover and register tools
        self._discover_tools()
        
    def _discover_tools(self):
        """Discover and validate all available tools."""
        try:
            discovered = self.tool_registry.discover_tools()
            logger.info(f"Discovered {len(discovered)} tool modules")
            
            # Validate tools and build available tools dict
            for tool_name in self.tool_registry.get_available_tools():
                if self.tool_registry.validate_tool(tool_name):
                    schema = self.tool_registry.get_tool_schema(tool_name)
                    self.available_tools[tool_name] = {
                        'name': tool_name,
                        'schema': schema,
                        'status': 'available'
                    }
                    logger.info(f"Tool '{tool_name}' registered and validated")
                else:
                    logger.error(f"Tool '{tool_name}' failed validation")
                    
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools with their schemas.
        
        Returns:
            List of tool information dictionaries
        """
        return list(self.available_tools.values())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary
            
        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        # Get tool instance to access metadata
        tool_instance = self.tool_registry.get_tool_instance(tool_name)
        return tool_instance.get_info()
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        try:
            tool_instance = self.tool_registry.get_tool_instance(tool_name)
            
            # Execute tool
            result = tool_instance.safe_execute(params)
            
            logger.info(f"Tool '{tool_name}' executed successfully: {result.get('success', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}': {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def reload_tools(self) -> Dict[str, Any]:
        """
        Reload all tools (useful for development).
        
        Returns:
            Reload status information
        """
        try:
            old_count = len(self.available_tools)
            self.available_tools.clear()
            
            reloaded = self.tool_registry.reload_tools()
            self._discover_tools()
            
            new_count = len(self.available_tools)
            
            logger.info(f"Tools reloaded: {old_count} -> {new_count}")
            
            return {
                "success": True,
                "old_count": old_count,
                "new_count": new_count,
                "reloaded_modules": reloaded
            }
            
        except Exception as e:
            logger.error(f"Tool reload failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status.
        
        Returns:
            Server status information
        """
        return {
            "status": "running",
            "tools_available": len(self.available_tools),
            "tool_names": list(self.available_tools.keys())
        }
    
    def start(self):
        """Start the MCP server and keep it running."""
        logger.info("MCP Server starting...")
        logger.info(f"Available tools: {list(self.available_tools.keys())}")
        logger.info("Server ready to handle requests")
        
        print("\n=== MCP Server Running ===")
        print(f"Tools loaded: {len(self.available_tools)}")
        print(f"Available tools: {', '.join(self.available_tools.keys())}")
        print("\nThe server is now running. Press Ctrl+C to stop.")
        
        try:
            # Keep the server running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nShutting down MCP Server...")
            logger.info("MCP Server stopped by user")
            return


def main():
    """Entry point for running the MCP server."""
    try:
        server = MCPServer()
        
        # Print server status
        status = server.get_server_status()
        print("\n=== MCP Server Status ===")
        print(f"Status: {status['status']}")
        print(f"Tools available: {status['tools_available']}")
        print(f"Tool names: {', '.join(status['tool_names'])}")
        
        # List all tools with schemas
        print("\n=== Available Tools ===")
        for tool in server.list_tools():
            print(f"- {tool['name']}: {tool['schema'].get('description', 'No description')}")
        
        print("\n=== Server Ready ===")
        print("MCP server initialized and ready to handle requests")
        
        # Start server (in a real implementation, this would start the MCP protocol handler)
        server.start()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()