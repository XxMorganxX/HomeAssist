"""
MCP Server implementation using FastMCP.
Provides smart home and personal assistant tools via MCP protocol.
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path (for non-package modules under project)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FastMCP and our components
try:
    from fastmcp import FastMCP
except ImportError:
    logger.error("FastMCP not installed. Run: pip install fastmcp")
    sys.exit(1)

from mcp_server.tool_registry import ToolRegistry
from mcp_server.mcp_adapter import MCPToolAdapter


def create_mcp_server(host: str = "127.0.0.1", port: int = 3000) -> tuple[FastMCP, ToolRegistry]:
    """
    Create and configure the FastMCP server with all tools.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 3000)

    Returns:
        Tuple of (Configured FastMCP server instance, ToolRegistry instance)
    """
    # Initialize FastMCP server with verbose logging for tool calls
    mcp = FastMCP(name="Smart Home Assistant")
    
    # Initialize tool registry and adapter
    tool_registry = ToolRegistry()
    adapter = MCPToolAdapter(mcp)
    
    # Discover and register all tools
    try:
        discovered = tool_registry.discover_tools()
        logger.info(f"Discovered {len(discovered)} tool modules")
        
        registered_count = 0
        for tool_name in tool_registry.get_available_tools():
            if tool_registry.validate_tool(tool_name):
                try:
                    tool_instance = tool_registry.get_tool_instance(tool_name)
                    adapter.register_tool(tool_instance)
                    registered_count += 1
                    logger.info(f"Registered tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Failed to register tool '{tool_name}': {e}")
            else:
                logger.error(f"Tool '{tool_name}' failed validation")
        
        logger.info(f"Successfully registered {registered_count} tools")
        
    except Exception as e:
        logger.error(f"Tool discovery/registration failed: {e}")
        raise
    
    return mcp, tool_registry


def get_server_info(mcp: FastMCP, tool_registry: ToolRegistry) -> None:
    """
    Display server information.
    
    Args:
        mcp: FastMCP server instance
        tool_registry: ToolRegistry instance with discovered tools
    """
    # Safe to print here; guarded by transport in main()
    print("\n=== MCP Server Information ===")
    print(f"Server Name: Smart Home Assistant")
    print(f"Protocol: Model Context Protocol (MCP)")
    print(f"Transport: HTTP")
    print("\n=== Available Tools ===")
    
    # Use the existing tool_registry instead of creating a new one
    try:
        available_tools = tool_registry.get_available_tools()
        for tool_name in available_tools:
            if tool_registry.validate_tool(tool_name):
                tool_instance = tool_registry.get_tool_instance(tool_name)
                print(f"- {tool_name}: {tool_instance.description}")
    except Exception as e:
        logger.error(f"Error listing tools: {e}")


def main():
    """Entry point for running the MCP server."""
    parser = argparse.ArgumentParser(description="Smart Home MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind to (default: 3000)")
    parser.add_argument("--transport", default="http", choices=["http", "stdio", "sse"], 
                       help="Transport method (default: http)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output (for background service)")
    
    args = parser.parse_args()
    
    try:
        # Create and configure MCP server
        if not args.quiet:
            logger.info("Initializing MCP server...")
        mcp, tool_registry = create_mcp_server(args.host, args.port)
        
        # Display server information (skip in quiet/stdio mode)
        if not args.quiet and args.transport not in ["stdio"]:
            get_server_info(mcp, tool_registry)
        
        # Start the server
        if not args.quiet:
            logger.info(f"Starting MCP server on {args.host}:{args.port} using {args.transport} transport")
        
        if args.transport not in ["stdio"] and not args.quiet:
            print(f"\n=== Starting MCP Server ===")
            print(f"Host: {args.host}")
            print(f"Port: {args.port}")
            print(f"Transport: {args.transport}")
            print(f"\nServer is starting... Press Ctrl+C to stop.")
        
        # Run the server with specified transport
        if args.transport == "http":
            mcp.run(transport="http", host=args.host, port=args.port)
        elif args.transport == "sse":
            # SSE transport for persistent background server
            mcp.run(transport="sse", host=args.host, port=args.port)
        else:
            # In stdio mode, avoid stdout prints; FastMCP will use stdio for protocol
            mcp.run(transport="stdio")
            
    except KeyboardInterrupt:
        if not args.quiet:
            logger.info("Server stopped by user")
            if args.transport != "stdio":
                print("\n\nMCP Server stopped.")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        if args.transport != "stdio" and not args.quiet:
            print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()