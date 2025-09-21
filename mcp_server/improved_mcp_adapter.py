"""
Improved MCP Tool Adapter that preserves parameter descriptions.
"""

import logging
from typing import Dict, Any
from fastmcp import FastMCP
from mcp_server.improved_base_tool import ImprovedBaseTool

logger = logging.getLogger(__name__)


class ImprovedMCPToolAdapter:
    """
    Improved adapter that converts ImprovedBaseTool instances to FastMCP tools
    while preserving detailed parameter descriptions.
    """
    
    def __init__(self, mcp_server: FastMCP):
        """Initialize the adapter with a FastMCP server instance."""
        self.mcp_server = mcp_server
        self.registered_tools: Dict[str, ImprovedBaseTool] = {}

    def register_tool(self, tool_instance: ImprovedBaseTool) -> None:
        """Register an ImprovedBaseTool instance as an MCP tool."""
        if not isinstance(tool_instance, ImprovedBaseTool):
            raise TypeError("All tools must inherit from ImprovedBaseTool")

        tool_name = tool_instance.name
        self.registered_tools[tool_name] = tool_instance

        try:
            mcp_tool_func = tool_instance.to_fastmcp_function()
            # Register with FastMCP
            self.mcp_server.tool()(mcp_tool_func)
            logger.info(f"Successfully registered tool '{tool_name}' with FastMCP")
        except Exception as e:
            logger.error(f"Failed to register tool '{tool_name}': {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Optional helper to print tool name and params in dev preset before execution."""
        tool = self.registered_tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not registered"}
        try:
            # Print dev log for tool call
            try:
                from assistant_framework.config import get_active_preset
                if get_active_preset().lower().startswith("dev"):
                    import json
                    print(f"[DEV] TOOL CALL → {tool_name}: {json.dumps(arguments, ensure_ascii=False)}")
            except Exception:
                pass
            return tool.execute(arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_registered_tool_names(self) -> list:
        """Get list of registered tool names."""
        return list(self.registered_tools.keys())

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a registered tool."""
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        tool_instance = self.registered_tools[tool_name]
        return tool_instance.get_info()