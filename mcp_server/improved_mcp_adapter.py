"""
Improved MCP Tool Adapter that preserves parameter descriptions.
"""

import logging
from typing import Dict, Any
from fastmcp import FastMCP
from mcp_server.base_tool import BaseTool
from mcp_server.improved_base_tool import ImprovedBaseTool

logger = logging.getLogger(__name__)


class ImprovedMCPToolAdapter:
    """
    Improved adapter that converts BaseTool instances to FastMCP tools
    while preserving detailed parameter descriptions.
    """
    
    def __init__(self, mcp_server: FastMCP):
        """
        Initialize the adapter with a FastMCP server instance.
        
        Args:
            mcp_server: FastMCP server to register tools with
        """
        self.mcp_server = mcp_server
        self.registered_tools: Dict[str, BaseTool] = {}
        
    def register_tool(self, tool_instance: BaseTool) -> None:
        """
        Register a BaseTool instance as an MCP tool.
        
        Args:
            tool_instance: Instance of a class inheriting from BaseTool
        """
        tool_name = tool_instance.name
        
        # Store the tool instance for reference
        self.registered_tools[tool_name] = tool_instance
        
        try:
            # Check if it's an improved tool that can generate its own function
            if isinstance(tool_instance, ImprovedBaseTool):
                # Use the improved approach
                mcp_tool_func = tool_instance.to_fastmcp_function()
                logger.info(f"Using improved approach for tool '{tool_name}'")
            else:
                # Fall back to the old approach for legacy tools
                mcp_tool_func = self._create_legacy_mcp_tool_function(tool_instance)
                logger.info(f"Using legacy approach for tool '{tool_name}'")
            
            # Register with FastMCP
            decorated_func = self.mcp_server.tool()(mcp_tool_func)
            
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
    
    def _create_legacy_mcp_tool_function(self, tool_instance: BaseTool):
        """
        Legacy approach for tools that don't inherit from ImprovedBaseTool.
        This is the same as the original implementation.
        """
        tool_name = tool_instance.name
        tool_description = tool_instance.description
        
        # Get the tool's schema to understand parameters
        schema = tool_instance.get_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Build parameter string for exec
        param_list = []
        for param_name, param_info in properties.items():
            if param_name in required:
                param_list.append(param_name)
            else:
                # Add default value for optional parameters
                default_val = param_info.get("default", None)
                if default_val is None:
                    param_list.append(f"{param_name}=None")
                elif isinstance(default_val, str):
                    param_list.append(f"{param_name}='{default_val}'")
                else:
                    param_list.append(f"{param_name}={default_val}")
        
        params_str = ", ".join(param_list) if param_list else ""
        
        # Build kwargs collection code
        kwargs_code_lines = []
        for p in param_list:
            if '=' in p:
                param_name = p.split('=')[0]
                kwargs_code_lines.append(f"    if {param_name} is not None: kwargs['{param_name}'] = {param_name}")
            else:
                kwargs_code_lines.append(f"    kwargs['{p}'] = {p}")
        
        kwargs_code = '\n'.join(kwargs_code_lines) if kwargs_code_lines else "    pass"
        
        func_code = f"""
async def {tool_name}({params_str}) -> Dict[str, Any]:
    '''
    {tool_description}
    '''
    # Collect parameters into kwargs dict
    kwargs = {{}}
{kwargs_code}
    
    # Log tool call in dev mode
    try:
        from assistant_framework.config import get_active_preset
        if get_active_preset().lower().startswith("dev"):
            import json
            print(f"[DEV] TOOL CALL → {tool_name}: {{json.dumps(kwargs, ensure_ascii=False)}}")
    except Exception:
        pass
    
    try:
        # Execute the tool using its safe_execute method
        if inspect.iscoroutinefunction(tool_instance.execute):
            # Handle async tools
            result = await tool_instance.safe_execute(kwargs)
        else:
            # Handle sync tools
            result = tool_instance.safe_execute(kwargs)
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing MCP tool '{tool_name}': {{e}}")
        return {{
            "success": False,
            "error": f"Tool execution failed: {{str(e)}}"
        }}
"""
        
        # Create a namespace with required imports and the tool instance
        namespace = {
            'Dict': Dict,
            'Any': Any,
            'inspect': __import__('inspect'),
            'tool_instance': tool_instance,
            'logger': logger,
            'tool_name': tool_name
        }
        
        # Execute the function definition
        exec(func_code, namespace)
        
        # Get the created function
        mcp_tool_function = namespace[tool_name]
        
        return mcp_tool_function
    
    def get_registered_tool_names(self) -> list:
        """Get list of registered tool names."""
        return list(self.registered_tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get information about a registered tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary
        """
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        tool_instance = self.registered_tools[tool_name]
        return tool_instance.get_info()