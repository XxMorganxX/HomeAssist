"""
MCP Tool Adapter - Bridges BaseTool classes to FastMCP
"""

import logging
import inspect
from typing import Dict, Any, Callable, get_type_hints
from fastmcp import FastMCP
from mcp_server.base_tool import BaseTool

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """
    Adapter class that converts BaseTool instances into FastMCP tools.
    Maintains compatibility with existing tool architecture while enabling MCP protocol.
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
        
        # Store the tool instance for execution
        self.registered_tools[tool_name] = tool_instance
        
        # Create the MCP tool function
        mcp_tool_func = self._create_mcp_tool_function(tool_instance)
        
        # Register with FastMCP using the tool decorator
        decorated_func = self.mcp_server.tool()(mcp_tool_func)
        
        logger.info(f"Registered BaseTool '{tool_name}' as MCP tool")
        
    def _create_mcp_tool_function(self, tool_instance: BaseTool) -> Callable:
        """
        Create an MCP-compatible function from a BaseTool instance.
        
        Args:
            tool_instance: BaseTool instance to convert
            
        Returns:
            Function compatible with FastMCP tool registration
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
        
        # Create the function dynamically with proper parameters
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
    
    try:
        # Execute the tool using its execute method
        if inspect.iscoroutinefunction(tool_instance.execute):
            # Handle async tools
            result = await tool_instance.execute(kwargs)
        else:
            # Handle sync tools
            result = tool_instance.execute(kwargs)
        
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
            'inspect': inspect,
            'tool_instance': tool_instance,
            'logger': logger,
            'tool_name': tool_name
        }
        
        # Execute the function definition
        exec(func_code, namespace)
        
        # Get the created function
        mcp_tool_function = namespace[tool_name]
        
        # Add type hints based on schema
        self._add_type_hints(mcp_tool_function, properties, required)
        
        return mcp_tool_function
    
    def _add_type_hints(self, func: Callable, properties: Dict[str, Any], required: list) -> None:
        """
        Add type hints to the function based on JSON schema properties.
        
        Args:
            func: Function to add type hints to
            properties: JSON schema properties
            required: List of required parameter names
        """
        annotations = {}
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            
            # Convert JSON schema types to Python types
            if param_type == "string":
                python_type = str
            elif param_type == "integer":
                python_type = int
            elif param_type == "number":
                python_type = float
            elif param_type == "boolean":
                python_type = bool
            elif param_type == "array":
                python_type = list
            elif param_type == "object":
                python_type = dict
            else:
                python_type = Any
            
            # Make optional if not required
            if param_name not in required and param_type != "object":
                # Import Optional at runtime to avoid circular imports
                from typing import Optional
                python_type = Optional[python_type]
            
            annotations[param_name] = python_type
        
        # Set return type
        annotations["return"] = Dict[str, Any]
        
        # Apply annotations to function
        func.__annotations__ = annotations
    
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