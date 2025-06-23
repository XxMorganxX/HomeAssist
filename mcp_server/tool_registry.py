"""
Dynamic tool discovery and registration system.
Automatically finds and loads MCP tools from the tools directory.
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for dynamically discovering and managing MCP tools."""
    
    def __init__(self, tools_directory: str = None):
        """
        Initialize tool registry.
        
        Args:
            tools_directory: Path to directory containing tool modules
        """
        if tools_directory is None:
            tools_directory = Path(__file__).parent / "tools"
        
        self.tools_directory = Path(tools_directory)
        self.registered_tools: Dict[str, Type] = {}
        self.tool_instances: Dict[str, Any] = {}
        
    def discover_tools(self) -> List[str]:
        """
        Discover all tool modules in the tools directory.
        
        Returns:
            List of discovered tool names
        """
        discovered = []
        
        if not self.tools_directory.exists():
            logger.warning(f"Tools directory {self.tools_directory} does not exist")
            return discovered
            
        for file_path in self.tools_directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip private files
                
            module_name = file_path.stem
            try:
                self._load_tool_module(module_name)
                discovered.append(module_name)
                logger.info(f"Discovered tool module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to load tool module {module_name}: {e}")
                
        return discovered
    
    def _load_tool_module(self, module_name: str) -> None:
        """Load a tool module and register any tools it contains."""
        try:
            # Import the module
            module_path = f"mcp_server.tools.{module_name}"
            module = importlib.import_module(module_path)
            
            # Find tool classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_tool_class(obj) and obj.__module__ == module_path:
                    tool_name = getattr(obj, 'name', name.lower())
                    self.registered_tools[tool_name] = obj
                    logger.info(f"Registered tool: {tool_name} from {module_name}")
                    
        except ImportError as e:
            logger.error(f"Could not import module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading tool module {module_name}: {e}")
    
    def _is_tool_class(self, obj: Type) -> bool:
        """Check if a class is a valid tool class."""
        # Import here to avoid circular imports
        from .base_tool import BaseTool
        
        return (inspect.isclass(obj) and 
                issubclass(obj, BaseTool) and 
                obj is not BaseTool)
    
    def get_tool_instance(self, tool_name: str, core_services=None) -> Any:
        """
        Get or create an instance of a tool.
        
        Args:
            tool_name: Name of the tool
            core_services: Core services to inject into tool
            
        Returns:
            Tool instance
        """
        if tool_name not in self.tool_instances:
            if tool_name not in self.registered_tools:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            tool_class = self.registered_tools[tool_name]
            self.tool_instances[tool_name] = tool_class(core_services)
            
        return self.tool_instances[tool_name]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.registered_tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Dict:
        """Get JSON schema for a tool."""
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool_class = self.registered_tools[tool_name]
        # Create temporary instance to get schema
        temp_instance = tool_class(None)
        return temp_instance.get_schema()
    
    def reload_tools(self) -> List[str]:
        """
        Reload all tools (useful for development).
        
        Returns:
            List of reloaded tool names
        """
        # Clear existing registrations
        self.registered_tools.clear()
        self.tool_instances.clear()
        
        # Invalidate import cache for tool modules
        tools_module_prefix = "mcp_server.tools."
        modules_to_remove = [
            module_name for module_name in list(importlib.sys.modules.keys())
            if module_name.startswith(tools_module_prefix)
        ]
        
        for module_name in modules_to_remove:
            del importlib.sys.modules[module_name]
            
        # Rediscover tools
        return self.discover_tools()
    
    def validate_tool(self, tool_name: str) -> bool:
        """
        Validate that a tool is properly implemented.
        
        Args:
            tool_name: Name of tool to validate
            
        Returns:
            True if tool is valid
        """
        try:
            tool_class = self.registered_tools.get(tool_name)
            if not tool_class:
                return False
                
            # Check required attributes/methods
            required_methods = ['execute', 'get_schema']
            for method in required_methods:
                if not hasattr(tool_class, method):
                    logger.error(f"Tool {tool_name} missing required method: {method}")
                    return False
                    
            # Try to create instance and get schema
            temp_instance = tool_class(None)
            schema = temp_instance.get_schema()
            
            if not isinstance(schema, dict):
                logger.error(f"Tool {tool_name} schema must be a dictionary")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Tool validation failed for {tool_name}: {e}")
            return False