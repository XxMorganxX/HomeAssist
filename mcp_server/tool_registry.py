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

try:
    from tools_config import is_tool_enabled, get_tool_config
except ImportError:
    # Fallback if config doesn't exist
    def is_tool_enabled(tool_name: str) -> bool:
        return True
    def get_tool_config(tool_name: str) -> dict:
        return {}

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
            # Search both tools and improved_tools directories
            print("Path2:" + str(Path(__file__).parent / "tools"))
            print("Path2:" + str(Path(__file__).parent / "improved_tools"))
            self.tools_directories = [
                Path(__file__).parent / "improved_tools"
            ]
        else:
            self.tools_directories = [Path(tools_directory)]
        
        self.registered_tools: Dict[str, Type] = {}
        self.tool_instances: Dict[str, Any] = {}
        
    def discover_tools(self) -> List[str]:
        """
        Discover all tool modules in the tools directories.
        
        Returns:
            List of discovered tool names
        """
        discovered = []
        
        for tools_dir in self.tools_directories:
            if not tools_dir.exists():
                logger.warning(f"Tools directory {tools_dir} does not exist")
                continue
                
            # Determine module prefix based on directory
            if tools_dir.name == "improved_tools":
                module_prefix = "mcp_server.improved_tools"
            else:
                module_prefix = "mcp_server.tools"
                
            for file_path in tools_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue  # Skip private files
                    
                module_name = file_path.stem
                try:
                    self._load_tool_module(module_name, module_prefix)
                    discovered.append(module_name)
                    logger.info(f"Discovered tool module: {module_name} from {module_prefix}")
                except Exception as e:
                    logger.error(f"Failed to load tool module {module_name} from {module_prefix}: {e}")
                    
        return discovered
    
    def _load_tool_module(self, module_name: str, module_prefix: str = "tools") -> None:
        """Load a tool module and register any tools it contains."""
        try:
            # Import the module
            module_path = f"{module_prefix}.{module_name}"
            
            # Perform a shallow import that tolerates missing heavy dependencies by delaying attribute access
            module = importlib.import_module(module_path)
            
            # Find tool classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_tool_class(obj) and obj.__module__ == module_path:
                    tool_name = getattr(obj, 'name', name.lower())
                    
                    # Check if tool is enabled
                    if not is_tool_enabled(tool_name):
                        logger.info(f"Tool '{tool_name}' is disabled in configuration")
                        continue
                    
                    self.registered_tools[tool_name] = obj
                    logger.info(f"Registered tool: {tool_name} from {module_name}")
                    
        except ImportError as e:
            logger.error(f"Could not import module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading tool module {module_name}: {e}")
    
    def _is_tool_class(self, obj: Type) -> bool:
        """Check if a class is a valid tool class."""
        # Import here to avoid circular imports
        from mcp_server.base_tool import BaseTool
        from mcp_server.improved_base_tool import ImprovedBaseTool
        
        return (inspect.isclass(obj) and 
                (issubclass(obj, BaseTool) or issubclass(obj, ImprovedBaseTool)) and 
                obj not in (BaseTool, ImprovedBaseTool))
    
    def get_tool_instance(self, tool_name: str) -> Any:
        """
        Get or create an instance of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance
        """
        if tool_name not in self.tool_instances:
            if tool_name not in self.registered_tools:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            tool_class = self.registered_tools[tool_name]
            # Pass tool configuration if the tool accepts it
            try:
                tool_config = get_tool_config(tool_name)
                if tool_config:
                    self.tool_instances[tool_name] = tool_class(config=tool_config)
                else:
                    self.tool_instances[tool_name] = tool_class()
            except TypeError:
                # Tool doesn't accept config parameter
                self.tool_instances[tool_name] = tool_class()
            # In dev preset, print tool instance creation
            try:
                from assistant_framework.config import get_active_preset
                if get_active_preset().lower().startswith("dev"):
                    print(f"[DEV] TOOL INSTANTIATED → {tool_name}")
            except Exception:
                pass
            
        return self.tool_instances[tool_name]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.registered_tools.keys())
    
    def get_all_tools_status(self) -> Dict[str, bool]:
        """
        Get status of all discovered tools (enabled/disabled).
        
        Returns:
            Dict mapping tool names to their enabled status
        """
        all_tools = {}
        
        # Check all Python files in all tools directories
        for tools_dir in self.tools_directories:
            if tools_dir.exists():
                for file_path in tools_dir.glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue
                        
                    module_name = file_path.stem
                    # For simplicity, assume tool name matches module name
                    # In practice, we'd need to load the module to get the actual tool name
                    all_tools[module_name] = is_tool_enabled(module_name)
        
        return all_tools
    
    def get_tool_schema(self, tool_name: str) -> Dict:
        """Get JSON schema for a tool."""
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool_class = self.registered_tools[tool_name]
        # Create temporary instance to get schema
        temp_instance = tool_class()
        try:
            from assistant_framework.config import get_active_preset
            if get_active_preset().lower().startswith("dev"):
                print(f"[DEV] TOOL SCHEMA → {tool_name}")
        except Exception:
            pass
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
        tools_module_prefix = "tools."
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
            temp_instance = tool_class()
            schema = temp_instance.get_schema()
            
            if not isinstance(schema, dict):
                logger.error(f"Tool {tool_name} schema must be a dictionary")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Tool validation failed for {tool_name}: {e}")
            return False