"""
Base tool class for FastMCP-compatible tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Literal
import inspect
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Base class for MCP tools that can generate FastMCP-compatible functions
    with rich parameter descriptions in docstrings and signatures.
    """

    # Tool metadata (should be overridden in subclasses)
    name: str = None
    description: str = None
    version: str = "1.0.0"

    def __init__(self):
        """Initialize the tool."""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Validate tool metadata
        if not self.name:
            self.name = self.__class__.__name__.lower().replace('tool', '')
        if not self.description:
            self.description = f"MCP tool: {self.name}"

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            params: Input parameters matching the tool's schema

        Returns:
            Dictionary containing the tool's output

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool's input parameters.

        Returns:
            JSON schema dictionary defining the tool's interface
        """
        pass

    def to_fastmcp_function(self):
        """
        Convert this tool to a FastMCP-compatible function with rich descriptions.

        Returns:
            Function that can be registered with FastMCP using @mcp.tool()
        """
        schema = self.get_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Build parameter annotations and defaults
        annotations = {}
        defaults = {}

        for param_name, param_info in properties.items():
            # Handle type mapping with enum support
            param_type = param_info.get("type", "string")

            if param_type == "string" and "enum" in param_info:
                # Create Literal type for enums, but only when non-empty
                enum_vals = list(param_info.get("enum") or [])
                if len(enum_vals) > 0:
                    annotations[param_name] = Literal[tuple(enum_vals)]
                else:
                    annotations[param_name] = str
            elif param_type == "string":
                annotations[param_name] = str
            elif param_type == "integer":
                annotations[param_name] = int
            elif param_type == "number":
                annotations[param_name] = float
            elif param_type == "boolean":
                annotations[param_name] = bool
            elif param_type == "array":
                annotations[param_name] = list
            elif param_type == "object":
                annotations[param_name] = dict
            else:
                annotations[param_name] = Any

            # Set defaults for optional parameters
            if param_name not in required:
                defaults[param_name] = param_info.get("default")

        # Create the dynamic function
        def dynamic_tool_function(**kwargs):
            """Dynamic function that executes the underlying tool."""
            try:
                # Filter kwargs to only include defined parameters
                filtered_params = {k: v for k, v in kwargs.items() if k in properties}

                # Add defaults for missing optional parameters
                for param_name, default_val in defaults.items():
                    if param_name not in filtered_params and default_val is not None:
                        filtered_params[param_name] = default_val

                # Execute the tool (logging is handled by the response provider)
                result = self.execute(filtered_params)
                return result

            except Exception as e:
                logger.error(f"Error executing tool '{self.name}': {e}")
                return {
                    "success": False,
                    "error": f"Tool execution failed: {str(e)}"
                }

        # Set function metadata
        dynamic_tool_function.__name__ = self.name
        dynamic_tool_function.__annotations__ = {**annotations, 'return': Dict[str, Any]}

        # Create enhanced docstring with parameter descriptions
        param_docs = []
        for param_name, param_info in properties.items():
            desc = param_info.get("description", "No description")
            param_type_name = annotations.get(param_name, "Any").__name__ if hasattr(annotations.get(param_name, "Any"), "__name__") else str(annotations.get(param_name, "Any"))
            required_marker = " [REQUIRED]" if param_name in required else ""
            default_info = f" (default: {defaults.get(param_name)})" if param_name in defaults and defaults[param_name] is not None else ""
            param_docs.append(f"        {param_name}: {desc}{required_marker}{default_info}")

        # Create comprehensive docstring
        docstring = f"""{self.description}

    Args:
{chr(10).join(param_docs)}

    Returns:
        Dictionary containing the tool's execution result with success status.
    """
        dynamic_tool_function.__doc__ = docstring

        # Create proper function signature with defaults
        sig_params = []
        for param_name in properties:
            default_val = defaults.get(param_name, inspect.Parameter.empty)
            param = inspect.Parameter(
                param_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default_val,
                annotation=annotations.get(param_name, Any)
            )
            sig_params.append(param)

        # Create and set the signature
        new_signature = inspect.Signature(
            parameters=sig_params,
            return_annotation=Dict[str, Any]
        )
        dynamic_tool_function.__signature__ = new_signature

        return dynamic_tool_function

    def get_info(self) -> Dict[str, Any]:
        """
        Get metadata about this tool.

        Returns:
            Dictionary with tool information
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate input parameters against the tool's schema.

        Args:
            params: Parameters to validate

        Returns:
            True if parameters are valid

        Raises:
            ValueError: If validation fails
        """
        schema = self.get_schema()

        # Basic validation - check required fields exist
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required parameter: {field}")

        return True

