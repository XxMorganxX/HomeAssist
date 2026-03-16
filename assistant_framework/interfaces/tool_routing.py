"""
Abstract interface for tool routing providers.

Tool routing providers handle tool selection and argument extraction:
given a user message, they decide which tools to call and with what parameters.
Tool execution (MCP) and response composition remain in the response provider.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

try:
    from ..models.data_models import ToolCall, HandoffContext
except ImportError:
    from assistant_framework.models.data_models import ToolCall, HandoffContext


class ToolRoutingInterface(ABC):
    """Abstract base class for all tool routing providers."""

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the tool routing provider.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def route(
        self,
        user_message: str,
        context: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        handoff: Optional[HandoffContext] = None,
    ) -> List[ToolCall]:
        """
        Determine which tools to call for a user message.

        Args:
            user_message: The user's natural-language request
            context: Conversation history (list of role/content dicts)
            available_tools: Tool schemas in OpenAI function-calling format.
                             Providers with built-in schemas may ignore this.
            handoff: Extracted intent, tone, and style context

        Returns:
            List of ToolCall objects (name + arguments). May be empty if
            the provider decides no tool is needed.
        """
        pass

    @abstractmethod
    async def check_additional_tools(
        self,
        user_message: str,
        tool_calls_so_far: List[ToolCall],
        context: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        handoff: Optional[HandoffContext] = None,
    ) -> List[ToolCall]:
        """
        Decide whether more tools are needed after a round of execution.

        Called iteratively: the response provider executes the returned tools,
        then calls this again until the list is empty or the iteration limit
        is reached.

        Args:
            user_message: The user's original request
            tool_calls_so_far: Tools already executed (with results populated)
            context: Conversation history
            available_tools: Tool schemas (same format as route())
            handoff: Extracted intent, tone, and style context

        Returns:
            List of additional ToolCall objects, or empty list when done.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources held by the provider."""
        pass

    @property
    def supports_iterative_routing(self) -> bool:
        """
        Whether this provider supports check_additional_tools().

        Providers that return False will skip the iterative tool-check loop;
        only the initial route() call will be used.
        """
        return True

    @property
    def capabilities(self) -> dict:
        """Provider capability metadata."""
        return {
            "iterative_routing": self.supports_iterative_routing,
        }
