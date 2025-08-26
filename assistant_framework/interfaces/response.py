"""
Abstract interface for response/LLM providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict, Optional, Any
from ..models.data_models import ResponseChunk


class ResponseInterface(ABC):
    """Abstract base class for all response/LLM providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the response provider and any required connections (e.g., MCP).
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def stream_response(self, 
                            message: str, 
                            context: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[ResponseChunk]:
        """
        Stream a response for the given message with optional context.
        
        Args:
            message: The user message to respond to
            context: Optional conversation history
            
        Yields:
            ResponseChunk: Response chunks as they become available
        """
        pass
    
    @abstractmethod
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools/functions.
        
        Returns:
            List of tool definitions
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool/function call.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result as string
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the response provider."""
        pass
    
    @property
    def capabilities(self) -> dict:
        """
        Get provider capabilities.
        
        Returns:
            dict: Dictionary of provider capabilities
        """
        return {
            'streaming': True,
            'batch': True,
            'tools': True,
            'max_tokens': 2000,
            'models': []
        }