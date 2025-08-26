"""
Abstract interface for context management providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class ContextInterface(ABC):
    """Abstract base class for all context management providers."""
    
    @abstractmethod
    def initialize(self, system_prompt: Optional[str] = None) -> None:
        """
        Initialize the context manager.
        
        Args:
            system_prompt: Optional system prompt to set
        """
        pass
    
    @abstractmethod
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender ('user', 'assistant', 'system', 'tool')
            content: Content of the message
            metadata: Optional metadata for the message
        """
        pass
    
    @abstractmethod
    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add multiple messages to the conversation history.
        
        Args:
            messages: List of message dictionaries
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        pass
    
    @abstractmethod
    def get_recent_history(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the most recent n messages.
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        pass
    
    @abstractmethod
    def trim_history(self, max_messages: int) -> None:
        """
        Trim conversation history to a maximum number of messages.
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        pass
    
    @abstractmethod
    def count_tokens(self) -> int:
        """
        Count the total number of tokens in the conversation history.
        
        Returns:
            Total token count
        """
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary containing context summary information
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the conversation history to initial state."""
        pass
    
    @property
    def capabilities(self) -> dict:
        """
        Get provider capabilities.
        
        Returns:
            dict: Dictionary of provider capabilities
        """
        return {
            'token_counting': True,
            'auto_trimming': True,
            'metadata_support': True,
            'persistence': False
        }