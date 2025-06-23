"""
Abstract base class for MCP tools.
Provides a consistent interface and access to core services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CoreServices:
    """Container for core services that tools can access."""
    
    def __init__(self, 
                 audio_processor=None,
                 speech_services=None, 
                 conversation_manager=None,
                 config=None):
        """
        Initialize core services container.
        
        Args:
            audio_processor: Audio processing utilities (VADChunker, etc.)
            speech_services: Speech-to-text and chat services
            conversation_manager: Conversation history management
            config: Configuration object
        """
        self.audio_processor = audio_processor
        self.speech_services = speech_services
        self.conversation_manager = conversation_manager
        self.config = config


class BaseTool(ABC):
    """
    Abstract base class for all MCP tools.
    
    Tools should inherit from this class and implement the required methods.
    """
    
    # Tool metadata (should be overridden in subclasses)
    name: str = None
    description: str = None
    version: str = "1.0.0"
    
    def __init__(self, core_services: Optional[CoreServices]):
        """
        Initialize the tool with access to core services.
        
        Args:
            core_services: Container with audio processing, speech services, etc.
        """
        self.core_services = core_services
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Validate tool metadata
        if not self.name:
            self.name = self.__class__.__name__.lower()
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
            
        Note:
            Basic implementation - tools can override for custom validation
        """
        schema = self.get_schema()
        
        # Basic validation - check required fields exist
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required parameter: {field}")
                
        return True
    
    def safe_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with error handling and validation.
        
        Args:
            params: Input parameters
            
        Returns:
            Dictionary with 'success', 'result', and optionally 'error' keys
        """
        try:
            # Validate parameters
            self.validate_params(params)
            
            # Execute the tool
            result = self.execute(params)
            
            return {
                "success": True,
                "result": result
            }
            
        except ValueError as e:
            self.logger.error(f"Validation error in {self.name}: {e}")
            return {
                "success": False,
                "error": f"Parameter validation failed: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Execution error in {self.name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    # Convenience methods for accessing core services
    
    @property
    def audio_processor(self):
        """Get audio processing utilities."""
        return self.core_services.audio_processor if self.core_services else None
    
    @property 
    def speech_services(self):
        """Get speech services (Whisper, ChatGPT)."""
        return self.core_services.speech_services if self.core_services else None
    
    @property
    def conversation_manager(self):
        """Get conversation manager."""
        return self.core_services.conversation_manager if self.core_services else None
    
    @property
    def config(self):
        """Get configuration object."""
        return self.core_services.config if self.core_services else None
    
    def log_info(self, message: str):
        """Log info message with tool context."""
        self.logger.info(f"[{self.name}] {message}")
        
    def log_error(self, message: str):
        """Log error message with tool context."""
        self.logger.error(f"[{self.name}] {message}")
        
    def log_debug(self, message: str):
        """Log debug message with tool context."""
        self.logger.debug(f"[{self.name}] {message}")


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass


class ToolValidationError(Exception):
    """Exception raised when tool parameter validation fails."""
    pass