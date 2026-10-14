"""
Base classes for provider implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    from ..utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity
except ImportError:
    from assistant_framework.utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity


class StreamingProviderBase(ABC):
    """
    Base class for streaming audio providers.
    
    Provides:
    - Consistent lifecycle management
    - Automatic cleanup on errors
    - State tracking
    - Error handling integration
    """
    
    def __init__(self, config: Dict[str, Any], error_handler: Optional[ErrorHandler] = None):
        self.config = config
        self.error_handler = error_handler or ErrorHandler()
        
        # State management
        self._is_active = False
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._stream = None
        
        # Subclass must set this
        self._component_name = "unknown"
    
    @property
    def is_active(self) -> bool:
        """Check if provider is currently active."""
        return self._is_active
    
    async def start_safe(self):
        """
        Safe start with automatic cleanup on failure.
        
        Raises:
            RuntimeError: If already active
            Exception: If initialization fails
        """
        async with self._lock:
            if self._is_active:
                raise RuntimeError(f"{self._component_name} already active")
            
            try:
                print(f"🚀 Starting {self._component_name}...")
                await self._initialize_stream()
                self._is_active = True
                self._stop_event.clear()
                print(f"✅ {self._component_name} started")
                
            except Exception as e:
                print(f"❌ Failed to start {self._component_name}: {e}")
                
                # Log error
                error = ComponentError(
                    component=self._component_name,
                    severity=ErrorSeverity.FATAL,
                    message=f"Initialization failed",
                    exception=e
                )
                await self.error_handler.handle_error(error)
                
                # Cleanup on failure
                await self._cleanup_stream()
                raise
    
    async def stop_safe(self):
        """
        Safe stop with guaranteed cleanup.
        """
        async with self._lock:
            if not self._is_active:
                return
            
            print(f"🛑 Stopping {self._component_name}...")
            self._stop_event.set()
            self._is_active = False
            
            try:
                await self._cleanup_stream()
                print(f"✅ {self._component_name} stopped")
            except Exception as e:
                print(f"⚠️  Error during cleanup of {self._component_name}: {e}")
                
                error = ComponentError(
                    component=self._component_name,
                    severity=ErrorSeverity.WARNING,
                    message=f"Cleanup error",
                    exception=e
                )
                await self.error_handler.handle_error(error)
    
    async def restart(self):
        """Restart the provider."""
        print(f"🔄 Restarting {self._component_name}...")
        await self.stop_safe()
        await asyncio.sleep(0.5)  # Give system time to settle
        await self.start_safe()
    
    @abstractmethod
    async def _initialize_stream(self):
        """
        Provider-specific initialization.
        
        Subclasses must implement this to set up their streaming resources.
        """
        pass
    
    @abstractmethod
    async def _cleanup_stream(self):
        """
        Provider-specific cleanup.
        
        Subclasses must implement this to clean up their streaming resources.
        This MUST be safe to call multiple times.
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        One-time initialization (e.g., API client setup).
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """One-time cleanup (called on orchestrator shutdown)."""
        pass



