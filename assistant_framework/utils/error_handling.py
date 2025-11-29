"""
Structured error handling with recovery strategies.
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Dict, Any, TypeVar, List
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = "warning"          # Log and continue
    RECOVERABLE = "recoverable"  # Attempt recovery
    FATAL = "fatal"              # Must stop component


@dataclass
class ComponentError:
    """Structured error information."""
    component: str
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    traceback_str: Optional[str] = None
    
    def __post_init__(self):
        """Capture traceback if exception provided."""
        if self.exception and not self.traceback_str:
            self.traceback_str = ''.join(
                traceback.format_exception(
                    type(self.exception),
                    self.exception,
                    self.exception.__traceback__
                )
            )


class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    
    Features:
    - Severity-based handling
    - Automatic retry with backoff
    - Component-specific recovery strategies
    - Error history tracking
    """
    
    def __init__(self):
        self._error_log: List[ComponentError] = []
        self._recovery_strategies: Dict[str, Callable] = {}
        self._max_history = 100
    
    def register_recovery(self, component: str, strategy: Callable):
        """
        Register recovery strategy for a component.
        
        Args:
            component: Component name
            strategy: Async recovery function that takes ComponentError
        """
        self._recovery_strategies[component] = strategy
        print(f"✅ Registered recovery strategy for: {component}")
    
    async def handle_error(self, error: ComponentError) -> bool:
        """
        Handle error based on severity.
        
        Args:
            error: Error to handle
            
        Returns:
            True if handled successfully or recoverable, False if fatal
        """
        # Add to history
        self._error_log.append(error)
        if len(self._error_log) > self._max_history:
            self._error_log.pop(0)
        
        # Handle based on severity
        if error.severity == ErrorSeverity.WARNING:
            return await self._handle_warning(error)
        elif error.severity == ErrorSeverity.RECOVERABLE:
            return await self._handle_recoverable(error)
        else:  # FATAL
            return await self._handle_fatal(error)
    
    async def _handle_warning(self, error: ComponentError) -> bool:
        """Handle warning-level error."""
        print(f"⚠️  {error.component}: {error.message}")
        if error.exception:
            print(f"   Exception: {error.exception}")
        return True
    
    async def _handle_recoverable(self, error: ComponentError) -> bool:
        """Handle recoverable error with retry."""
        print(f"🔧 {error.component}: {error.message} (attempting recovery)")
        
        strategy = self._recovery_strategies.get(error.component)
        if not strategy:
            print(f"⚠️  No recovery strategy for {error.component}")
            return False
        
        try:
            await strategy(error)
            print(f"✅ Recovery successful for {error.component}")
            return True
        except Exception as e:
            print(f"❌ Recovery failed for {error.component}: {e}")
            return False
    
    async def _handle_fatal(self, error: ComponentError) -> bool:
        """Handle fatal error."""
        print(f"💀 FATAL ERROR in {error.component}: {error.message}")
        if error.exception:
            print(f"   Exception: {error.exception}")
        if error.traceback_str:
            print("   Traceback:")
            print(error.traceback_str)
        return False
    
    def get_error_history(self, component: Optional[str] = None) -> List[ComponentError]:
        """
        Get error history, optionally filtered by component.
        
        Args:
            component: Optional component name to filter by
            
        Returns:
            List of errors
        """
        if component:
            return [e for e in self._error_log if e.component == component]
        return self._error_log.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        summary = {
            'total_errors': len(self._error_log),
            'by_severity': {},
            'by_component': {}
        }
        
        for error in self._error_log:
            # By severity
            severity = error.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # By component
            component = error.component
            summary['by_component'][component] = summary['by_component'].get(component, 0) + 1
        
        return summary


T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    error_handler: Optional[ErrorHandler] = None,
    component_name: str = "unknown"
) -> T:
    """
    Retry function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch
        error_handler: Optional error handler for logging
        component_name: Component name for error logging
        
    Returns:
        Result of function call
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            # Log error if handler provided
            if error_handler:
                error = ComponentError(
                    component=component_name,
                    severity=ErrorSeverity.RECOVERABLE if attempt < max_attempts - 1 else ErrorSeverity.FATAL,
                    message=f"Attempt {attempt + 1}/{max_attempts} failed",
                    exception=e,
                    context={'attempt': attempt + 1, 'max_attempts': max_attempts}
                )
                await error_handler.handle_error(error)
            
            # If last attempt, re-raise
            if attempt == max_attempts - 1:
                raise
            
            # Wait before retry
            print(f"⚠️  Attempt {attempt + 1}/{max_attempts} failed: {e}")
            print(f"   Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= backoff_factor
    
    # Should never reach here, but for type checker
    raise last_exception  # type: ignore


async def safe_cleanup(*cleanup_funcs: Callable):
    """
    Safely run multiple cleanup functions, ensuring all run even if some fail.
    
    Args:
        *cleanup_funcs: Async cleanup functions to run
    """
    errors = []
    
    for func in cleanup_funcs:
        try:
            await func()
        except Exception as e:
            errors.append((func.__name__, e))
            print(f"⚠️  Cleanup error in {func.__name__}: {e}")
    
    if errors:
        print(f"⚠️  {len(errors)} cleanup errors occurred")



