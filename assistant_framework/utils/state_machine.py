"""
Audio state machine for coordinating component lifecycle.
"""

import asyncio
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


class AudioState(Enum):
    """Audio system states."""
    IDLE = auto()
    WAKE_WORD_LISTENING = auto()
    TRANSCRIBING = auto()
    PROCESSING_RESPONSE = auto()
    SYNTHESIZING = auto()
    TRANSITIONING = auto()
    ERROR = auto()


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: AudioState
    to_state: AudioState
    component: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioStateMachine:
    """
    Manages audio component lifecycle with explicit states and transitions.
    
    Features:
    - Validates state transitions
    - Automatic cleanup on transitions
    - State history tracking
    - Component ownership tracking
    """
    
    def __init__(self, mode: str = "prod", transition_delay: float = 0.25):
        self._state = AudioState.IDLE
        self._current_component: Optional[str] = None
        self._lock = asyncio.Lock()
        self._transition_history: List[StateTransition] = []
        self._cleanup_handlers: Dict[str, Callable] = {}
        # Configurable delay for audio system settling between component switches
        # Lower = faster turnaround, but may cause conflicts on some systems
        self._transition_delay = transition_delay

        # Normalize mode and compute IDLE transitions based on environment
        env_mode = (mode or "prod").lower()
        if env_mode in ("dev", "development", "test", "testing"):
            idle_transitions = [
                AudioState.WAKE_WORD_LISTENING,
                AudioState.TRANSCRIBING,
                AudioState.PROCESSING_RESPONSE,
                AudioState.SYNTHESIZING,
                AudioState.ERROR,
            ]
        else:  # production defaults
            idle_transitions = [
                AudioState.WAKE_WORD_LISTENING,
                AudioState.ERROR,
            ]

        # Define valid transitions
        self._valid_transitions = {
            AudioState.IDLE: idle_transitions,
            AudioState.WAKE_WORD_LISTENING: [
                AudioState.TRANSCRIBING,
                AudioState.IDLE,
                AudioState.ERROR
            ],
            AudioState.TRANSCRIBING: [
                AudioState.PROCESSING_RESPONSE,
                AudioState.IDLE,
                AudioState.ERROR
            ],
            AudioState.PROCESSING_RESPONSE: [
                AudioState.SYNTHESIZING,
                AudioState.IDLE,
                AudioState.ERROR
            ],
            AudioState.SYNTHESIZING: [
                AudioState.IDLE,
                AudioState.WAKE_WORD_LISTENING,
                AudioState.TRANSCRIBING,  # Allow barge-in: interrupt TTS to start new transcription
                AudioState.ERROR
            ],
            AudioState.TRANSITIONING: [
                AudioState.IDLE,
                AudioState.WAKE_WORD_LISTENING,
                AudioState.TRANSCRIBING,
                AudioState.PROCESSING_RESPONSE,
                AudioState.SYNTHESIZING,
                AudioState.ERROR
            ],
            AudioState.ERROR: [
                AudioState.IDLE
            ]
        }
    
    @property
    def current_state(self) -> AudioState:
        """Get current state."""
        return self._state
    
    @property
    def current_component(self) -> Optional[str]:
        """Get current component owner."""
        return self._current_component
    
    def register_cleanup_handler(self, component: str, handler: Callable):
        """
        Register cleanup handler for a component.
        
        Args:
            component: Component name (e.g., "wakeword", "transcription")
            handler: Async cleanup function
        """
        self._cleanup_handlers[component] = handler
    
    async def transition_to(
        self,
        target_state: AudioState,
        component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Safely transition to a new state.
        
        Args:
            target_state: Desired state
            component: Component requesting the transition
            metadata: Optional metadata about the transition
            
        Returns:
            True if transition successful, False otherwise
            
        Raises:
            ValueError: If transition is invalid
        """
        async with self._lock:
            # Validate transition
            if target_state not in self._valid_transitions.get(self._state, []):
                raise ValueError(
                    f"Invalid transition: {self._state.name} → {target_state.name}"
                )
            
            print(f"🔄 State transition: {self._state.name} → {target_state.name} (component: {component})")
            
            # Record transition
            transition = StateTransition(
                from_state=self._state,
                to_state=target_state,
                component=component or "unknown",
                metadata=metadata or {}
            )
            self._transition_history.append(transition)
            
            # Cleanup previous component if different
            if self._current_component and self._current_component != component:
                await self._cleanup_component(self._current_component)
                
                # Wait for audio system to settle
                if target_state != AudioState.ERROR:
                    print(f"⏳ Settling for {self._transition_delay}s...")
                    await asyncio.sleep(self._transition_delay)
            
            # Update state
            previous_state = self._state
            self._state = target_state
            self._current_component = component
            
            print(f"✅ State: {target_state.name} (component: {component})")
            
            return True
    
    async def _cleanup_component(self, component: str):
        """
        Cleanup a specific component.
        
        Args:
            component: Component to cleanup
        """
        handler = self._cleanup_handlers.get(component)
        if handler:
            try:
                print(f"🧹 Cleaning up component: {component}")
                await handler()
                print(f"✅ Component cleaned: {component}")
            except Exception as e:
                print(f"⚠️  Error cleaning component {component}: {e}")
        else:
            print(f"ℹ️  No cleanup handler for: {component}")
    
    async def emergency_reset(self):
        """
        Emergency reset to IDLE state, cleaning up all components.
        Use when recovery is needed. Safe to call multiple times.
        """
        async with self._lock:
            # Skip if already in IDLE with no component (idempotent)
            if self._state == AudioState.IDLE and self._current_component is None:
                print("ℹ️  Already in IDLE state, skipping emergency reset")
                return
            
            print("🚨 Emergency state reset initiated")
            
            # Cleanup current component
            if self._current_component:
                await self._cleanup_component(self._current_component)
            
            # Reset state
            self._state = AudioState.IDLE
            self._current_component = None
            
            print("✅ State reset to IDLE")
    
    def get_transition_history(self, last_n: int = 10) -> List[StateTransition]:
        """
        Get recent transition history.
        
        Args:
            last_n: Number of recent transitions to return
            
        Returns:
            List of recent transitions
        """
        return self._transition_history[-last_n:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'state': self._state.name,
            'component': self._current_component,
            'history_size': len(self._transition_history),
            'last_transition': self._transition_history[-1] if self._transition_history else None
        }

