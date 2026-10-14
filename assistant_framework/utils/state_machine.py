"""
Audio state machine for coordinating component lifecycle.
"""

import asyncio
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from assistant_framework.utils.audio.tones import (
    beep_transition_idle_to_wakeword,
    beep_transition_idle_to_synthesizing,
    beep_transition_idle_to_transcribing,
    beep_transition_wakeword_to_transcribing,
    beep_transition_wakeword_to_processing,
    beep_transition_wakeword_to_synthesizing,
    beep_transition_wakeword_to_idle,
    beep_transition_transcribing_to_processing,
    beep_transition_transcribing_to_idle,
    beep_transition_processing_to_synthesizing,
    beep_transition_processing_to_transcribing,
    beep_transition_processing_to_idle,
    beep_transition_synthesizing_to_idle,
    beep_transition_synthesizing_to_wakeword,
    beep_transition_synthesizing_to_transcribing,
    beep_transition_error_to_idle,
    beep_transition_to_error,
)


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
    - Distinct audio beeps for each transition
    """
    
    # Mapping of (from_state, to_state) -> beep function
    # Each unique transition gets a distinct sound
    _TRANSITION_BEEPS: Dict[Tuple["AudioState", "AudioState"], Callable[[], None]] = {}
    
    @classmethod
    def _init_transition_beeps(cls) -> None:
        """Initialize the transition beep mapping (called once on first use)."""
        if cls._TRANSITION_BEEPS:
            return  # Already initialized
        
        cls._TRANSITION_BEEPS = {
            # IDLE transitions
            (AudioState.IDLE, AudioState.WAKE_WORD_LISTENING): beep_transition_idle_to_wakeword,
            (AudioState.IDLE, AudioState.SYNTHESIZING): beep_transition_idle_to_synthesizing,
            (AudioState.IDLE, AudioState.TRANSCRIBING): beep_transition_idle_to_transcribing,
            
            # WAKE_WORD_LISTENING transitions
            (AudioState.WAKE_WORD_LISTENING, AudioState.TRANSCRIBING): beep_transition_wakeword_to_transcribing,
            (AudioState.WAKE_WORD_LISTENING, AudioState.PROCESSING_RESPONSE): beep_transition_wakeword_to_processing,
            (AudioState.WAKE_WORD_LISTENING, AudioState.SYNTHESIZING): beep_transition_wakeword_to_synthesizing,
            (AudioState.WAKE_WORD_LISTENING, AudioState.IDLE): beep_transition_wakeword_to_idle,
            (AudioState.WAKE_WORD_LISTENING, AudioState.ERROR): beep_transition_to_error,
            
            # TRANSCRIBING transitions
            (AudioState.TRANSCRIBING, AudioState.PROCESSING_RESPONSE): beep_transition_transcribing_to_processing,
            (AudioState.TRANSCRIBING, AudioState.IDLE): beep_transition_transcribing_to_idle,
            (AudioState.TRANSCRIBING, AudioState.ERROR): beep_transition_to_error,
            
            # PROCESSING_RESPONSE transitions
            (AudioState.PROCESSING_RESPONSE, AudioState.SYNTHESIZING): beep_transition_processing_to_synthesizing,
            (AudioState.PROCESSING_RESPONSE, AudioState.TRANSCRIBING): beep_transition_processing_to_transcribing,
            (AudioState.PROCESSING_RESPONSE, AudioState.IDLE): beep_transition_processing_to_idle,
            (AudioState.PROCESSING_RESPONSE, AudioState.ERROR): beep_transition_to_error,
            
            # SYNTHESIZING transitions
            (AudioState.SYNTHESIZING, AudioState.IDLE): beep_transition_synthesizing_to_idle,
            (AudioState.SYNTHESIZING, AudioState.WAKE_WORD_LISTENING): beep_transition_synthesizing_to_wakeword,
            (AudioState.SYNTHESIZING, AudioState.TRANSCRIBING): beep_transition_synthesizing_to_transcribing,
            (AudioState.SYNTHESIZING, AudioState.ERROR): beep_transition_to_error,
            
            # ERROR transitions
            (AudioState.ERROR, AudioState.IDLE): beep_transition_error_to_idle,
        }
    
    def __init__(self, mode: str = "prod", transition_delay: float = 0.25, enable_transition_beeps: Optional[bool] = None):
        self._state = AudioState.IDLE
        self._current_component: Optional[str] = None
        self._lock = asyncio.Lock()
        self._transition_history: List[StateTransition] = []
        self._cleanup_handlers: Dict[str, Callable] = {}
        # Configurable delay for audio system settling between component switches
        # Lower = faster turnaround, but may cause conflicts on some systems
        self._transition_delay = transition_delay
        
        # Whether to play distinct beeps on state transitions
        # If not explicitly set, read from config
        if enable_transition_beeps is None:
            try:
                from assistant_framework.config import ENABLE_TRANSITION_BEEPS
                self._enable_transition_beeps = ENABLE_TRANSITION_BEEPS
            except ImportError:
                self._enable_transition_beeps = True
        else:
            self._enable_transition_beeps = enable_transition_beeps
        
        # Initialize transition beeps mapping
        self._init_transition_beeps()

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
                AudioState.PROCESSING_RESPONSE,  # For proactive responses (LLM-generated briefing)
                AudioState.SYNTHESIZING,  # For pre-generated briefing openers (TTS only, no LLM)
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
                AudioState.TRANSCRIBING,  # Allow barge-in: interrupt processing to start new transcription
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
                
                # Wait for audio system to settle ONLY when switching between
                # components that use different audio resources
                # Skip delay for same-component or non-audio transitions
                needs_settling = (
                    target_state != AudioState.ERROR and
                    self._transition_delay > 0 and
                    self._current_component in ("wakeword", "transcription", "tts") and
                    component in ("wakeword", "transcription", "tts")
                )
                if needs_settling:
                    print(f"⏳ Settling for {self._transition_delay}s...")
                    await asyncio.sleep(self._transition_delay)
            
            # Update state
            previous_state = self._state
            self._state = target_state
            self._current_component = component
            
            print(f"✅ State: {target_state.name} (component: {component})")
            
            # Play distinct beep for this transition
            if self._enable_transition_beeps:
                self._play_transition_beep(previous_state, target_state)
            
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
    
    def _play_transition_beep(self, from_state: AudioState, to_state: AudioState) -> None:
        """
        Play the distinct beep for a state transition.
        
        Args:
            from_state: The state we're transitioning from
            to_state: The state we're transitioning to
        """
        beep_fn = self._TRANSITION_BEEPS.get((from_state, to_state))
        if beep_fn:
            try:
                beep_fn()
            except Exception as e:
                # Never let beep failures affect state transitions
                print(f"⚠️  Beep failed for {from_state.name} → {to_state.name}: {e}")
    
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

