"""
Common data structures for the assistant framework.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class AudioFormat(str, Enum):
    """Enum for audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    PCM16 = "pcm16"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class TranscriptionResult:
    """Standardized output from all STT providers."""
    text: str
    is_final: bool
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    confidence: Optional[float] = None
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{'[FINAL]' if self.is_final else '[PARTIAL]'} {self.text}"


@dataclass
class ToolCall:
    """Represents a tool/function call."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'arguments': self.arguments,
            'call_id': self.call_id,
            'result': self.result
        }


@dataclass
class ResponseChunk:
    """Standardized output from all API/LLM providers."""
    content: str
    is_complete: bool
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_tool_calls(self) -> bool:
        """Check if this chunk contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def get_full_content(self) -> str:
        """Get the complete content including tool results if any."""
        if not self.has_tool_calls():
            return self.content
        
        parts = [self.content] if self.content else []
        for tool in self.tool_calls:
            if tool.result:
                parts.append(f"Tool {tool.name} result: {tool.result}")
        return "\n".join(parts)


@dataclass
class AudioOutput:
    """Standardized output from all TTS providers."""
    audio_data: bytes
    format: AudioFormat
    sample_rate: int
    duration: Optional[float] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_size_mb(self) -> float:
        """Get audio data size in megabytes."""
        return len(self.audio_data) / (1024 * 1024)
    
    def is_valid(self) -> bool:
        """Check if audio output is valid."""
        return len(self.audio_data) > 0 and self.sample_rate > 0


@dataclass
class WakeWordEvent:
    """Represents a wake word detection event."""
    model_name: str
    score: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationEvent:
    """
    Represents a termination phrase detection event.
    
    Used for parallel detection of conversation-ending phrases like "over out"
    that can interrupt any ongoing operation (TTS, processing, transcription).
    """
    phrase_name: str  # e.g., "over_out", "stop_listening"
    score: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    interrupted_state: Optional[str] = None  # What state was interrupted (SYNTHESIZING, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """Represents a message in conversation history."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with APIs."""
        msg = {
            'role': self.role.value,
            'content': self.content
        }
        
        if self.name:
            msg['name'] = self.name
        
        if self.tool_call_id:
            msg['tool_call_id'] = self.tool_call_id
            
        if self.tool_calls:
            msg['tool_calls'] = [tc.to_dict() for tc in self.tool_calls]
            
        return msg
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary format."""
        tool_calls = None
        if 'tool_calls' in data:
            tool_calls = [
                ToolCall(
                    name=tc.get('name', ''),
                    arguments=tc.get('arguments', {}),
                    call_id=tc.get('call_id'),
                    result=tc.get('result')
                )
                for tc in data['tool_calls']
            ]
        
        return cls(
            role=MessageRole(data['role']),
            content=data.get('content', ''),
            name=data.get('name'),
            tool_call_id=data.get('tool_call_id'),
            tool_calls=tool_calls,
            metadata=data.get('metadata', {})
        )


# =============================================================================
# CONTEXT BUNDLE
# =============================================================================

@dataclass
class ContextBundle:
    """
    Unified context bundle for response generation.
    
    Contains all context needed for a response in a single structure,
    eliminating redundant iteration over conversation history.
    
    Attributes:
        response_context: Full context for response generation (system + recent messages)
        tool_context: Compact context for tool selection (system + fewer messages)
        system_content: The combined system prompt content (for structured assembly)
    """
    response_context: List[Dict[str, Any]]
    tool_context: List[Dict[str, Any]]
    system_content: Optional[str] = None
    
    @property
    def has_system_message(self) -> bool:
        """Check if context includes a system message."""
        return bool(self.response_context and 
                   self.response_context[0].get("role") == "system")


# =============================================================================
# STATE TRANSITION TYPES
# =============================================================================

class TransitionReason(str, Enum):
    """
    Why a state transition is happening.
    
    Used by the orchestrator's _transition() wrapper to provide rich context
    for debugging and diagnostics. Every transition should specify a reason.
    """
    # Wake word related
    WAKE_DETECTED = "wake_detected"
    WAKE_STOP_FOR_TRANSCRIPTION = "wake_stop_for_transcription"
    WAKE_RESUME_WARM = "wake_resume_warm"
    
    # Transcription related
    TRANSCRIPTION_START = "transcription_start"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    TRANSCRIPTION_NO_INPUT = "transcription_no_input"
    TRANSCRIPTION_ERROR = "transcription_error"
    TRANSCRIPTION_CANCELLED = "transcription_cancelled"
    
    # Response/LLM related
    RESPONSE_START = "response_start"
    RESPONSE_COMPLETE = "response_complete"
    RESPONSE_ERROR = "response_error"
    RESPONSE_NO_OUTPUT = "response_no_output"
    
    # TTS related
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    TTS_BARGE_IN = "tts_barge_in"
    TTS_ERROR = "tts_error"
    
    # Conversation flow
    TERMINATION_DETECTED = "termination_detected"
    SESSION_END = "session_end"
    BRIEFING_DELIVERY = "briefing_delivery"
    
    # Error handling
    ERROR_RECOVERY = "error_recovery"
    MANUAL_RESET = "manual_reset"
    INVALID_STATE_RECOVERY = "invalid_state_recovery"


@dataclass
class TransitionContext:
    """
    Rich context for state transitions.
    
    Every call to orchestrator._transition() should include a TransitionContext
    to provide full visibility into why transitions happen. This enables:
    - Searchable logs (filter by reason, initiator)
    - Postmortem debugging (dump recent history on errors)
    - Metrics and alerting (track transition patterns)
    
    Example:
        await self._transition(
            AudioState.TRANSCRIBING,
            component="transcription",
            ctx=TransitionContext(
                reason=TransitionReason.WAKE_STOP_FOR_TRANSCRIPTION,
                initiated_by="system",
                conversation_id=self._recorder.current_session_id,
            )
        )
    """
    reason: TransitionReason
    initiated_by: str  # "wakeword", "user", "tts", "error_handler", "system"
    
    # Optional context (include what you have)
    conversation_id: Optional[str] = None
    wake_word: Optional[str] = None
    tool_name: Optional[str] = None
    user_message_snippet: Optional[str] = None  # First 50 chars of user message
    error_message: Optional[str] = None
    
    # Audio device state (critical for debugging device conflicts)
    audio_device: Dict[str, Any] = field(default_factory=dict)
    
    # Catch-all for anything else
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to flat metadata dict for state machine."""
        meta = {
            "reason": self.reason.value,
            "initiated_by": self.initiated_by,
        }
        
        if self.conversation_id:
            meta["conversation_id"] = self.conversation_id
        if self.wake_word:
            meta["wake_word"] = self.wake_word
        if self.tool_name:
            meta["tool_name"] = self.tool_name
        if self.user_message_snippet:
            meta["user_message"] = self.user_message_snippet
        if self.error_message:
            meta["error"] = self.error_message
        if self.audio_device:
            meta["audio_device"] = self.audio_device
        if self.extra:
            meta.update(self.extra)
        
        return meta