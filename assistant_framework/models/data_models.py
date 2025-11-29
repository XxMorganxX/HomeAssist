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