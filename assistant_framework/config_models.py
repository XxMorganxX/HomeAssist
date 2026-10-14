"""
Pydantic configuration models with validation.
"""

from pydantic import BaseModel, Field, validator, model_validator
from typing import Optional, Dict, Any
from pathlib import Path
import os


class AssemblyAIConfig(BaseModel):
    """AssemblyAI transcription configuration."""
    api_key: str = Field(..., description="AssemblyAI API key")
    sample_rate: int = Field(16000, ge=8000, le=48000, description="Audio sample rate")
    format_turns: bool = Field(True, description="Enable turn formatting")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid AssemblyAI API key (too short)')
        return v
    
    class Config:
        env_prefix = 'ASSEMBLYAI_'


class OpenAIConfig(BaseModel):
    """OpenAI response configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4o-realtime-preview-2024-12-17", description="Model name")
    max_tokens: int = Field(2000, ge=100, le=4000, description="Maximum tokens")
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Temperature")
    system_prompt: str = Field("", description="System prompt")
    recency_bias_prompt: str = Field("", description="Recency bias prompt")
    mcp_server_path: Optional[str] = Field(None, description="MCP server path")
    mcp_venv_python: Optional[str] = Field(None, description="MCP venv Python path")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or len(v) < 20:
            raise ValueError('Invalid OpenAI API key')
        return v
    
    @validator('mcp_server_path')
    def validate_mcp_path(cls, v):
        if v and not Path(v).exists():
            raise ValueError(f'MCP server not found at: {v}')
        return v
    
    class Config:
        env_prefix = 'OPENAI_'


class GoogleTTSConfig(BaseModel):
    """Google TTS configuration."""
    voice: str = Field("en-US-Chirp3-HD-Sadachbia", description="Voice name")
    speed: float = Field(1.3, ge=0.25, le=4.0, description="Speech speed")
    pitch: float = Field(-1.2, ge=-20.0, le=20.0, description="Pitch in semitones")
    language_code: str = Field("en-US", description="Language code")
    audio_encoding: str = Field("MP3", description="Audio encoding")
    
    @validator('audio_encoding')
    def validate_encoding(cls, v):
        valid_encodings = ['MP3', 'WAV', 'OGG']
        if v not in valid_encodings:
            raise ValueError(f'Invalid encoding. Must be one of: {valid_encodings}')
        return v
    
    class Config:
        env_prefix = 'GOOGLE_TTS_'


class WakeWordConfig(BaseModel):
    """Wake word detection configuration."""
    model_dir: str = Field(
        default_factory=lambda: os.getenv("WAKEWORD_MODEL_DIR", "./audio_data/wake_word_models"),
        description="Model directory"
    )
    model_name: str = Field("alexa_v0.1", description="Model name")
    sample_rate: int = Field(16000, ge=8000, le=48000, description="Sample rate")
    chunk: int = Field(1280, ge=128, le=4096, description="Audio chunk size")
    threshold: float = Field(0.2, ge=0.0, le=1.0, description="Detection threshold")
    cooldown_seconds: float = Field(2.0, ge=0.0, description="Cooldown after detection")
    min_playback_interval: float = Field(0.5, ge=0.0, description="Min interval between detections")
    input_device_index: Optional[int] = Field(None, description="Audio input device index")
    verbose: bool = Field(False, description="Enable verbose logging")
    
    @validator('model_dir')
    def validate_model_dir(cls, v):
        path = Path(v)
        if not path.exists():
            # Create if doesn't exist
            path.mkdir(parents=True, exist_ok=True)
        return v


class ContextConfig(BaseModel):
    """Context management configuration."""
    system_prompt: str = Field("", description="System prompt")
    model: str = Field("gpt-4", description="Model for context management")
    max_messages: int = Field(21, ge=1, le=100, description="Maximum messages to keep")
    enable_debug: bool = Field(False, description="Enable debug logging")
    response_recent_messages: int = Field(8, ge=1, le=50, description="Messages to send to responder")


class FrameworkConfig(BaseModel):
    """Complete framework configuration."""
    transcription: AssemblyAIConfig
    response: OpenAIConfig
    tts: GoogleTTSConfig
    context: ContextConfig
    wakeword: WakeWordConfig
    
    @model_validator(mode='after')
    def validate_api_keys(self):
        """Validate all API keys are present."""
        errors = []
        
        if not self.transcription or not self.transcription.api_key:
            errors.append("AssemblyAI API key required")
        
        if not self.response or not self.response.api_key:
            errors.append("OpenAI API key required")
        
        # Check Google credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            errors.append("GOOGLE_APPLICATION_CREDENTIALS environment variable required")
        
        if errors:
            raise ValueError('; '.join(errors))
        
        return self
    
    @classmethod
    def from_env(cls) -> 'FrameworkConfig':
        """Load configuration from environment variables."""
        return cls(
            transcription=AssemblyAIConfig(
                api_key=os.getenv('ASSEMBLYAI_API_KEY', '')
            ),
            response=OpenAIConfig(
                api_key=os.getenv('OPENAI_API_KEY', ''),
                system_prompt=os.getenv('SYSTEM_PROMPT', '')
            ),
            tts=GoogleTTSConfig(),
            context=ContextConfig(),
            wakeword=WakeWordConfig()
        )
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy configuration dictionary format."""
        return {
            'transcription': {
                'provider': 'assemblyai',
                'config': self.transcription.dict()
            },
            'response': {
                'provider': 'openai_websocket',
                'config': self.response.dict()
            },
            'tts': {
                'provider': 'google_tts',
                'config': self.tts.dict()
            },
            'context': {
                'provider': 'unified',
                'config': self.context.dict()
            },
            'wakeword': {
                'provider': 'openwakeword',
                'config': self.wakeword.dict()
            }
        }

