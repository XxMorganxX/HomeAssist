"""
Abstract interface for text-to-speech providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
from pathlib import Path
from ..models.data_models import AudioOutput


class TextToSpeechInterface(ABC):
    """Abstract base class for all text-to-speech providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the TTS provider.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Optional voice identifier
            speed: Optional speed modifier (1.0 = normal)
            pitch: Optional pitch modifier in semitones
            
        Returns:
            AudioOutput: Synthesized audio data
        """
        pass
    
    async def stream_synthesize(self, 
                               text: str,
                               voice: Optional[str] = None,
                               speed: Optional[float] = None,
                               pitch: Optional[float] = None) -> AsyncIterator[AudioOutput]:
        """
        Stream synthesized speech (optional, not all providers support this).
        
        Args:
            text: Text to synthesize
            voice: Optional voice identifier  
            speed: Optional speed modifier
            pitch: Optional pitch modifier
            
        Yields:
            AudioOutput: Audio chunks as they become available
        """
        raise NotImplementedError("This provider does not support streaming synthesis")
    
    @abstractmethod
    def play_audio(self, audio: AudioOutput) -> None:
        """
        Play synthesized audio.
        
        Args:
            audio: Audio data to play
        """
        pass
    
    @abstractmethod
    async def save_audio(self, audio: AudioOutput, path: Path) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio data to save
            path: Path to save the audio file
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the TTS provider."""
        pass
    
    @property
    def capabilities(self) -> dict:
        """
        Get provider capabilities.
        
        Returns:
            dict: Dictionary of provider capabilities
        """
        return {
            'streaming': False,
            'batch': True,
            'voices': [],
            'languages': ['en-US'],
            'audio_formats': ['mp3', 'wav'],
            'speed_range': (0.5, 2.0),
            'pitch_range': (-20, 20)
        }