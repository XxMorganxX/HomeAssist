"""
Abstract interface for transcription/speech-to-text providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from ..models.data_models import TranscriptionResult


class TranscriptionInterface(ABC):
    """Abstract base class for all transcription providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the transcription provider.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def start_streaming(self) -> AsyncIterator[TranscriptionResult]:
        """
        Start streaming transcription from audio input.
        
        Yields:
            TranscriptionResult: Transcription results as they become available
        """
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> None:
        """Stop the streaming transcription."""
        pass
    
    @property
    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if transcription is currently active.
        
        Returns:
            bool: True if transcription is active, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the transcription provider."""
        pass
    
    @property
    def capabilities(self) -> dict:
        """
        Get provider capabilities.
        
        Returns:
            dict: Dictionary of provider capabilities
        """
        return {
            'streaming': True,
            'batch': False,
            'languages': ['en-US'],
            'audio_formats': ['pcm16'],
            'sample_rates': [16000]
        }