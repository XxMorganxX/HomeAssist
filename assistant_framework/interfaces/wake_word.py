"""
Abstract interface for wake word providers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator
from ..models.data_models import WakeWordEvent


class WakeWordInterface(ABC):
    """Abstract base class for all wake word providers."""

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the wake word provider.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def start_detection(self) -> AsyncIterator[WakeWordEvent]:
        """
        Start wake word detection from audio input.

        Yields:
            WakeWordEvent: Detection events as they occur
        """
        pass

    @abstractmethod
    async def stop_detection(self) -> None:
        """Stop wake word detection."""
        pass

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """
        Check if wake word detection is currently running.

        Returns:
            bool: True if detection is running, False otherwise
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the wake word provider."""
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
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'threshold_configurable': True,
        }

 
