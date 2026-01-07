"""
Abstract interface for termination phrase detection providers.

Enables parallel detection of conversation-ending phrases like "over out"
that can interrupt any ongoing operation (TTS, processing, transcription).

This runs as a separate detection stream alongside the main conversation flow,
allowing instant conversation termination without waiting for transcription.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from ..models.data_models import TerminationEvent


class TerminationInterface(ABC):
    """
    Abstract base class for termination phrase detection providers.
    
    Unlike wake word detection (which starts conversations), termination detection
    runs continuously during active conversations to enable instant shutdown
    when the user says phrases like "over out" or "stop listening".
    
    Key characteristics:
    - Runs in parallel with TTS, response generation, and transcription
    - Uses dedicated wake word model(s) for termination phrases
    - Lightweight - single model focused on termination only
    - Process-isolated to prevent crashes affecting main app
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the termination detection provider.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def start_detection(self) -> AsyncIterator[TerminationEvent]:
        """
        Start parallel termination phrase detection.
        
        This should be started when entering conversation mode and runs
        alongside other operations (TTS, transcription, etc.). It yields
        events when termination phrases are detected.

        Yields:
            TerminationEvent: Detection events as they occur
        """
        pass

    @abstractmethod
    async def stop_detection(self) -> None:
        """Stop termination phrase detection."""
        pass
    
    @abstractmethod
    async def pause_detection(self) -> None:
        """
        Pause detection temporarily (e.g., during wake word listening).
        
        Keeps the process warm for fast resume.
        """
        pass
    
    @abstractmethod
    async def resume_detection(self) -> None:
        """
        Resume detection after pause.
        """
        pass

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """
        Check if detection is currently running.

        Returns:
            bool: True if detection is running, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def is_paused(self) -> bool:
        """
        Check if detection is paused (process warm but not actively detecting).

        Returns:
            bool: True if paused, False otherwise
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        pass
    
    def set_current_state(self, state: str) -> None:
        """
        Update the current conversation state (for metadata in events).
        
        Args:
            state: Current AudioState name (e.g., "SYNTHESIZING", "TRANSCRIBING")
        """
        pass  # Optional implementation

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
            'parallel_execution': True,  # Key capability - runs alongside other operations
            'warm_mode': True,  # Supports pause/resume for fast restart
        }

