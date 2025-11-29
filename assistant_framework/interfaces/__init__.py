"""
Abstract interfaces for the assistant framework components.
"""

from .transcription import TranscriptionInterface
from .response import ResponseInterface
from .text_to_speech import TextToSpeechInterface
from .context import ContextInterface
from .wake_word import WakeWordInterface

__all__ = [
    'TranscriptionInterface',
    'ResponseInterface',
    'TextToSpeechInterface',
    'ContextInterface',
    'WakeWordInterface'
]