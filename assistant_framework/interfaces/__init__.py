"""
Abstract interfaces for the assistant framework components.
"""

from .transcription import TranscriptionInterface
from .response import ResponseInterface
from .text_to_speech import TextToSpeechInterface
from .context import ContextInterface
from .wake_word import WakeWordInterface
from .embedding import EmbeddingInterface
from .vector_store import VectorStoreInterface, VectorSearchResult, VectorRecord
from .termination import TerminationInterface

__all__ = [
    'TranscriptionInterface',
    'ResponseInterface',
    'TextToSpeechInterface',
    'ContextInterface',
    'WakeWordInterface',
    'EmbeddingInterface',
    'VectorStoreInterface',
    'VectorSearchResult',
    'VectorRecord',
    'TerminationInterface',
]