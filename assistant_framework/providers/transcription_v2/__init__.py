"""
Fully async transcription providers (v2).
"""

from .assemblyai_async import AssemblyAIAsyncProvider
from .openai_whisper import OpenAIWhisperProvider

__all__ = ['AssemblyAIAsyncProvider', 'OpenAIWhisperProvider']



