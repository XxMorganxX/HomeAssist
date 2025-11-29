"""
Text-to-speech provider implementations.
"""

from .google_tts import GoogleTTSProvider
from .local_tts import LocalTTSProvider

__all__ = ['GoogleTTSProvider', 'LocalTTSProvider']