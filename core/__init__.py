"""
Core audio processing and speech services.
This module contains the efficient, deterministic logic for audio processing.
"""

from .audio_processing import VADChunker, wav_bytes_from_frames, calculate_rms
from .speech_services import SpeechServices, ConversationManager
from .streaming_chatbot import StreamingChatbot

__all__ = [
    'VADChunker',
    'wav_bytes_from_frames', 
    'calculate_rms',
    'SpeechServices',
    'ConversationManager',
    'StreamingChatbot'
]