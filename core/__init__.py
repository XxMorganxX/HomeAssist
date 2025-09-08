"""
Core audio processing and speech services.
This module contains the efficient, deterministic logic for audio processing.
"""

# Avoid importing heavy numpy-dependent modules at package import time
try:
    from .audio_processing import VADChunker, wav_bytes_from_frames, calculate_rms
except Exception:
    VADChunker = None  # type: ignore
    wav_bytes_from_frames = None  # type: ignore
    calculate_rms = None  # type: ignore

try:
    from .streaming_chatbot import StreamingChatbot
except Exception:
    StreamingChatbot = None  # type: ignore


# Conditionally import speech services based on config
try:
    import config
    if config.USE_REALTIME_API:
        try:
            from .speech_services_realtime import SpeechServices, ConversationManager
        except ImportError:
            from .speech_services import SpeechServices, ConversationManager
    else:
        from .speech_services import SpeechServices, ConversationManager
except ImportError:
    # Fallback if config is not available during import
    from .speech_services import SpeechServices, ConversationManager

__all__ = [
    'VADChunker',
    'wav_bytes_from_frames', 
    'calculate_rms',
    'SpeechServices',
    'ConversationManager',
    'StreamingChatbot'
]