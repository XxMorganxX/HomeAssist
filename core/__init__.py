"""
Lightweight core package initializer.

This module avoids importing heavy dependencies (numpy, webrtcvad) and any
speech service implementations at import time. Instead, it provides lazy
accessors so that importing submodules like `core.state_management.statemanager`
does not trigger realtime/audio initialization or configuration prints.
"""

from typing import List

__all__: List[str] = [
    'VADChunker',
    'wav_bytes_from_frames',
    'calculate_rms',
    'SpeechServices',
    'ConversationManager',
    'StreamingChatbot',
]


def __getattr__(name):
    """Lazily import heavy modules and conditionally expose symbols.

    - Audio processing symbols are imported from `.audio_processing` on demand.
    - Speech services are resolved on demand, deciding realtime vs traditional
      only when accessed, to avoid importing `config` during package import.
    - `StreamingChatbot` is imported on demand from `.streaming_chatbot`.
    """
    if name in {'VADChunker', 'wav_bytes_from_frames', 'calculate_rms'}:
        from . import audio_processing as _audio
        return getattr(_audio, name)

    if name in {'SpeechServices', 'ConversationManager'}:
        # Decide which implementation to expose only when needed
        try:
            import config as _config  # type: ignore
            use_realtime = bool(getattr(_config, 'USE_REALTIME_API', False))
        except Exception:
            use_realtime = False

        if use_realtime:
            try:
                from .speech_services_realtime import SpeechServices as _SS, ConversationManager as _CM
                return _SS if name == 'SpeechServices' else _CM
            except Exception:
                from .speech_services import SpeechServices as _SS, ConversationManager as _CM
                return _SS if name == 'SpeechServices' else _CM
        else:
            from .speech_services import SpeechServices as _SS, ConversationManager as _CM
            return _SS if name == 'SpeechServices' else _CM

    if name == 'StreamingChatbot':
        from .streaming_chatbot import StreamingChatbot as _SC
        return _SC

    raise AttributeError(f"module 'core' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + __all__)