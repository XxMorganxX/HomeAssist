# Audio utilities package
# Audio device management, barge-in detection, shared audio bus, and tones

from .audio_manager import get_audio_manager, SharedAudioManager
from .barge_in import BargeInDetector, BargeInConfig, BargeInMode
from .shared_audio_bus import SharedAudioBus, SharedAudioBusConfig
from .tones import (
    beep_ready_to_listen,
    beep_wake_model_ready,
    beep_tool_success,
    beep_tool_failure,
)
from .device_manager import get_emeet_device, get_audio_device_config

__all__ = [
    # audio_manager
    "get_audio_manager",
    "SharedAudioManager",
    # barge_in
    "BargeInDetector",
    "BargeInConfig",
    "BargeInMode",
    # shared_audio_bus
    "SharedAudioBus",
    "SharedAudioBusConfig",
    # tones
    "beep_ready_to_listen",
    "beep_wake_model_ready",
    "beep_tool_success",
    "beep_tool_failure",
    # device_manager
    "get_emeet_device",
    "get_audio_device_config",
]

