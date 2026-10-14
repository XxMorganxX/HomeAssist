# Utils package - re-exports from subfolders for backward compatibility
#
# Note: Memory and some other modules have complex dependencies and are not
# eagerly imported to avoid circular imports. Import them directly:
#   from assistant_framework.utils.memory.vector_memory import VectorMemoryManager

# Audio utilities (commonly used, safe to import)
from .audio.audio_manager import get_audio_manager, SharedAudioManager
from .audio.barge_in import BargeInDetector, BargeInConfig, BargeInMode
from .audio.shared_audio_bus import SharedAudioBus, SharedAudioBusConfig
from .audio.device_manager import get_emeet_device, get_audio_device_config, AudioDeviceConfig
from .audio.tones import (
    beep_ready_to_listen,
    beep_wake_model_ready,
    beep_tool_success,
    beep_tool_failure,
    beep_system_ready,
    beep_listening_start,
    beep_response_start,
    beep_error,
    beep_shutdown,
    beep_wake_detected,
    beep_send_detected,
)

# Logging utilities (commonly used, safe to import)
from .logging.console_logger import (
    console_log,
    console_log_async,
    log_boot,
    log_shutdown,
    log_conversation_end,
    log_wake_word,
    log_user_message,
    log_assistant_response,
    log_tool_call,
    log_error,
    log_info,
)
from .logging.logging_config import setup_logging, vprint, eprint
from .logging.metrics import InMemoryMetrics, ComponentMetrics, Timer, MetricNames
from .logging.conversation_recorder import ConversationRecorder

# Core utilities (still at root level)
from .state_machine import AudioStateMachine, AudioState
from .error_handling import ErrorHandler, ComponentError, ErrorSeverity

# Briefing utilities (lazy import to avoid circular deps)
# Use: from assistant_framework.utils.briefing.briefing_manager import BriefingManager

# Memory utilities (lazy import to avoid circular deps)
# Use: from assistant_framework.utils.memory.vector_memory import VectorMemoryManager
# Use: from assistant_framework.utils.memory.persistent_memory import PersistentMemoryManager

__all__ = [
    # Audio
    "get_audio_manager",
    "SharedAudioManager",
    "BargeInDetector",
    "BargeInConfig",
    "BargeInMode",
    "SharedAudioBus",
    "SharedAudioBusConfig",
    "get_emeet_device",
    "get_audio_device_config",
    "AudioDeviceConfig",
    "beep_ready_to_listen",
    "beep_wake_model_ready",
    "beep_tool_success",
    "beep_tool_failure",
    "beep_system_ready",
    "beep_listening_start",
    "beep_response_start",
    "beep_error",
    "beep_shutdown",
    "beep_wake_detected",
    "beep_send_detected",
    # Logging
    "console_log",
    "console_log_async",
    "log_boot",
    "log_shutdown",
    "log_conversation_end",
    "log_wake_word",
    "log_user_message",
    "log_assistant_response",
    "log_tool_call",
    "log_error",
    "log_info",
    "setup_logging",
    "vprint",
    "eprint",
    "InMemoryMetrics",
    "ComponentMetrics",
    "Timer",
    "MetricNames",
    "ConversationRecorder",
    # Core
    "AudioStateMachine",
    "AudioState",
    "ErrorHandler",
    "ComponentError",
    "ErrorSeverity",
]
