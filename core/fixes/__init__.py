"""
Fixes and utilities for the RasPi Smart Home system.

This package contains various fixes and utilities to handle platform-specific issues
and suppress common warnings that might clutter the output.
"""

from .suppress_warnings import suppress_common_warnings
from .macos_audio_fix import suppress_auhal_errors, configure_macos_audio, check_audio_device_availability, get_macos_audio_config

__all__ = ['suppress_common_warnings', 'suppress_auhal_errors', 'configure_macos_audio', 'check_audio_device_availability', 'get_macos_audio_config'] 