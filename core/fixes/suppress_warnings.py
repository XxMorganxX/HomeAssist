"""
Centralized warning suppression for the RasPi Smart Home project.

This module handles suppression of various deprecation warnings from dependencies.
Import this module at the beginning of any entry point to suppress known warnings.
"""

import warnings
import sys
import platform

# Import macOS audio fixes if on macOS
if platform.system() == 'Darwin':
    try:
        from core.fixes.macos_audio_fix import suppress_auhal_errors, configure_macos_audio
    except ImportError:
        # If running from a different directory structure
        try:
            from macos_audio_fix import suppress_auhal_errors, configure_macos_audio
        except ImportError:
            suppress_auhal_errors = None
            configure_macos_audio = None

def suppress_common_warnings():
    """Suppress common deprecation warnings from dependencies."""
    
    # Suppress pkg_resources deprecation warning
    # This warning comes from older packages that haven't migrated to importlib yet
    warnings.filterwarnings("ignore", category=UserWarning, 
                          message=".*pkg_resources is deprecated.*")
    
    # Suppress numpy dtype deprecation warnings if they occur
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                          message=".*numpy.dtype size changed.*")
    
    # Suppress tensorflow/onnx warnings if they occur
    warnings.filterwarnings("ignore", category=FutureWarning,
                          module="tensorflow|onnxruntime")
    
    # Suppress google auth warnings about using file-based credentials
    warnings.filterwarnings("ignore", category=UserWarning,
                          message=".*credentials were discovered.*")
    
    # Suppress sounddevice/PortAudio warnings on macOS
    if platform.system() == 'Darwin':
        warnings.filterwarnings("ignore", category=UserWarning, module="sounddevice")
        
        # Apply macOS audio fixes if available
        if suppress_auhal_errors:
            suppress_auhal_errors()
        if configure_macos_audio:
            configure_macos_audio()

# Automatically suppress warnings when this module is imported
suppress_common_warnings()