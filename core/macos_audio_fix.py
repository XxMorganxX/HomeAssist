"""
macOS-specific audio fixes for AUHAL errors during development.
Provides error suppression and audio system optimization.
"""

import os
import sys
import platform
import contextlib
import warnings
import logging

def is_macos():
    """Check if running on macOS."""
    return platform.system() == 'Darwin'

def suppress_auhal_errors():
    """Suppress AUHAL error messages on macOS."""
    if not is_macos():
        return
    
    # Redirect stderr to suppress AUHAL errors
    class AUHALFilter:
        def __init__(self, stream):
            self.stream = stream
            self.suppress_patterns = [
                b'||PaMacCore (AUHAL)||',
                b'Error on line',
                b'err=\'-50\'',
                b'Unknown Error'
            ]
        
        def write(self, data):
            # Check if this is an AUHAL error
            if isinstance(data, bytes):
                for pattern in self.suppress_patterns:
                    if pattern in data:
                        return len(data)  # Pretend we wrote it
            elif isinstance(data, str):
                for pattern in self.suppress_patterns:
                    if pattern.decode() in data:
                        return len(data)
            
            # Not an AUHAL error, write normally
            return self.stream.write(data)
        
        def flush(self):
            return self.stream.flush()
        
        def fileno(self):
            return self.stream.fileno()
    
    # Apply the filter
    sys.stderr = AUHALFilter(sys.stderr)

def configure_macos_audio():
    """Configure audio settings to minimize AUHAL errors."""
    if not is_macos():
        return
    
    # Set environment variables for better audio behavior
    os.environ['PYTHON_COREAUDIO_VERBOSITY'] = '0'  # Reduce CoreAudio verbosity
    os.environ['PA_MIN_LATENCY_MSEC'] = '50'  # Increase minimum latency
    
    # Suppress PortAudio warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sounddevice')
    
    # Configure logging to suppress audio-related warnings
    logging.getLogger('sounddevice').setLevel(logging.ERROR)
    logging.getLogger('portaudio').setLevel(logging.ERROR)

@contextlib.contextmanager
def macos_audio_context():
    """Context manager for macOS audio operations."""
    if is_macos():
        # Store original stderr
        original_stderr = sys.stderr
        
        try:
            # Apply AUHAL suppression
            suppress_auhal_errors()
            yield
        finally:
            # Restore original stderr
            sys.stderr = original_stderr
    else:
        # Not macOS, just yield
        yield

def get_macos_audio_config():
    """Get macOS-optimized audio configuration."""
    if not is_macos():
        return {}
    
    return {
        'latency': 'high',  # Use high latency mode to reduce errors
        'prime_output_buffers_using_stream_callback': False,
        # Don't override blocksize here - let the caller specify it
    }

def reset_coreaudio():
    """Reset CoreAudio on macOS to fix AUHAL errors."""
    if not is_macos():
        return
    
    try:
        import subprocess
        # Kill coreaudiod to force it to restart
        subprocess.run(['sudo', 'killall', 'coreaudiod'], 
                      capture_output=True, text=True, timeout=5)
        # Give it time to restart
        import time
        time.sleep(1.0)
        print("🔄 CoreAudio reset completed")
    except Exception as e:
        print(f"⚠️ Could not reset CoreAudio: {e}")

def check_audio_device_availability():
    """Check if audio devices are available and not in use."""
    if not is_macos():
        return True
    
    try:
        import sounddevice as sd
        # Query devices to check availability
        devices = sd.query_devices()
        input_device = sd.query_devices(kind='input')
        
        if input_device is None:
            print("❌ No input audio device available")
            return False
            
        # Check if we can actually open a test stream
        try:
            test_stream = sd.InputStream(
                samplerate=16000,
                blocksize=1024,
                channels=1,
                dtype='int16'
            )
            test_stream.close()
            return True
        except Exception as e:
            if '-50' in str(e):
                print("⚠️ Audio device is locked by another application")
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking audio device: {e}")
        return False

# Auto-apply fixes when module is imported on macOS
if is_macos():
    configure_macos_audio()
    
    # Check if AUHAL suppression is enabled in config
    try:
        import config
        if getattr(config, 'SUPPRESS_AUHAL_ERRORS', True):
            suppress_auhal_errors()
    except ImportError:
        # If config not available, suppress by default during development
        suppress_auhal_errors()