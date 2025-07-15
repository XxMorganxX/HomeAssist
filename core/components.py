"""
Core components for the RasPi Smart Home voice assistant.
Provides modular, configurable components for wake word detection,
conversation handling, and system orchestration.
"""

import os
import sys
import time
import json
import random
import threading
import subprocess
import platform
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal
from openwakeword.model import Model
from openwakeword.utils import download_models
from enum import Enum

# Shared audio manager to prevent conflicts
class SharedAudioManager:
    """Manages shared audio resources and unified stream access."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.audio_in_use = False
            self.current_user = None
            self._shared_stream = None
            self._stream_config = None
            self._subscribers = {}  # Component name -> callback function
            self._initialized = True
    
    def request_audio_access(self, detector_name: str, timeout: float = 1.0) -> bool:
        """Request exclusive access to audio resources."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if not self.audio_in_use:
                    self.audio_in_use = True
                    self.current_user = detector_name
                    return True
            time.sleep(0.05)  # Small delay before retry
        return False
    
    def release_audio_access(self, detector_name: str):
        """Release audio resources."""
        with self._lock:
            if self.current_user == detector_name:
                self.audio_in_use = False
                self.current_user = None
    
    def create_shared_stream(self, samplerate: int, blocksize: int, channels: int = 1, dtype=np.int16, retry_count: int = 3):
        """Create a shared audio stream that multiple components can subscribe to."""
        with self._lock:
            if self._shared_stream is not None:
                return True  # Stream already exists
            
            # Check audio device availability on macOS
            if platform.system() == 'Darwin':
                try:
                    from core.fixes.macos_audio_fix import check_audio_device_availability
                    if not check_audio_device_availability():
                        print("⚠️ Audio device not available, attempting recovery...")
                        # Try to reset any stuck audio streams
                        sd._terminate()
                        sd._initialize()
                        time.sleep(0.5)
                except ImportError:
                    pass
            
            # Try multiple times with delays for macOS audio system
            for attempt in range(retry_count):
                if attempt > 0:
                    if config.DEBUG_MODE:
                        print(f"🔄 Retry attempt {attempt + 1}/{retry_count} for shared stream creation...")
                    time.sleep(0.5 * attempt)  # Exponential backoff
                
                try:
                    # Validate audio device and get compatible parameters
                    device_info = self._validate_audio_device(samplerate, blocksize, channels)
                    if not device_info:
                        continue  # Try next attempt
                    
                    # Use validated parameters
                    actual_samplerate = device_info['samplerate']
                    actual_blocksize = device_info['blocksize']
                    actual_channels = device_info['channels']
                    
                    self._stream_config = {
                        'samplerate': actual_samplerate,
                        'blocksize': actual_blocksize,
                        'channels': actual_channels,
                        'dtype': dtype,
                        'needs_resampling': actual_samplerate != samplerate,
                        'target_samplerate': samplerate  # Store original target rate
                    }
                
                    if config.DEBUG_MODE:
                        print(f"🎤 Creating shared stream: {actual_samplerate}Hz, {actual_blocksize} samples, {actual_channels} ch")
                        if self._stream_config['needs_resampling']:
                            print(f"⚠️ Will resample from {actual_samplerate}Hz to {samplerate}Hz")
                    
                    # Create stream with additional error handling for macOS audio issues
                    try:
                        # Add macOS-specific configuration if available
                        extra_settings = {}
                        if platform.system() == 'Darwin':
                            try:
                                from core.fixes.macos_audio_fix import get_macos_audio_config
                                extra_settings = get_macos_audio_config()
                            except ImportError:
                                pass
                        
                        self._shared_stream = sd.RawInputStream(
                            samplerate=actual_samplerate,
                            blocksize=actual_blocksize,
                            channels=actual_channels,
                            dtype=dtype,
                            callback=self._shared_audio_callback,
                            **extra_settings
                        )
                        
                        self._shared_stream.start()
                        
                        if config.DEBUG_MODE:
                            print("✅ Shared audio stream created and started successfully")
                        
                    except Exception as stream_error:
                        # Handle specific audio errors with fallback strategies
                        error_str = str(stream_error)
                        if "PaMacCore" in error_str or "Error -50" in error_str:
                            print(f"⚠️ macOS audio error: {error_str}")
                            print("🔄 Attempting recovery with different parameters...")
                            
                            # Try with smaller block size as fallback
                            fallback_blocksize = max(64, actual_blocksize // 2)
                            try:
                                self._shared_stream = sd.RawInputStream(
                                    samplerate=actual_samplerate,
                                    blocksize=fallback_blocksize,
                                    channels=actual_channels,
                                    dtype=dtype,
                                    callback=self._shared_audio_callback
                                )
                                
                                # Update config with fallback parameters
                                self._stream_config['blocksize'] = fallback_blocksize
                                
                                self._shared_stream.start()
                                print(f"✅ Shared stream created with fallback blocksize: {fallback_blocksize}")
                                
                            except Exception as fallback_error:
                                print(f"❌ Fallback also failed: {fallback_error}")
                                raise stream_error  # Re-raise original error
                        else:
                            raise stream_error  # Re-raise for non-macOS errors
                    
                    return True  # Success!
                    
                except Exception as e:
                    if attempt == retry_count - 1:  # Last attempt
                        print(f"❌ Failed to create shared audio stream after {retry_count} attempts: {e}")
                        return False
                    else:
                        if config.DEBUG_MODE:
                            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                        continue  # Try next attempt
            
            # Should not reach here
            return False
    
    def _validate_audio_device(self, target_samplerate: int, target_blocksize: int, target_channels: int) -> dict:
        """Validate audio device capabilities and return compatible parameters."""
        try:
            # Get default input device info
            device_info = sd.query_devices(kind='input')
            
            if config.DEBUG_MODE:
                print(f"🎤 Audio device: {device_info['name']}")
                print(f"   Native rate: {device_info['default_samplerate']} Hz")
                print(f"   Max channels: {device_info['max_input_channels']}")
            
            # Validate channels
            if device_info['max_input_channels'] < target_channels:
                print(f"❌ Device only supports {device_info['max_input_channels']} channels, need {target_channels}")
                return None
            
            # Use device's native sample rate to avoid resampling in driver
            device_samplerate = int(device_info['default_samplerate'])
            
            # Validate sample rate is reasonable
            if device_samplerate < 8000 or device_samplerate > 192000:
                print(f"⚠️ Unusual device sample rate: {device_samplerate} Hz, using target rate")
                device_samplerate = target_samplerate
            
            # Calculate appropriate block size for device sample rate
            # Keep the same time duration but adjust sample count
            time_duration = target_blocksize / target_samplerate
            device_blocksize = int(time_duration * device_samplerate)
            
            # Ensure block size is reasonable
            if device_blocksize < 64:
                device_blocksize = 64
            elif device_blocksize > 8192:
                device_blocksize = 8192
            
            return {
                'samplerate': device_samplerate,
                'blocksize': device_blocksize,
                'channels': target_channels
            }
            
        except Exception as e:
            print(f"❌ Error validating audio device: {e}")
            return None
    
    def subscribe_to_stream(self, component_name: str, callback: callable) -> bool:
        """Subscribe a component to receive audio data from the shared stream."""
        with self._lock:
            if self._shared_stream is None:
                return False
            
            self._subscribers[component_name] = callback
            return True
    
    def unsubscribe_from_stream(self, component_name: str):
        """Unsubscribe a component from the shared stream."""
        with self._lock:
            if component_name in self._subscribers:
                del self._subscribers[component_name]
    
    def stop_shared_stream(self):
        """Stop the shared audio stream."""
        with self._lock:
            if self._shared_stream:
                try:
                    self._shared_stream.stop()
                    self._shared_stream.close()
                except Exception as e:
                    print(f"⚠️ Error stopping shared stream: {e}")
                finally:
                    self._shared_stream = None
                    self._stream_config = None
                    self._subscribers.clear()
    
    def _shared_audio_callback(self, indata, frames, time_info, status):
        """Callback for shared audio stream that distributes data to subscribers."""
        if status:
            print(f"⚠️ Shared audio status: {status}")
        
        # Convert audio data to numpy array
        audio_array = np.frombuffer(indata, dtype=self._stream_config['dtype'])
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()
        
        # Apply resampling if needed
        if self._stream_config.get('needs_resampling', False):
            audio_array = self._resample_audio(audio_array)
        
        # Distribute to all subscribers
        for component_name, callback in self._subscribers.items():
            try:
                callback(audio_array)
            except Exception as e:
                print(f"⚠️ Error in audio callback for {component_name}: {e}")
    
    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio data to the target sample rate."""
        try:
            # Get the actual sample rate from stream config
            actual_rate = self._stream_config['samplerate']
            
            # Get the target rate from stream config
            target_rate = self._stream_config['target_samplerate']
            
            if actual_rate == target_rate:
                return audio_data
            
            # Calculate resampling ratio
            ratio = target_rate / actual_rate
            
            # Resample using scipy
            num_samples = int(len(audio_data) * ratio)
            if num_samples > 0:
                resampled = signal.resample(audio_data, num_samples)
                
                # Convert back to original dtype
                return resampled.astype(self._stream_config['dtype'])
            else:
                return audio_data
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Shared stream resampling error: {e}")
            return audio_data
    
    def get_stream_info(self) -> dict:
        """Get information about the current shared stream."""
        with self._lock:
            if self._shared_stream is None:
                return {"active": False}
            
            return {
                "active": True,
                "config": self._stream_config.copy(),
                "subscribers": list(self._subscribers.keys())
            }
        
    def is_subscribed(self, component_name: str) -> bool:
        """Check if a component is subscribed to the shared stream."""
        with self._lock:
            return component_name in self._subscribers

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

# Conditionally import the appropriate chatbot based on configuration
if config.USE_REALTIME_API and config.REALTIME_STREAMING_MODE:
    try:
        from core.streaming_chatbot_realtime import RealtimeStreamingChatbot as ToolEnabledStreamingChatbot
        USING_REALTIME_CHATBOT = True
        print("🚀 Using OpenAI Realtime API for conversations")
    except ImportError as e:
        print(f"⚠️ Failed to import realtime chatbot: {e}")
        print("⚠️ Falling back to chunk-based streaming")
        from core.streaming_chatbot import ToolEnabledStreamingChatbot
        USING_REALTIME_CHATBOT = False
else:
    from core.streaming_chatbot import ToolEnabledStreamingChatbot
    USING_REALTIME_CHATBOT = False

from core.audio_processing import VADChunker


class ComponentState(Enum):
    """Component state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class Component(ABC):
    """Abstract base class for system components."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.STOPPED
        self.error_message: Optional[str] = None
        
    @abstractmethod
    def start(self) -> bool:
        """Start the component. Returns True on success."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the component. Returns True on success."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if component is running properly."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status information."""
        return {
            "name": self.name,
            "state": self.state.value,
            "error": self.error_message,
            "healthy": self.is_healthy()
        }


class WakeWordDetector(Component):
    """Wake word detection component using OpenWakeWord."""
    
    def __init__(self, 
                 wake_word_callback: Callable[[], None],
                 model_name: str = "hey_monkey",
                 threshold: float = 0.3,
                 cooldown_seconds: float = 2.0):
        """
        Initialize wake word detector.
        
        Args:
            wake_word_callback: Function to call when wake word is detected
            model_name: Wake word model name (without .tflite extension)
            threshold: Detection threshold (0-1)
            cooldown_seconds: Cooldown period between detections
        """
        super().__init__("WakeWordDetector")
        self.wake_word_callback = wake_word_callback
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Internal state
        self.oww_model: Optional[Model] = None
        self.audio_stream = None
        self.detection_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Audio resource management
        self.audio_manager = SharedAudioManager()
        self.has_audio_access = False
        self.using_shared_stream = False
        
        # Detection state management
        self.last_activation_time = 0
        self.previous_scores = {}
        self.is_armed = {}
        
        # Audio configuration
        self.DTYPE = np.int16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1280
        self.needs_resampling = False
        
        # Thread management
        self._shutdown_event = threading.Event()
        
        # Paths
        self.word_model_dir = "./audio_data/wake_word_models"
        self.word_model_path = f"{self.word_model_dir}/{model_name}.tflite"
        self.opener_audio_dir = "./audio_data/opener_audio"
        
    def start(self) -> bool:
        """Start wake-word detection using the already-running shared microphone stream."""
        # All heavy lifting (model loading, state init) is handled inside start_shared_stream()
        return self.start_shared_stream()
    
    def stop(self) -> bool:
        """Stop wake word detection."""
        try:
            self.state = ComponentState.STOPPING
            self.running = False
            
            # Signal shutdown event for faster thread termination
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
            
            if config.DEBUG_MODE:
                print(f"🛑 Stopping {self.name}...")
            
            # Wait for detection thread to finish (but not if we're in the thread itself)
            if self.detection_thread and self.detection_thread.is_alive():
                import time
                import threading
                
                # Check if we're trying to join from within the same thread
                current_thread = threading.current_thread()
                if current_thread == self.detection_thread:
                    if config.DEBUG_MODE:
                        print(f"🔄 Skipping thread join (stopping from within detection thread)")
                else:
                    thread_start = time.time()
                    if config.DEBUG_MODE:
                        print(f"🛑 Waiting for {self.name} detection thread to stop...")
                    timeout = 0.2 if config.FAST_SHUTDOWN else 0.3  # Reduced timeout since we have faster shutdown
                    self.detection_thread.join(timeout=timeout)
                    thread_time = time.time() - thread_start
                    if self.detection_thread.is_alive():
                        print(f"⚠️ {self.name} detection thread did not stop gracefully after {thread_time:.2f}s")
                        # Force thread cleanup by setting running to False again
                        self.running = False
                    elif config.DEBUG_MODE:
                        print(f"   Detection thread stopped in {thread_time:.3f}s")
            
            # Clean up audio resources with comprehensive error handling
            audio_cleanup_success = self._cleanup_audio_resources()
            
            # Release audio access or unsubscribe from shared stream
            if self.using_shared_stream:
                self.audio_manager.unsubscribe_from_stream(self.name)
                self.using_shared_stream = False
                if config.DEBUG_MODE:
                    print(f"🔓 Unsubscribed {self.name} from shared stream")
            elif self.has_audio_access:
                self.audio_manager.release_audio_access(self.name)
                self.has_audio_access = False
                if config.DEBUG_MODE:
                    print(f"🔓 Audio access released for {self.name}")
            
            # Clear audio buffer
            if hasattr(self, '_audio_buffer'):
                self._audio_buffer.clear()
            
            # Reset detection state
            self.last_activation_time = 0
            self.previous_scores.clear()
            self.is_armed.clear()
            
            # Reset shutdown event for next start
            self._shutdown_event.clear()
            
            self.state = ComponentState.STOPPED
            if config.DEBUG_MODE:
                print(f"🛑 {self.name} stopped {'successfully' if audio_cleanup_success else 'with audio issues'}")
            
            return audio_cleanup_success
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to stop {self.name}: {e}")
            return False
    
    def start_shared_stream(self) -> bool:
        """Start wake word detection using shared audio stream."""
        try:
            self.state = ComponentState.STARTING
            
            # Ensure model directory exists and initialize model
            if not os.path.exists(self.word_model_path):
                os.makedirs(self.word_model_dir, exist_ok=True)
                download_models(target_directory=self.word_model_dir)
            
            # Initialize OpenWakeWord model
            onnx_model_path = self.word_model_path.replace('.tflite', '.onnx')
            if os.path.exists(onnx_model_path):
                self.oww_model = Model(
                    wakeword_models=[onnx_model_path], 
                    inference_framework='onnx'
                )
                if config.DEBUG_MODE:
                    print(f"Using ONNX model for shared stream: {onnx_model_path}")
            else:
                self.oww_model = Model(
                    wakeword_models=[self.word_model_path], 
                    inference_framework='tflite'
                )
                if config.DEBUG_MODE:
                    print(f"Using TFLite model for shared stream: {self.word_model_path}")
            
            # Initialize state tracking
            for model_name in self.oww_model.models.keys():
                self.previous_scores[model_name] = 0
                self.is_armed[model_name] = True
            
            # Subscribe to shared stream
            if not self.audio_manager.subscribe_to_stream(self.name, self._shared_audio_callback):
                print(f"❌ Could not subscribe {self.name} to shared stream")
                self.state = ComponentState.ERROR
                return False
            
            self.using_shared_stream = True
            
            # Start detection thread
            self.running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                name=f"{self.name}_detection",
                daemon=True
            )
            self.detection_thread.start()
            
            self.state = ComponentState.RUNNING
            if config.DEBUG_MODE:
                print(f"🎯 {self.name} started with shared stream successfully")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to start {self.name} with shared stream: {e}")
            return False
    
    def _shared_audio_callback(self, audio_data):
        """Callback for shared audio stream."""
        # Apply resampling if needed
        if self.needs_resampling:
            audio_data = self._resample_audio(audio_data)
        
        # Initialize audio buffer if not exists
        if not hasattr(self, '_audio_buffer'):
            self._audio_buffer = []
        
        # Store audio data for processing in detection loop
        self._audio_buffer.append(audio_data)
        
        # Keep buffer size manageable (max 10 frames)
        if len(self._audio_buffer) > 10:
            self._audio_buffer.pop(0)
    
    def _cleanup_audio_resources(self) -> bool:
        """Comprehensive audio resource cleanup."""
        import time
        cleanup_success = True
        
        try:
            if self.audio_stream:
                if config.DEBUG_MODE:
                    print(f"🔇 Cleaning up audio stream for {self.name}...")
                
                # Stop the stream first
                try:
                    if hasattr(self.audio_stream, 'active') and self.audio_stream.active:
                        self.audio_stream.stop()
                        if config.DEBUG_MODE:
                            print("   Audio stream stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping audio stream: {e}")
                    cleanup_success = False
                
                # Close the stream
                try:
                    self.audio_stream.close()
                    if config.DEBUG_MODE:
                        print("   Audio stream closed")
                except Exception as e:
                    print(f"⚠️ Error closing audio stream: {e}")
                    cleanup_success = False
                
                # Clear the reference
                self.audio_stream = None
                
                # Wait longer for audio resources to be fully released by macOS
                # This helps prevent PaMacCore errors when creating new streams
                time.sleep(0.25)
                
                if config.DEBUG_MODE:
                    print("🔇 Audio stream cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during audio cleanup: {e}")
            cleanup_success = False
        
        return cleanup_success
    
    def is_healthy(self) -> bool:
        """Check if wake word detector is healthy."""
        try:
            # Basic health checks
            basic_health = (self.state == ComponentState.RUNNING and 
                           self.running and 
                           self.detection_thread and 
                           self.detection_thread.is_alive() and
                           self.oww_model)
            
            if not basic_health:
                if config.DEBUG_MODE:
                    print(f"🔍 Wake word detector basic health check failed:")
                    print(f"   State: {self.state.value}")
                    print(f"   Running: {self.running}")
                    print(f"   Thread exists: {self.detection_thread is not None}")
                    print(f"   Thread alive: {self.detection_thread.is_alive() if self.detection_thread else 'N/A'}")
                    print(f"   Model loaded: {self.oww_model is not None}")
                return False
            
            # Audio health check depends on mode
            if self.using_shared_stream:
                # In shared stream mode, check if we're subscribed
                audio_healthy = self.audio_manager.is_subscribed(self.name)
                if not audio_healthy and config.DEBUG_MODE:
                    print(f"🔍 Wake word detector audio health check failed: not subscribed to shared stream")
                return audio_healthy
            else:
                # In exclusive mode, check if we have an audio stream
                audio_healthy = self.audio_stream is not None
                if not audio_healthy and config.DEBUG_MODE:
                    print(f"🔍 Wake word detector audio health check failed: no exclusive audio stream")
                return audio_healthy
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"❌ Error checking wake word detector health: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback for SoundDevice stream."""
        if status:
            if config.DEBUG_MODE:
                print(f"⚠️ Audio status: {status}")
        
        # Convert audio data to numpy array and then to the format expected by OpenWakeWord
        audio_array = np.frombuffer(indata, dtype=self.DTYPE)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()
        
        # Apply resampling if needed
        if self.needs_resampling:
            audio_array = self._resample_audio(audio_array)
        
        # Store audio data for processing in detection loop
        if hasattr(self, '_audio_buffer'):
            self._audio_buffer.append(audio_array)
        else:
            self._audio_buffer = [audio_array]
    
    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Resample audio data to the target sample rate."""
        try:
            if self.device_rate == self.RATE:
                return audio_data
            
            # Calculate resampling ratio
            ratio = self.RATE / self.device_rate
            
            # Resample using scipy
            num_samples = int(len(audio_data) * ratio)
            resampled = signal.resample(audio_data, num_samples)
            
            # Convert back to int16
            return resampled.astype(np.int16)
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Resampling error: {e}")
            return audio_data
    
    def _validate_audio_device(self) -> dict:
        """Validate audio device capabilities and configure parameters."""
        try:
            # Get default input device info
            device_info = sd.query_devices(kind='input')
            
            if config.DEBUG_MODE:
                print(f"Using audio input device: {device_info['name']}")
                print(f"Device native sample rate: {device_info['default_samplerate']} Hz")
                print(f"Max input channels: {device_info['max_input_channels']}")
            
            # Validate device supports required channels
            if device_info['max_input_channels'] < self.CHANNELS:
                raise Exception(f"Device only supports {device_info['max_input_channels']} channels, need {self.CHANNELS}")
            
            # Configure sample rate
            self.device_rate = int(device_info['default_samplerate'])
            
            # Validate sample rate is reasonable
            if self.device_rate < 8000 or self.device_rate > 192000:
                print(f"⚠️ Unusual device sample rate: {self.device_rate} Hz")
            
            # Check if resampling is needed
            self.needs_resampling = self.device_rate != self.RATE
            if self.needs_resampling:
                if config.DEBUG_MODE:
                    print(f"⚠️ Resampling needed: {self.device_rate} Hz → {self.RATE} Hz")
                
                # Validate resampling is feasible
                ratio = self.RATE / self.device_rate
                if ratio < 0.1 or ratio > 10.0:
                    print(f"⚠️ Extreme resampling ratio: {ratio:.2f}")
            
            # Calculate appropriate chunk size for device rate
            self.device_chunk = int(self.CHUNK * self.device_rate / self.RATE) if self.needs_resampling else self.CHUNK
            
            # Ensure chunk size is reasonable
            if self.device_chunk < 64:
                self.device_chunk = 64
                print(f"⚠️ Adjusted device chunk size to minimum: {self.device_chunk}")
            elif self.device_chunk > 8192:
                self.device_chunk = 8192
                print(f"⚠️ Adjusted device chunk size to maximum: {self.device_chunk}")
            
            if config.DEBUG_MODE:
                print(f"Device chunk size: {self.device_chunk} samples")
                print(f"Target chunk size: {self.CHUNK} samples")
            
            return device_info
            
        except Exception as e:
            raise Exception(f"Audio device validation failed: {e}")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        if config.DEBUG_MODE:
            print("🔊 Wake word detection started...")
        
        # Initialize audio buffer
        self._audio_buffer = []
        
        while self.running and not getattr(self, '_shutdown_event', threading.Event()).is_set():
            try:
                # Check shutdown more frequently
                if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                    break
                
                # Safety check: ensure model is properly initialized
                if not self.oww_model or not hasattr(self.oww_model, 'prediction_buffer'):
                    if config.DEBUG_MODE:
                        print("⚠️ Wake word model not properly initialized, skipping iteration")
                    # Use shorter sleep and check shutdown
                    for _ in range(10):  # 10 x 0.01s = 0.1s total, but check shutdown every 0.01s
                        if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                            break
                        time.sleep(0.01)
                    continue
                
                # Get audio frame from buffer
                if not hasattr(self, '_audio_buffer') or not self._audio_buffer:
                    # Use very short sleep and check shutdown frequently
                    for _ in range(2):  # 2 x 0.005s = 0.01s total
                        if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                            break
                        time.sleep(0.005)
                    continue
                
                # Check shutdown before processing
                if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                    break
                
                # Get the oldest audio frame
                audio_array = self._audio_buffer.pop(0)
                
                # Debug: Check if we're getting audio data
                if config.DEBUG_MODE:
                    rms = np.sqrt(np.mean(audio_array**2))
                    if rms > 100:  # Only print when there's actual audio
                        print(f"Audio RMS: {rms:.0f}")
                
                # Check shutdown before expensive prediction
                if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                    break
                
                # Process through OpenWakeWord
                prediction = self.oww_model.predict(audio_array)
                current_time = time.time()
                
                # Check shutdown after prediction
                if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                    break
                
                # Check each model for detection
                for model_name in self.oww_model.prediction_buffer.keys():
                    # Check shutdown in inner loop too
                    if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                        break
                    
                    # Ensure state dictionaries are properly initialized for this model
                    if model_name not in self.previous_scores:
                        self.previous_scores[model_name] = 0
                    if model_name not in self.is_armed:
                        self.is_armed[model_name] = True
                    
                    scores = list(self.oww_model.prediction_buffer[model_name])
                    if not scores:  # Skip if no scores available
                        continue
                    current_score = scores[-1]
                    previous_score = self.previous_scores.get(model_name, 0)
                    
                    # Time since last activation
                    time_since_last = current_time - self.last_activation_time
                    
                    # Debug output for scores above a lower threshold
                    if config.DEBUG_MODE and current_score > 0.05:
                        print(f"[{model_name}] Score: {current_score:.3f} (threshold: {self.threshold})")
                    
                    # Rising edge detection with cooldown
                    if (self.is_armed.get(model_name, True) and
                        previous_score <= self.threshold and 
                        current_score > self.threshold and 
                        time_since_last > self.cooldown_seconds):
                        
                        print(f"🎯 Wake word detected! Score: {current_score:.3f}")
                        
                        self.last_activation_time = current_time
                        self.is_armed[model_name] = False
                        
                        # Play greeting audio and get masking duration
                        mask_duration = self._play_greeting_with_mic_masking()
                        
                        # Trigger callback immediately with masking info
                        if self.wake_word_callback:
                            # Pass masking duration so conversation can ignore mic input for this time
                            if hasattr(self.wake_word_callback, '__code__') and self.wake_word_callback.__code__.co_argcount > 1:
                                self.wake_word_callback(mask_duration)
                            else:
                                self.wake_word_callback()
                    
                    # Update previous score
                    self.previous_scores[model_name] = current_score
                    
                    # Re-arm after cooldown
                    if not self.is_armed.get(model_name, True) and time_since_last > self.cooldown_seconds:
                        self.is_armed[model_name] = True
                    
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                if config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                    break
                # Shorter error sleep with shutdown check
                for _ in range(10):  # 10 x 0.01s = 0.1s total
                    if not self.running or getattr(self, '_shutdown_event', threading.Event()).is_set():
                        break
                    time.sleep(0.01)
        
        if config.DEBUG_MODE:
            print("🔊 Wake word detection loop ended")
    
    def _play_greeting_audio(self):
        """Play a random greeting audio file and wait for it to finish."""
        try:
            if not os.path.exists(self.opener_audio_dir):
                return
            
            audio_files = [f for f in os.listdir(self.opener_audio_dir) 
                          if f.endswith(('.wav', '.mp3', '.mov'))]
            
            if audio_files:
                greeting_file = random.choice(audio_files)
                audio_path = os.path.join(self.opener_audio_dir, greeting_file)
                
                if config.DEBUG_MODE:
                    print(f"🎵 Playing greeting: {greeting_file}")
                
                # Play audio and wait for it to finish
                try:
                    result = subprocess.run(['afplay', audio_path], 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL,
                                          timeout=10.0)  # Maximum 10 second timeout
                    
                    if config.DEBUG_MODE:
                        print("✅ Greeting audio finished")
                        
                except subprocess.TimeoutExpired:
                    if config.DEBUG_MODE:
                        print("⏰ Greeting audio timed out")
                except Exception as audio_error:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Audio playback error: {audio_error}")
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Greeting audio error: {e}")
    
    def _play_greeting_audio_async(self):
        """Play greeting audio and return when safe to start recording."""
        def play_audio_and_signal():
            # Play the greeting audio and wait for it to finish
            self._play_greeting_audio()
            # Add extra buffer time for audio system to settle
            time.sleep(0.5)
            # Set flag that it's safe to start recording
            self._greeting_finished = True
        
        # Reset the flag
        self._greeting_finished = False
        
        # Start audio in background thread
        audio_thread = threading.Thread(target=play_audio_and_signal, daemon=True)
        audio_thread.start()
        
        return audio_thread  # Return thread so caller can wait if needed
    
    def _play_greeting_with_mic_masking(self):
        """Play greeting audio while masking microphone input for exact duration."""
        try:
            if not os.path.exists(self.opener_audio_dir):
                return 0
            
            audio_files = [f for f in os.listdir(self.opener_audio_dir) 
                          if f.endswith(('.wav', '.mp3', '.mov'))]
            
            if audio_files:
                greeting_file = random.choice(audio_files)
                audio_path = os.path.join(self.opener_audio_dir, greeting_file)
                
                if config.DEBUG_MODE:
                    print(f"🎵 Playing greeting: {greeting_file}")
                
                # Get audio duration first (for precise masking)
                try:
                    import subprocess
                    result = subprocess.run(['afinfo', audio_path], 
                                          capture_output=True, text=True, timeout=5.0)
                    
                    # Parse duration from afinfo output
                    duration = 0
                    for line in result.stdout.split('\n'):
                        if 'estimated duration:' in line.lower():
                            duration_str = line.split(':')[-1].strip().split()[0]
                            duration = float(duration_str)
                            break
                    
                    if duration == 0:
                        duration = 2.0  # Fallback duration
                        
                except Exception:
                    duration = 2.0  # Fallback duration
                
                # Start audio playback in background
                audio_process = subprocess.Popen(['afplay', audio_path], 
                                               stdout=subprocess.DEVNULL, 
                                               stderr=subprocess.DEVNULL)
                
                if config.DEBUG_MODE:
                    print(f"🔇 Masking microphone for {duration:.1f}s during greeting")
                
                return duration  # Return exact duration for masking
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Greeting audio error: {e}")
            return 0


class ConversationHandler(Component):
    """Handles voice conversations with the AI assistant."""
    
    def __init__(self, 
                 conversation_end_callback: Optional[Callable[[], None]] = None):
        """
        Initialize conversation handler.
        
        Args:
            conversation_end_callback: Function to call when conversation ends
        """
        super().__init__("ConversationHandler")
        self.conversation_end_callback = conversation_end_callback
        self.chatbot: Optional[ToolEnabledStreamingChatbot] = None
        self.conversation_thread: Optional[threading.Thread] = None
        self.running = False
        
    def start(self) -> bool:
        """Start conversation handler (creates chatbot instance)."""
        try:
            self.state = ComponentState.STARTING
            
            # Create chatbot instance
            self.chatbot = ToolEnabledStreamingChatbot()
            
            # Pre-warm realtime connection to reduce wake word to recording latency
            if USING_REALTIME_CHATBOT and hasattr(self.chatbot, 'speech_services'):
                if config.DEBUG_MODE:
                    print("🔄 Pre-warming realtime connection...")
                self.chatbot.speech_services._ensure_connected()
            
            self.state = ComponentState.RUNNING
            if config.DEBUG_MODE:
                print(f"🤖 {self.name} started successfully")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to start {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop conversation handler."""
        try:
            self.state = ComponentState.STOPPING
            self.running = False
            
            # Wait for conversation thread to finish
            if self.conversation_thread and self.conversation_thread.is_alive():
                self.conversation_thread.join(timeout=5.0)
            
            self.chatbot = None
            self.state = ComponentState.STOPPED
            
            if config.DEBUG_MODE:
                print(f"🛑 {self.name} stopped")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to stop {self.name}: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if conversation handler is healthy."""
        try:
            healthy = (self.state == ComponentState.RUNNING and self.chatbot is not None)
            
            if not healthy and config.DEBUG_MODE:
                print(f"🔍 Conversation handler health check failed:")
                print(f"   State: {self.state.value}")
                print(f"   Chatbot exists: {self.chatbot is not None}")
                
            return healthy
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"❌ Error checking conversation handler health: {e}")
            return False
    
    def start_conversation(self, mask_duration=0) -> bool:
        """Start a new conversation session."""
        if not self.is_healthy():
            print("❌ ConversationHandler not ready")
            return False
        
        try:
            if config.DEBUG_MODE:
                print("💬 Starting conversation...")
            
            # Start conversation in separate thread with masking info
            self.running = True
            self.conversation_thread = threading.Thread(
                target=self._conversation_loop,
                args=(mask_duration,),
                daemon=True
            )
            self.conversation_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start conversation: {e}")
            return False
    
    def end_conversation_immediately(self):
        """End the conversation immediately and save to database."""
        try:
            if config.DEBUG_MODE:
                print("🛑 Ending conversation immediately via terminal word")
            
            # Signal the chatbot to end the session
            if self.chatbot:
                self.chatbot.session_ended = True
                
                # Save current conversation to database
                if hasattr(self.chatbot, 'conversation') and self.chatbot.conversation:
                    messages = self.chatbot.conversation.get_chat_minus_sys_prompt()
                    if messages:
                        self.chatbot.send_to_db(messages)
                        if config.DEBUG_MODE:
                            print("💾 Conversation saved to database")
            
            # Force the conversation to end by setting running to False
            self.running = False
            
            # Force termination of the conversation thread if it's still running
            if self.conversation_thread and self.conversation_thread.is_alive():
                if config.DEBUG_MODE:
                    print("🛑 Forcefully terminating conversation thread...")
                
                # Wait briefly for natural termination
                self.conversation_thread.join(timeout=0.2)
                
                # If thread is still alive, it's stuck - call callback manually
                if self.conversation_thread.is_alive():
                    if config.DEBUG_MODE:
                        print("⚠️ Conversation thread stuck, calling callback manually")
                    if self.conversation_end_callback:
                        self.conversation_end_callback()
            else:
                # No thread running, call callback directly
                if config.DEBUG_MODE:
                    print("🔄 No conversation thread, calling callback directly")
                if self.conversation_end_callback:
                    self.conversation_end_callback()
                
        except Exception as e:
            print(f"❌ Error ending conversation immediately: {e}")
    
    def _conversation_loop(self, mask_duration=0):
        """Run conversation until it ends."""
        try:
            if config.DEBUG_MODE:
                print("💬 Starting conversation loop...")
            
            # Check if we're using realtime chatbot
            if USING_REALTIME_CHATBOT and hasattr(self.chatbot, 'start_realtime_session'):
                if config.DEBUG_MODE:
                    print("🎯 Using realtime streaming mode")
                
                # Start realtime session (connection should already be pre-warmed)
                print("🎤  Listening...")
                
                # Pass mask duration to chatbot for smart audio masking
                self.chatbot.run(mask_duration)
                    
            else:
                if config.DEBUG_MODE:
                    print("📦 Using chunk-based streaming mode")
                # Run the standard chatbot
                self.chatbot.run()
            
            # Conversation ended normally
            if config.DEBUG_MODE:
                print("💬 Conversation ended normally")
                
        except Exception as e:
            print(f"❌ Conversation error: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
        finally:
            # Ensure cleanup happens regardless of how conversation ended
            self._cleanup_conversation_resources()
            
            # Always notify callback after cleanup
            if self.conversation_end_callback:
                try:
                    if config.DEBUG_MODE:
                        print("🔄 Calling conversation end callback...")
                    self.conversation_end_callback()
                except Exception as callback_error:
                    print(f"❌ Error in conversation end callback: {callback_error}")
                    if config.DEBUG_MODE:
                        import traceback
                        traceback.print_exc()
    
    def _cleanup_conversation_resources(self):
        """Clean up conversation resources and reset state."""
        try:
            if config.DEBUG_MODE:
                print("🧹 Cleaning up conversation resources...")
            
            # Set running to false
            self.running = False
            
            # Reset chatbot session state
            if self.chatbot:
                self.chatbot.session_ended = True
                
                # Handle realtime-specific cleanup
                if USING_REALTIME_CHATBOT and hasattr(self.chatbot, 'stop_realtime_session'):
                    try:
                        if config.DEBUG_MODE:
                            print("🔌 Stopping realtime session...")
                        self.chatbot.stop_realtime_session()
                    except Exception as e:
                        print(f"⚠️ Error stopping realtime session: {e}")
                
                # Clear accumulated chunks and reset state (for chunk-based mode)
                if hasattr(self.chatbot, 'accumulated_chunks'):
                    self.chatbot.accumulated_chunks.clear()
                if hasattr(self.chatbot, 'last_speech_activity'):
                    self.chatbot.last_speech_activity = None
            
            # Clear conversation thread reference
            self.conversation_thread = None
            
            # Ensure any audio resources are released in streaming chatbot
            # (The SharedAudioManager will handle this automatically via finally block)
            
            if config.DEBUG_MODE:
                print("✅ Conversation resources cleaned up")
                
        except Exception as e:
            print(f"⚠️ Error cleaning up conversation resources: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    def submit_conversation_to_file(self, conversation: List[Dict[str, Any]]):
        """Submit conversation to file."""
        try:
            with open(config.CONVERSATION_FILE, 'a') as f:
                json.dump(conversation, f)
        except Exception as e:
            print(f"❌ Error submitting conversation to file: {e}")

# Global orchestrator instance for cross-component access
_global_orchestrator = None

def get_global_orchestrator() -> Optional['ComponentOrchestrator']:
    """Get the global orchestrator instance."""
    return _global_orchestrator

def set_global_orchestrator(orchestrator: 'ComponentOrchestrator'):
    """Set the global orchestrator instance."""
    global _global_orchestrator
    _global_orchestrator = orchestrator
    




class ComponentOrchestrator:
    """Central orchestrator for managing all system components."""
    
    def __init__(self):
        global _global_orchestrator
        _global_orchestrator = self
        
        self.components: Dict[str, Component] = {}
        self.running = False
        self.mode = config.CONVERSATION_MODE
        
        # Component instances
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.terminal_word_detector: Optional[WakeWordDetector] = None
        self.conversation_handler: Optional[ConversationHandler] = None
        
        # State switching flag to prevent double transitions
        self._switching_state = False
        self._conversation_ended = False
        
        if config.DEBUG_MODE:
            print(f"🎛️ ComponentOrchestrator initialized in {self.mode} mode")
    
    def initialize_components(self) -> bool:
        """Initialize all components based on configuration."""
        try:
            # Always create conversation handler
            self.conversation_handler = ConversationHandler(
                conversation_end_callback=self._on_conversation_end
            )
            self.components["conversation"] = self.conversation_handler
            
            # Create wake word detector if enabled
            if config.WAKE_WORD_ENABLED and self.mode == "wake_word":
                self.wake_word_detector = WakeWordDetector(
                    wake_word_callback=self._on_wake_word_detected,
                    model_name=config.WAKE_WORD_MODEL,
                    threshold=config.WAKE_WORD_THRESHOLD,
                    cooldown_seconds=config.WAKE_WORD_COOLDOWN
                )
                self.components["wake_word"] = self.wake_word_detector
            
            # Create terminal word detector if enabled
            if config.TERMINAL_WORD_ENABLED and self.mode == "wake_word":
                self.terminal_word_detector = WakeWordDetector(
                    wake_word_callback=self._on_terminal_word_detected,
                    model_name=config.TERMINAL_WORD_MODEL,
                    threshold=config.TERMINAL_WORD_THRESHOLD,
                    cooldown_seconds=config.TERMINAL_WORD_COOLDOWN
                )
                self.components["terminal_word"] = self.terminal_word_detector
            
            if config.DEBUG_MODE:
                print(f"✅ Initialized {len(self.components)} components")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            return False
    
    def start_all(self) -> bool:
        """Start all components."""
        try:
            # Ensure the shared microphone stream is running BEFORE any component starts
            from core.components import SharedAudioManager
            import numpy as np
            audio_mgr = SharedAudioManager()
            if not audio_mgr.create_shared_stream(
                samplerate=config.SAMPLE_RATE,
                blocksize=config.FRAME_SIZE,
                channels=1,
                dtype=np.int16,
            ):
                print("❌ Failed to create shared microphone stream")
                return False
            else:
                if config.DEBUG_MODE:
                    info = audio_mgr.get_stream_info()
                    print(f"✅ Shared microphone stream active: {info['config']}")

            if config.STARTUP_SOUND:
                self._play_startup_sound()
            
            # Start conversation handler first
            if not self.conversation_handler.start():
                return False
            
            # Start wake word detector if enabled (only active when asleep)
            if self.wake_word_detector and not self.wake_word_detector.start():
                return False
                
            # Terminal word detector is created but not started initially (will be started during conversation)
            
            self.running = True
            
            # Handle different modes
            if self.mode == "wake_word":
                print("🎤 Voice assistant ready - say the wake word to start")
            elif self.mode == "continuous":
                print("🎤 Voice assistant ready - always listening")
                self._start_continuous_conversation()
            elif self.mode == "interactive":
                print("🎤 Voice assistant ready - press Enter to start conversation")
                self._start_interactive_mode()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start components: {e}")
            return False
    
    def stop_all(self) -> bool:
        """Stop all components."""
        try:
            import time
            self.running = False
            
            success = True
            for name, component in self.components.items():
                print(f"🛑 Stopping component: {name}")
                component_start = time.time()
                if not component.stop():
                    success = False
                component_time = time.time() - component_start
                print(f"   {name} stopped in {component_time:.2f}s")
            
            print("👋 Voice assistant components stopped")
            return success
            
        except Exception as e:
            print(f"❌ Failed to stop components: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        status = {
            "mode": self.mode,
            "running": self.running,
            "components": {}
        }
        
        for name, component in self.components.items():
            status["components"][name] = component.get_status()
        
        return status
    
    def run_forever(self):
        """Run the orchestrator forever (until interrupted)."""
        try:
            while self.running:
                # Health check
                self._health_check()
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.stop_all()
    
    def _on_wake_word_detected(self, mask_duration=0):
        """Callback when wake word is detected."""
        try:
            current_time = time.time()
            
            # Rate limiting: prevent too many rapid wake word activations
            if hasattr(self, '_last_wake_word_activation'):
                time_since_last = current_time - self._last_wake_word_activation
                min_interval = 5.0  # Minimum 5 seconds between wake word activations
                
                if time_since_last < min_interval:
                    if config.DEBUG_MODE:
                        print(f"🚫 Wake word rate limited: {time_since_last:.1f}s < {min_interval}s minimum")
                    return
            
            self._last_wake_word_activation = current_time
            
            if config.DEBUG_MODE:
                print("🎯 Wake word detected - starting conversation")
                print(f"🔍 System state check:")
                print(f"   Conversation ended: {self._conversation_ended}")
                print(f"   Switching state: {self._switching_state}")
                if self.conversation_handler:
                    print(f"   Conversation handler state: {self.conversation_handler.state.value}")
                    print(f"   Conversation handler healthy: {self.conversation_handler.is_healthy()}")
                    print(f"   Conversation running: {getattr(self.conversation_handler, 'running', 'N/A')}")
            
            # Validate and fix system state before proceeding
            if config.DEBUG_MODE:
                print("🔧 Validating and fixing system state...")
            self._validate_and_fix_system_state()
            
            # Force reset any stuck states before starting conversation
            self._conversation_ended = False
            self._switching_state = False
            
            # Check if conversation handler is stuck in running state
            if self.conversation_handler and self.conversation_handler.running:
                if config.DEBUG_MODE:
                    print("⚠️ Conversation handler stuck in running state, forcing reset...")
                try:
                    self.conversation_handler.running = False
                    if self.conversation_handler.conversation_thread and self.conversation_handler.conversation_thread.is_alive():
                        if config.DEBUG_MODE:
                            print("🛑 Forcefully ending stuck conversation thread...")
                        # Signal the chatbot to end if it exists
                        if self.conversation_handler.chatbot:
                            self.conversation_handler.chatbot.session_ended = True
                        # Wait briefly for natural termination
                        self.conversation_handler.conversation_thread.join(timeout=0.1)
                    # Clear thread reference
                    self.conversation_handler.conversation_thread = None
                except Exception as reset_error:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Error resetting conversation handler: {reset_error}")
            
            # Switch detector states: disable wake word, enable terminal word
            self._switch_to_listening_mode()
            
            if self.conversation_handler:
                # Ensure conversation handler is ready
                if not self.conversation_handler.is_healthy():
                    if config.DEBUG_MODE:
                        print("⚠️ Conversation handler not healthy, attempting to fix...")
                    
                    # Try to reset conversation handler state
                    try:
                        if self.conversation_handler.state != ComponentState.RUNNING:
                            if config.DEBUG_MODE:
                                print("🔄 Restarting conversation handler...")
                            self.conversation_handler.stop()
                            time.sleep(0.1)
                            if not self.conversation_handler.start():
                                print("❌ Failed to restart conversation handler")
                                self._switch_to_wake_word_mode()
                                return
                    except Exception as restart_error:
                        print(f"❌ Error restarting conversation handler: {restart_error}")
                        self._switch_to_wake_word_mode()
                        return
                
                # Add error handling for conversation start
                try:
                    if config.DEBUG_MODE:
                        print("🚀 Attempting to start conversation...")
                    success = self.conversation_handler.start_conversation(mask_duration)
                    if not success:
                        print("⚠️ Failed to start conversation - returning to wake word mode")
                        if config.DEBUG_MODE:
                            print("🔍 Conversation start failure details:")
                            print(f"   Handler state: {self.conversation_handler.state.value}")
                            print(f"   Handler healthy: {self.conversation_handler.is_healthy()}")
                            print(f"   Chatbot exists: {self.conversation_handler.chatbot is not None}")
                            self._debug_system_state()
                        self._switch_to_wake_word_mode()
                    else:
                        if config.DEBUG_MODE:
                            print("✅ Conversation started successfully")
                except Exception as conv_error:
                    print(f"❌ Error starting conversation: {conv_error}")
                    if config.DEBUG_MODE:
                        import traceback
                        traceback.print_exc()
                        self._debug_system_state()
                    self._switch_to_wake_word_mode()
            else:
                print("❌ No conversation handler available")
                if config.DEBUG_MODE:
                    self._debug_system_state()
                self._switch_to_wake_word_mode()
                    
        except Exception as e:
            print(f"❌ Error in wake word detection callback: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            # Force reset state on any error
            try:
                self._conversation_ended = False
                self._switching_state = False
                self._switch_to_wake_word_mode()
            except Exception as recovery_error:
                print(f"❌ Error in wake word callback recovery: {recovery_error}")
    
    def _on_terminal_word_detected(self):
        """Callback when terminal word is detected via OpenWakeWord (backup detection)."""
        if config.DEBUG_MODE:
            print("🛑 Terminal word detected via OpenWakeWord (backup) - ending conversation")
        
        # This provides backup terminal detection in case transcription-based detection fails
        # End the conversation immediately
        if self.conversation_handler:
            self.conversation_handler.end_conversation_immediately()
        
        # Switch back to wake word mode
        self._switch_to_wake_word_mode()
    
    def _on_conversation_end(self):
        """Callback when conversation ends."""
        try:
            # Prevent double callback processing
            if self._conversation_ended:
                if config.DEBUG_MODE:
                    print("⏭️ Conversation end already processed, skipping")
                return
                
            self._conversation_ended = True
            
            # Clear session summary when conversation ends
            try:
                from main import clear_session_summary
                clear_session_summary()
            except Exception as e:
                print(f"⚠️ Error clearing session summary: {e}")
            
            if config.DEBUG_MODE:
                print("💬 Conversation ended - returning to wake word detection")
                print(f"🔍 Pre-restart state check:")
                if self.conversation_handler:
                    print(f"   Conversation handler state: {self.conversation_handler.state.value}")
                    print(f"   Conversation running: {getattr(self.conversation_handler, 'running', 'N/A')}")
                    print(f"   Conversation thread alive: {self.conversation_handler.conversation_thread.is_alive() if self.conversation_handler.conversation_thread else 'No thread'}")
                if self.wake_word_detector:
                    print(f"   Wake word detector state: {self.wake_word_detector.state.value}")
            
            # Force cleanup of conversation handler state
            if self.conversation_handler:
                try:
                    # Ensure conversation handler is properly reset
                    self.conversation_handler.running = False
                    if hasattr(self.conversation_handler, 'conversation_thread'):
                        self.conversation_handler.conversation_thread = None
                    if config.DEBUG_MODE:
                        print("🧹 Conversation handler state cleaned up")
                except Exception as cleanup_error:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Error cleaning conversation handler: {cleanup_error}")
            
            # Add a small delay to ensure conversation resources are fully released
            import time
            time.sleep(0.2)
            
            # Restart the wake word detector with comprehensive error handling
            try:
                self._restart_wake_word_detector()
            except Exception as restart_error:
                print(f"❌ Critical error during wake word restart: {restart_error}")
                if config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                
                # Attempt emergency recovery
                self._emergency_wake_word_recovery()
            
        except Exception as e:
            print(f"❌ Critical error in conversation end callback: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            
            # Ensure state is reset even if restart fails
            self._conversation_ended = True
            
            # Force reset switching state to prevent getting stuck
            self._switching_state = False
            
            # Attempt emergency recovery
            try:
                self._emergency_wake_word_recovery()
            except Exception as recovery_error:
                print(f"💥 Emergency recovery also failed: {recovery_error}")
                print("🚨 System may require manual restart")
                
                # Final fallback - force reset all states
                try:
                    self._conversation_ended = True
                    self._switching_state = False
                    if self.conversation_handler:
                        self.conversation_handler.running = False
                        self.conversation_handler.conversation_thread = None
                    print("🔄 Forced state reset completed")
                except Exception as final_error:
                    print(f"💥 Final fallback also failed: {final_error}")
                    print("🚨 System requires immediate restart")
    
    def _emergency_wake_word_recovery(self):
        """Emergency recovery procedure when normal wake word restart fails."""
        try:
            if config.DEBUG_MODE:
                print("🚨 Attempting emergency wake word recovery...")
            
            # Force reset conversation ended flag
            self._conversation_ended = True
            self._switching_state = False
            
            # Update state manager to asleep
            try:
                from core.state_management.statemanager import StateManager
                state_manager = StateManager()
                state_manager.set("chat_phase", "asleep")
                if config.DEBUG_MODE:
                    print("📝 Emergency: State updated to 'asleep'")
            except Exception as e:
                print(f"⚠️ Failed to update state manager in emergency recovery: {e}")
            
            # Try to recreate wake word detector if current one is broken
            if self.wake_word_detector:
                try:
                    # Force stop current detector
                    self.wake_word_detector.running = False
                    if hasattr(self.wake_word_detector, '_shutdown_event'):
                        self.wake_word_detector._shutdown_event.set()
                    if hasattr(self.wake_word_detector, 'audio_stream') and self.wake_word_detector.audio_stream:
                        try:
                            self.wake_word_detector.audio_stream.stop()
                            self.wake_word_detector.audio_stream.close()
                        except:
                            pass
                    self.wake_word_detector.state = ComponentState.STOPPED
                    
                    # Try to start it again
                    import time
                    time.sleep(0.5)  # Longer delay for emergency recovery
                    
                    if self.wake_word_detector.start():
                        print("✅ Emergency wake word recovery successful")
                        return
                    
                except Exception as e:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Emergency detector restart failed: {e}")
            
            print("⚠️ Emergency recovery completed but wake word may not be functional")
            print("💡 Try saying the wake word or restart the system if needed")
            
        except Exception as e:
            print(f"💥 Emergency recovery procedure failed: {e}")
            print("🚨 System requires manual restart")
    
    def _validate_and_fix_system_state(self) -> bool:
        """Validate system state and fix any inconsistencies."""
        try:
            if config.DEBUG_MODE:
                print("🔍 Validating system state...")
            
            state_issues = []
            fixes_applied = []
            
            # Check if switching state is stuck
            if self._switching_state:
                state_issues.append("Switching state stuck")
                self._switching_state = False
                fixes_applied.append("Reset switching state")
            
            # Check conversation handler state
            if self.conversation_handler:
                # Check for stuck running state without active thread
                if (self.conversation_handler.running and 
                    (not self.conversation_handler.conversation_thread or 
                     not self.conversation_handler.conversation_thread.is_alive())):
                    state_issues.append("Conversation handler stuck in running state")
                    self.conversation_handler.running = False
                    self.conversation_handler.conversation_thread = None
                    fixes_applied.append("Reset conversation handler state")
                
                # Check for unhealthy state when it should be running
                if (self.conversation_handler.state == ComponentState.RUNNING and 
                    not self.conversation_handler.is_healthy()):
                    state_issues.append("Conversation handler unhealthy")
                    try:
                        self.conversation_handler.stop()
                        time.sleep(0.1)
                        self.conversation_handler.start()
                        fixes_applied.append("Restarted conversation handler")
                    except Exception as e:
                        state_issues.append(f"Failed to restart conversation handler: {e}")
            
            # Check wake word detector state
            if self.wake_word_detector:
                # Check for stuck detection thread
                if (self.wake_word_detector.detection_thread and 
                    self.wake_word_detector.detection_thread.is_alive() and 
                    not self.wake_word_detector.running):
                    state_issues.append("Wake word detection thread stuck")
                    if hasattr(self.wake_word_detector, '_shutdown_event'):
                        self.wake_word_detector._shutdown_event.set()
                    fixes_applied.append("Signaled shutdown to stuck detection thread")
                
                # Check for unhealthy state when it should be running
                if (self.wake_word_detector.state == ComponentState.RUNNING and 
                    not self.wake_word_detector.is_healthy()):
                    state_issues.append("Wake word detector unhealthy")
                    try:
                        self.wake_word_detector.stop()
                        time.sleep(0.1)
                        self.wake_word_detector.start()
                        fixes_applied.append("Restarted wake word detector")
                    except Exception as e:
                        state_issues.append(f"Failed to restart wake word detector: {e}")
            
            if config.DEBUG_MODE:
                if state_issues:
                    print(f"🔧 State issues found: {', '.join(state_issues)}")
                    print(f"✅ Fixes applied: {', '.join(fixes_applied)}")
                else:
                    print("✅ System state validation passed")
            
            return len(state_issues) == 0
            
        except Exception as e:
            print(f"❌ Error validating system state: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return False
    
    def _debug_system_state(self):
        """Print comprehensive system state for debugging."""
        try:
            print("🔍 === SYSTEM STATE DEBUG ===")
            print(f"Orchestrator running: {self.running}")
            print(f"Mode: {self.mode}")
            print(f"Switching state: {self._switching_state}")
            print(f"Conversation ended: {self._conversation_ended}")
            
            if self.conversation_handler:
                print(f"ConversationHandler:")
                print(f"  State: {self.conversation_handler.state.value}")
                print(f"  Running: {getattr(self.conversation_handler, 'running', 'N/A')}")
                print(f"  Healthy: {self.conversation_handler.is_healthy()}")
                print(f"  Chatbot exists: {self.conversation_handler.chatbot is not None}")
                print(f"  Thread exists: {self.conversation_handler.conversation_thread is not None}")
                print(f"  Thread alive: {self.conversation_handler.conversation_thread.is_alive() if self.conversation_handler.conversation_thread else 'N/A'}")
            
            if self.wake_word_detector:
                print(f"WakeWordDetector:")
                print(f"  State: {self.wake_word_detector.state.value}")
                print(f"  Running: {self.wake_word_detector.running}")
                print(f"  Healthy: {self.wake_word_detector.is_healthy()}")
                print(f"  Model loaded: {self.wake_word_detector.oww_model is not None}")
                print(f"  Detection thread exists: {self.wake_word_detector.detection_thread is not None}")
                print(f"  Detection thread alive: {self.wake_word_detector.detection_thread.is_alive() if self.wake_word_detector.detection_thread else 'N/A'}")
                print(f"  Using shared stream: {self.wake_word_detector.using_shared_stream}")
                print(f"  Audio access: {self.wake_word_detector.has_audio_access}")
                if self.wake_word_detector.using_shared_stream:
                    print(f"  Subscribed to shared stream: {self.wake_word_detector.audio_manager.is_subscribed(self.wake_word_detector.name)}")
            
            if self.terminal_word_detector:
                print(f"TerminalWordDetector:")
                print(f"  State: {self.terminal_word_detector.state.value}")
                print(f"  Running: {self.terminal_word_detector.running}")
                print(f"  Healthy: {self.terminal_word_detector.is_healthy()}")
                print(f"  Detection thread exists: {self.terminal_word_detector.detection_thread is not None}")
                print(f"  Detection thread alive: {self.terminal_word_detector.detection_thread.is_alive() if self.terminal_word_detector.detection_thread else 'N/A'}")
            
            print("🔍 === END DEBUG ===")
            
        except Exception as e:
            print(f"❌ Error printing debug state: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    def _restart_wake_word_detector(self):
        """Restart the wake word detector after conversation ends."""
        try:
            if config.DEBUG_MODE:
                print("🔄 Restarting wake word detector...")
            
            if not self.wake_word_detector:
                print("❌ No wake word detector available")
                return
            
            # Ensure complete cleanup first
            if self.wake_word_detector.state != ComponentState.STOPPED:
                if config.DEBUG_MODE:
                    print(f"🛑 Wake word detector not stopped (state: {self.wake_word_detector.state.value}), stopping first...")
                
                if not self.wake_word_detector.stop():
                    print("❌ Failed to stop wake word detector before restart")
                    return
                
                # Brief pause to ensure cleanup is complete
                import time
                time.sleep(0.3)  # Increased from 0.2s
            
            # Verify audio access is available
            if not self._verify_audio_access_available():
                print("❌ Audio access not available for wake word detector restart")
                # Try to wait a bit longer and retry
                import time
                time.sleep(0.5)
                if not self._verify_audio_access_available():
                    print("❌ Audio access still not available after delay")
                    return
            
            # Attempt restart with more attempts and better error handling
            restart_success = False
            max_attempts = 5  # Increased from 3
            
            for attempt in range(max_attempts):
                if config.DEBUG_MODE and attempt > 0:
                    print(f"🔄 Wake word restart attempt {attempt + 1}/{max_attempts}")
                
                try:
                    if self.wake_word_detector.start():
                        restart_success = True
                        if config.DEBUG_MODE:
                            print("🎤 Wake word detector restarted successfully")
                        break
                    else:
                        if attempt < max_attempts - 1:
                            print(f"⚠️ Wake word restart attempt {attempt + 1} failed, retrying...")
                            import time
                            time.sleep(0.5)  # Increased delay between attempts
                        else:
                            print("❌ Failed to restart wake word detector after multiple attempts")
                            
                except Exception as start_error:
                    print(f"⚠️ Exception during wake word start attempt {attempt + 1}: {start_error}")
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(0.5)
                    else:
                        print("❌ All restart attempts failed with exceptions")
            
            if restart_success:
                # Update state manager only on successful restart
                from core.state_management.statemanager import StateManager
                state_manager = StateManager()
                state_manager.set("chat_phase", "asleep")
                if config.DEBUG_MODE:
                    print("📝 State manager updated to 'asleep'")
            else:
                print("💥 Wake word detector restart failed - triggering emergency recovery")
                # Don't immediately fail - try emergency recovery
                try:
                    self._emergency_wake_word_recovery()
                except Exception as emergency_error:
                    print(f"💥 Emergency recovery also failed: {emergency_error}")
                    print("🚨 System may require manual restart, but continuing to try...")
            
        except Exception as e:
            print(f"⚠️ Error restarting wake word detector: {e}")
            import traceback
            if config.DEBUG_MODE:
                traceback.print_exc()
            
            # Try emergency recovery even if restart function failed
            try:
                print("🔄 Attempting emergency recovery due to restart error...")
                self._emergency_wake_word_recovery()
            except Exception as emergency_error:
                print(f"💥 Emergency recovery failed: {emergency_error}")
                print("🚨 System may require manual restart, but continuing to try...")
    
    def _verify_audio_access_available(self) -> bool:
        """Verify that audio access is available for wake word detector."""
        try:
            # Check if audio manager shows resources as available
            if hasattr(self, 'wake_word_detector') and self.wake_word_detector:
                audio_manager = self.wake_word_detector.audio_manager
                
                # Check current audio usage
                if audio_manager.audio_in_use:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Audio still in use by: {audio_manager.current_user}")
                    
                    # Wait briefly for resources to be released
                    import time
                    for i in range(5):  # Wait up to 0.5 seconds
                        time.sleep(0.1)
                        if not audio_manager.audio_in_use:
                            break
                    
                    if audio_manager.audio_in_use:
                        return False
                
                # Try to request audio access temporarily to verify availability
                if audio_manager.request_audio_access("VerificationTest", timeout=1.0):
                    audio_manager.release_audio_access("VerificationTest")
                    if config.DEBUG_MODE:
                        print("✅ Audio access verification successful")
                    return True
                else:
                    if config.DEBUG_MODE:
                        print("❌ Audio access verification failed")
                    return False
            
            return True  # Default to true if no audio manager
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Error verifying audio access: {e}")
            return False
    
    def _switch_to_listening_mode(self):
        """Switch to listening mode: properly stop wake word detector for conversation."""
        if self._switching_state:
            if config.DEBUG_MODE:
                print("⏭️ State switch already in progress, skipping listening mode switch")
            return
            
        try:
            self._switching_state = True
            if config.DEBUG_MODE:
                print("🔄 Switching to listening mode...")
                
            # Properly stop wake word detector (not just set running=False)
            if self.wake_word_detector and self.wake_word_detector.state == ComponentState.RUNNING:
                if config.DEBUG_MODE:
                    print("🛑 Properly stopping wake word detector for conversation...")
                
                stop_success = self.wake_word_detector.stop()
                if stop_success:
                    if config.DEBUG_MODE:
                        print("✅ Wake word detector stopped successfully")
                else:
                    print("⚠️ Wake word detector stop had issues, but continuing...")
                
                # Brief pause to ensure cleanup is complete
                import time
                time.sleep(0.2)
            
            # Terminal word detector will be started by the StreamingChatbot after shared stream creation
            # This avoids timing issues with shared stream availability
            if config.DEBUG_MODE and config.TERMINAL_WORD_ENABLED:
                print("ℹ️ Terminal word detection: will be started by conversation handler with shared stream")
            
            if config.DEBUG_MODE:
                print("💬 Audio resources configured for conversation")
                    
            # Update state manager
            from core.state_management.statemanager import StateManager
            state_manager = StateManager()
            state_manager.set("chat_phase", "listening (active)")
            if config.DEBUG_MODE:
                print("📝 State updated to 'listening (active)'")
            
        except Exception as e:
            print(f"⚠️ Error switching to listening mode: {e}")
            import traceback
            if config.DEBUG_MODE:
                traceback.print_exc()
        finally:
            self._switching_state = False
    
    def _switch_to_wake_word_mode(self):
        """Switch to wake word mode: stop terminal word detector, start wake word detector."""
        # Use the same logic as asleep mode since both need to activate wake word detection
        self._switch_to_asleep_mode()
    
    def _switch_to_asleep_mode(self):
        """Switch to asleep mode: stop terminal word detector, start wake word detector."""
        if self._switching_state:
            if config.DEBUG_MODE:
                print("⏭️ State switch already in progress, skipping asleep mode switch")
            return
            
        try:
            self._switching_state = True
            if config.DEBUG_MODE:
                print("🔄 Switching to asleep mode...")
                
            # Stop terminal detector (handles both exclusive and shared stream modes)
            if self.terminal_word_detector and self.terminal_word_detector.state == ComponentState.RUNNING:
                if config.DEBUG_MODE:
                    print("🛑 Stopping terminal word detector...")
                self.terminal_word_detector.stop()
                if config.DEBUG_MODE:
                    print("🔇 Terminal word detector stopped")
            
            # Start wake word detector
            if self.wake_word_detector:
                if self.wake_word_detector.state == ComponentState.STOPPED:
                    # Small delay to allow audio resources to be released
                    time.sleep(0.1)
                    if self.wake_word_detector.start():
                        if config.DEBUG_MODE:
                            print("🎤 Wake word detector started successfully")
                    else:
                        print("❌ Failed to start wake word detector")
                elif self.wake_word_detector.state == ComponentState.RUNNING:
                    if config.DEBUG_MODE:
                        print("✅ Wake word detector already running")
                else:
                    if config.DEBUG_MODE:
                        print(f"⚠️ Wake word detector in unexpected state: {self.wake_word_detector.state.value}")
                        
            # Update state manager
            from core.state_management.statemanager import StateManager
            state_manager = StateManager()
            state_manager.set("chat_phase", "asleep")
            if config.DEBUG_MODE:
                print("📝 State manager updated to 'asleep'")
            
        except Exception as e:
            print(f"⚠️ Error switching to asleep mode: {e}")
        finally:
            self._switching_state = False
    
    def _start_continuous_conversation(self):
        """Start conversation immediately for continuous mode."""
        if self.conversation_handler:
            self.conversation_handler.start_conversation()
    
    def _start_interactive_mode(self):
        """Handle interactive mode where user manually triggers conversations."""
        def interactive_loop():
            while self.running:
                try:
                    input()  # Wait for Enter key
                    if self.running and self.conversation_handler:
                        print("🎤 Starting conversation...")
                        self.conversation_handler.start_conversation()
                except (EOFError, KeyboardInterrupt):
                    break
        
        # Start interactive loop in background thread
        interactive_thread = threading.Thread(target=interactive_loop, daemon=True)
        interactive_thread.start()
    
    def _health_check(self):
        """Check health of all components and restart if needed."""
        if not config.AUTO_RESTART:
            return
        
        # Rate limiting for health check restarts to prevent restart loops
        current_time = time.time()
        if not hasattr(self, '_last_health_check_restart'):
            self._last_health_check_restart = {}
        
        for name, component in self.components.items():
            try:
                # Check if component is unhealthy while supposedly running
                if component.state == ComponentState.RUNNING and not component.is_healthy():
                    # Rate limit restarts - minimum 30 seconds between restarts per component
                    last_restart = self._last_health_check_restart.get(name, 0)
                    time_since_restart = current_time - last_restart
                    min_restart_interval = 30.0  # 30 seconds minimum
                    
                    if time_since_restart < min_restart_interval:
                        if config.DEBUG_MODE:
                            print(f"⏭️ Health check restart rate limited for {name}: {time_since_restart:.1f}s < {min_restart_interval}s")
                        continue
                    
                    if config.DEBUG_MODE:
                        print(f"⚠️ {name} unhealthy, attempting restart...")
                    
                    # Record restart attempt time
                    self._last_health_check_restart[name] = current_time
                    
                    # Attempt graceful restart
                    try:
                        if component.stop():
                            time.sleep(1.0)
                            if component.start():
                                if config.DEBUG_MODE:
                                    print(f"✅ {name} restarted successfully via health check")
                            else:
                                print(f"❌ Health check restart failed for {name}")
                        else:
                            print(f"❌ Health check stop failed for {name}")
                    except Exception as restart_error:
                        print(f"❌ Health check restart exception for {name}: {restart_error}")
                        if config.DEBUG_MODE:
                            import traceback
                            traceback.print_exc()
                            
            except Exception as health_error:
                if config.DEBUG_MODE:
                    print(f"⚠️ Health check error for {name}: {health_error}")
                continue
    
    def _play_startup_sound(self):
        """Play startup sound if available."""
        try:
            startup_dir = "./audio_data/startup_audio"
            if os.path.exists(startup_dir):
                audio_files = [f for f in os.listdir(startup_dir) 
                              if f.endswith(('.wav', '.mp3', '.mov'))]
                if audio_files:
                    startup_file = random.choice(audio_files)
                    audio_path = os.path.join(startup_dir, startup_file)
                    subprocess.Popen(['afplay', audio_path], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Startup sound error: {e}")