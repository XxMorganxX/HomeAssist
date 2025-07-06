"""
Core components for the RasPi Smart Home voice assistant.
Provides modular, configurable components for wake word detection,
conversation handling, and system orchestration.
"""

import os
import sys
import time
import random
import threading
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
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
    
    def create_shared_stream(self, samplerate: int, blocksize: int, channels: int = 1, dtype=np.int16):
        """Create a shared audio stream that multiple components can subscribe to."""
        with self._lock:
            if self._shared_stream is not None:
                return True  # Stream already exists
            
            try:
                self._stream_config = {
                    'samplerate': samplerate,
                    'blocksize': blocksize,
                    'channels': channels,
                    'dtype': dtype
                }
                
                self._shared_stream = sd.RawInputStream(
                    samplerate=samplerate,
                    blocksize=blocksize,
                    channels=channels,
                    dtype=dtype,
                    callback=self._shared_audio_callback
                )
                
                self._shared_stream.start()
                return True
                
            except Exception as e:
                print(f"❌ Failed to create shared audio stream: {e}")
                return False
    
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
        
        # Distribute to all subscribers
        for component_name, callback in self._subscribers.items():
            try:
                callback(audio_array)
            except Exception as e:
                print(f"⚠️ Error in audio callback for {component_name}: {e}")
    
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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from core.streaming_chatbot import ToolEnabledStreamingChatbot
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
        
        # Paths
        self.word_model_dir = "./audio_data/wake_word_models"
        self.word_model_path = f"{self.word_model_dir}/{model_name}.tflite"
        self.opener_audio_dir = "./audio_data/opener_audio"
        
    def start(self) -> bool:
        """Start wake word detection."""
        try:
            self.state = ComponentState.STARTING
            
            # Request audio access first
            if not self.audio_manager.request_audio_access(self.name, timeout=2.0):
                print(f"❌ Could not obtain audio access for {self.name}")
                self.state = ComponentState.ERROR
                return False
            
            self.has_audio_access = True
            
            # Ensure model directory exists
            if not os.path.exists(self.word_model_path):
                os.makedirs(self.word_model_dir, exist_ok=True)
                download_models(target_directory=self.word_model_dir)
            
            # Initialize OpenWakeWord model - try ONNX first, fallback to TFLite
            onnx_model_path = self.word_model_path.replace('.tflite', '.onnx')
            if os.path.exists(onnx_model_path):
                self.oww_model = Model(
                    wakeword_models=[onnx_model_path], 
                    inference_framework='onnx'
                )
                if config.DEBUG_MODE:
                    print(f"Using ONNX model: {onnx_model_path}")
            else:
                self.oww_model = Model(
                    wakeword_models=[self.word_model_path], 
                    inference_framework='tflite'
                )
                if config.DEBUG_MODE:
                    print(f"Using TFLite model: {self.word_model_path}")
            
            # Initialize state tracking
            for model_name in self.oww_model.models.keys():
                self.previous_scores[model_name] = 0
                self.is_armed[model_name] = True
            
            # Initialize audio with SoundDevice
            try:
                # Validate and configure audio device
                device_info = self._validate_audio_device()
                
                # Start the audio stream using validated parameters
                self.audio_stream = sd.RawInputStream(
                    samplerate=self.device_rate,
                    blocksize=self.device_chunk,
                    device=None,  # Use default input device
                    channels=self.CHANNELS,
                    dtype=self.DTYPE,
                    callback=self._audio_callback
                )
                
                # Start the stream
                self.audio_stream.start()
                
                if config.DEBUG_MODE:
                    print(f"🎤 Audio stream started: {self.device_rate} Hz, {self.CHANNELS} channel(s), {self.DTYPE}")
                
            except Exception as e:
                raise Exception(f"Failed to initialize SoundDevice audio: {e}")
            
            # Start detection thread
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.state = ComponentState.RUNNING
            if config.DEBUG_MODE:
                print(f"🎤 {self.name} started successfully")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to start {self.name}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop wake word detection."""
        try:
            self.state = ComponentState.STOPPING
            self.running = False
            
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
                    timeout = 0.5 if config.FAST_SHUTDOWN else 1.0
                    self.detection_thread.join(timeout=timeout)
                    thread_time = time.time() - thread_start
                    if self.detection_thread.is_alive():
                        print(f"⚠️ {self.name} detection thread did not stop gracefully after {thread_time:.2f}s")
                        # Force thread cleanup by setting running to False again
                        self.running = False
                    elif config.DEBUG_MODE:
                        print(f"   Detection thread stopped in {thread_time:.2f}s")
            
            # Clean up audio resources with comprehensive error handling
            audio_cleanup_success = self._cleanup_audio_resources()
            
            # Release audio access
            if self.has_audio_access:
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
            
            self.state = ComponentState.STOPPED
            if config.DEBUG_MODE:
                print(f"🛑 {self.name} stopped {'successfully' if audio_cleanup_success else 'with audio issues'}")
            
            return audio_cleanup_success
            
        except Exception as e:
            self.error_message = str(e)
            self.state = ComponentState.ERROR
            print(f"❌ Failed to stop {self.name}: {e}")
            return False
    
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
                
                # Wait for audio resources to be fully released
                time.sleep(0.15)
                
                if config.DEBUG_MODE:
                    print("🔇 Audio stream cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during audio cleanup: {e}")
            cleanup_success = False
        
        return cleanup_success
    
    def is_healthy(self) -> bool:
        """Check if wake word detector is healthy."""
        return (self.state == ComponentState.RUNNING and 
                self.running and 
                self.detection_thread and 
                self.detection_thread.is_alive() and
                self.audio_stream and
                self.oww_model)
    
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
        
        while self.running:
            try:
                # Get audio frame from buffer
                if not hasattr(self, '_audio_buffer') or not self._audio_buffer:
                    time.sleep(0.01)  # Small delay if no audio data
                    continue
                
                # Get the oldest audio frame
                audio_array = self._audio_buffer.pop(0)
                
                # Debug: Check if we're getting audio data
                if config.DEBUG_MODE:
                    rms = np.sqrt(np.mean(audio_array**2))
                    if rms > 100:  # Only print when there's actual audio
                        print(f"Audio RMS: {rms:.0f}")
                
                # Process through OpenWakeWord
                prediction = self.oww_model.predict(audio_array)
                current_time = time.time()
                
                # Check each model for detection
                for model_name in self.oww_model.prediction_buffer.keys():
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
                    if (self.is_armed[model_name] and
                        previous_score <= self.threshold and 
                        current_score > self.threshold and 
                        time_since_last > self.cooldown_seconds):
                        
                        print(f"🎯 Wake word detected! Score: {current_score:.3f}")
                        
                        self.last_activation_time = current_time
                        self.is_armed[model_name] = False
                        
                        # Play greeting audio
                        self._play_greeting_audio()
                        
                        # Trigger callback
                        if self.wake_word_callback:
                            self.wake_word_callback()
                    
                    # Update previous score
                    self.previous_scores[model_name] = current_score
                    
                    # Re-arm after cooldown
                    if not self.is_armed[model_name] and time_since_last > self.cooldown_seconds:
                        self.is_armed[model_name] = True
                
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                if not self.running:
                    break
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _play_greeting_audio(self):
        """Play a random greeting audio file."""
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
                
                # Play audio in background
                subprocess.Popen(['afplay', audio_path], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Greeting audio error: {e}")


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
        return (self.state == ComponentState.RUNNING and self.chatbot is not None)
    
    def start_conversation(self) -> bool:
        """Start a new conversation session."""
        if not self.is_healthy():
            print("❌ ConversationHandler not ready")
            return False
        
        try:
            if config.DEBUG_MODE:
                print("💬 Starting conversation...")
            
            # Start conversation in separate thread
            self.running = True
            self.conversation_thread = threading.Thread(
                target=self._conversation_loop, 
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
                    messages = self.chatbot.conversation.get_messages()
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
    
    def _conversation_loop(self):
        """Run conversation until it ends."""
        try:
            if config.DEBUG_MODE:
                print("💬 Starting conversation loop...")
            
            # Run the chatbot
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
                
                # Clear accumulated chunks and reset state
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


class ComponentOrchestrator:
    """Central orchestrator for managing all system components."""
    
    def __init__(self):
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
    
    def _on_wake_word_detected(self):
        """Callback when wake word is detected."""
        if config.DEBUG_MODE:
            print("🎯 Wake word detected - starting conversation")
        
        # Reset conversation ended flag
        self._conversation_ended = False
        
        # Switch detector states: disable wake word, enable terminal word
        self._switch_to_listening_mode()
        
        if self.conversation_handler:
            self.conversation_handler.start_conversation()
    
    def _on_terminal_word_detected(self):
        """Callback when terminal word is detected (not used with transcription-based approach)."""
        # This callback is no longer used since we handle terminal phrases via transcription
        # in the conversation handler itself for better reliability
        pass
    
    def _on_conversation_end(self):
        """Callback when conversation ends."""
        try:
            # Prevent double callback processing
            if self._conversation_ended:
                if config.DEBUG_MODE:
                    print("⏭️ Conversation end already processed, skipping")
                return
                
            self._conversation_ended = True
            
            if config.DEBUG_MODE:
                print("💬 Conversation ended - returning to wake word detection")
            
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
            
            # Attempt emergency recovery
            try:
                self._emergency_wake_word_recovery()
            except Exception as recovery_error:
                print(f"💥 Emergency recovery also failed: {recovery_error}")
                print("🚨 System may require manual restart")
    
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
                
                # Brief pause to ensure cleanup
                import time
                time.sleep(0.2)
            
            # Verify audio access is available
            if not self._verify_audio_access_available():
                print("❌ Audio access not available for wake word detector restart")
                return
            
            # Attempt restart
            restart_success = False
            max_attempts = 3
            
            for attempt in range(max_attempts):
                if config.DEBUG_MODE and attempt > 0:
                    print(f"🔄 Wake word restart attempt {attempt + 1}/{max_attempts}")
                
                if self.wake_word_detector.start():
                    restart_success = True
                    if config.DEBUG_MODE:
                        print("🎤 Wake word detector restarted successfully")
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(f"⚠️ Wake word restart attempt {attempt + 1} failed, retrying...")
                        import time
                        time.sleep(0.3)
                    else:
                        print("❌ Failed to restart wake word detector after multiple attempts")
            
            if restart_success:
                # Update state manager only on successful restart
                from core.state_management.statemanager import StateManager
                state_manager = StateManager()
                state_manager.set("chat_phase", "asleep")
                if config.DEBUG_MODE:
                    print("📝 State manager updated to 'asleep'")
            else:
                print("💥 Wake word detector restart failed - system may not respond to wake words")
            
        except Exception as e:
            print(f"⚠️ Error restarting wake word detector: {e}")
            import traceback
            if config.DEBUG_MODE:
                traceback.print_exc()
    
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
                time.sleep(0.1)
            
            # NOTE: Do NOT start terminal word detector here - let conversation handler manage audio
            # The conversation handler needs exclusive microphone access
            if config.DEBUG_MODE:
                print("💬 Audio resources freed for conversation")
                    
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
                
            # Set the running flag to false for terminal detector (safer than stop())
            if self.terminal_word_detector and self.terminal_word_detector.state == ComponentState.RUNNING:
                self.terminal_word_detector.running = False
                # Release audio access immediately to allow wake word detector to start
                if self.terminal_word_detector.has_audio_access:
                    self.terminal_word_detector.audio_manager.release_audio_access(self.terminal_word_detector.name)
                    self.terminal_word_detector.has_audio_access = False
                if config.DEBUG_MODE:
                    print("🔇 Terminal word detector marked for stop")
            
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
        
        for name, component in self.components.items():
            if not component.is_healthy() and component.state == ComponentState.RUNNING:
                if config.DEBUG_MODE:
                    print(f"⚠️ {name} unhealthy, attempting restart...")
                
                component.stop()
                time.sleep(1.0)
                component.start()
    
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