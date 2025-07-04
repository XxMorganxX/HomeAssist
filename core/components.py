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
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models
from enum import Enum

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
        self.audio: Optional[pyaudio.PyAudio] = None
        self.mic_stream = None
        self.detection_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Detection state management
        self.last_activation_time = 0
        self.previous_scores = {}
        self.is_armed = {}
        
        # Audio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1280
        
        # Paths
        self.word_model_dir = "./audio_data/wake_word_models"
        self.word_model_path = f"{self.word_model_dir}/{model_name}.tflite"
        self.opener_audio_dir = "./audio_data/opener_audio"
        
    def start(self) -> bool:
        """Start wake word detection."""
        try:
            self.state = ComponentState.STARTING
            
            # Ensure model directory exists
            if not os.path.exists(self.word_model_path):
                os.makedirs(self.word_model_dir, exist_ok=True)
                download_models(target_directory=self.word_model_dir)
            
            # Initialize OpenWakeWord model
            self.oww_model = Model(
                wakeword_models=[self.word_model_path], 
                inference_framework='tflite'
            )
            
            # Initialize state tracking
            for model_name in self.oww_model.models.keys():
                self.previous_scores[model_name] = 0
                self.is_armed[model_name] = True
            
            # Initialize audio with retry logic for macOS
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            for attempt in range(max_retries):
                try:
                    if self.audio:
                        self.audio.terminate()
                    
                    self.audio = pyaudio.PyAudio()
                    
                    # Get default input device info
                    default_input = self.audio.get_default_input_device_info()
                    if config.DEBUG_MODE:
                        print(f"Using audio input device: {default_input['name']}")
                    
                    # Try to match device's native sample rate first
                    device_rate = int(default_input['defaultSampleRate'])
                    
                    # Open stream with device's native rate
                    self.mic_stream = self.audio.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=device_rate,
                        input=True,
                        frames_per_buffer=self.CHUNK
                    )
                    
                    # Test the stream
                    test_data = self.mic_stream.read(self.CHUNK, exception_on_overflow=False)
                    if len(test_data) > 0:
                        break
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Audio initialization attempt {attempt + 1} failed: {e}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"Failed to initialize audio after {max_retries} attempts: {e}")
            
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
            
            # Wait for detection thread to finish
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2.0)
            
            # Clean up audio resources
            if self.mic_stream:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
                self.mic_stream = None
            
            if self.audio:
                self.audio.terminate()
                self.audio = None
            
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
        """Check if wake word detector is healthy."""
        return (self.state == ComponentState.RUNNING and 
                self.running and 
                self.detection_thread and 
                self.detection_thread.is_alive() and
                self.mic_stream and
                self.oww_model)
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        if config.DEBUG_MODE:
            print("🔊 Wake word detection started...")
        
        while self.running:
            try:
                # Get audio frame with error handling
                try:
                    audio_data = self.mic_stream.read(self.CHUNK, exception_on_overflow=False)
                except OSError as e:
                    if "Input overflowed" in str(e):
                        # Skip this frame and continue
                        continue
                    else:
                        print(f"❌ Audio stream error: {e}")
                        break
                
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
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
                    
                    # Rising edge detection with cooldown
                    if (self.is_armed[model_name] and
                        previous_score <= self.threshold and 
                        current_score > self.threshold and 
                        time_since_last > self.cooldown_seconds):
                        
                        if config.DEBUG_MODE:
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
    
    def _conversation_loop(self):
        """Run conversation until it ends."""
        try:
            # Run the chatbot
            self.chatbot.run()
            
            # Conversation ended
            if config.DEBUG_MODE:
                print("💬 Conversation ended")
            
            # Notify callback
            if self.conversation_end_callback:
                self.conversation_end_callback()
                
        except Exception as e:
            print(f"❌ Conversation error: {e}")
        finally:
            self.running = False


class ComponentOrchestrator:
    """Central orchestrator for managing all system components."""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.running = False
        self.mode = config.CONVERSATION_MODE
        
        # Component instances
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.conversation_handler: Optional[ConversationHandler] = None
        
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
                    model_name="hey_monkey",
                    threshold=0.3,
                    cooldown_seconds=2.0
                )
                self.components["wake_word"] = self.wake_word_detector
            
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
            
            # Start wake word detector if enabled
            if self.wake_word_detector and not self.wake_word_detector.start():
                return False
            
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
            self.running = False
            
            success = True
            for name, component in self.components.items():
                if not component.stop():
                    success = False
            
            print("👋 Voice assistant stopped")
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
        
        if self.conversation_handler:
            self.conversation_handler.start_conversation()
    
    def _on_conversation_end(self):
        """Callback when conversation ends."""
        if config.DEBUG_MODE:
            print("💬 Conversation ended - returning to wake word detection")
        
        # In wake word mode, we automatically return to listening
        # No action needed - wake word detector keeps running
    
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