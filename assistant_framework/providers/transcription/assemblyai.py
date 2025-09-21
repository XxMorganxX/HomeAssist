"""
AssemblyAI WebSocket streaming transcription provider.
"""

import asyncio
import json
import pyaudio
import websocket
import threading
from typing import AsyncIterator, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlencode

try:
    # Try relative imports first (when used as package)
    from ...interfaces.transcription import TranscriptionInterface
    from ...models.data_models import TranscriptionResult
    from ...utils.audio_manager import get_audio_manager
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.transcription import TranscriptionInterface
    from models.data_models import TranscriptionResult
    from utils.audio_manager import get_audio_manager


class AssemblyAITranscriptionProvider(TranscriptionInterface):
    """AssemblyAI WebSocket streaming implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AssemblyAI provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: AssemblyAI API key
                - sample_rate: Audio sample rate (default: 16000)
                - format_turns: Whether to format turns (default: True)
        """
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required")
        
        self.sample_rate = config.get('sample_rate', 16000)
        self.format_turns = config.get('format_turns', True)
        
        # WebSocket configuration
        self.connection_params = {
            "sample_rate": self.sample_rate,
            "format_turns": self.format_turns,
        }
        self.api_endpoint_base = "wss://streaming.assemblyai.com/v3/ws"
        self.api_endpoint = f"{self.api_endpoint_base}?{urlencode(self.connection_params)}"
        
        # Audio configuration
        self.frames_per_buffer = 800  # 50ms of audio at 16kHz
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Audio callback for sharing data with other components (e.g., VAD)
        self.audio_callback = None
        
        # State management
        self.audio = None
        self.stream = None
        self.ws_app = None
        self.audio_thread = None
        self.ws_thread = None
        self.stop_event = threading.Event()
        self._is_active = False
        self.audio_manager = get_audio_manager()
        self._cleanup_lock = threading.Lock()
        
        # Async queue for transcription results
        self.result_queue = asyncio.Queue()
        self.session_id = None
        
    async def initialize(self) -> bool:
        """Initialize the AssemblyAI provider."""
        try:
            # Audio will be acquired when needed
            return True
        except Exception as e:
            print(f"Failed to initialize AssemblyAI provider: {e}")
            return False
    
    async def start_streaming(self) -> AsyncIterator[TranscriptionResult]:
        """Start streaming transcription from audio input."""
        if self._is_active:
            return
        
        try:
            # Store the current event loop for the WebSocket callback
            self._loop = asyncio.get_running_loop()
            
            # Acquire audio resources (force cleanup if necessary for clean state)
            print("🎯 Transcription attempting to acquire audio...")
            status_before = self.audio_manager.get_status()
            print(f"🔍 Audio status before acquisition: {status_before}")

            self.audio = self.audio_manager.acquire_audio("transcription", force_cleanup=True)
            if not self.audio:
                status_after = self.audio_manager.get_status()
                print(f"❌ Audio acquisition failed. Status: {status_after}")
                raise RuntimeError("Failed to acquire audio resources")

            print(f"✅ Transcription acquired audio successfully: {type(self.audio)}")

            # Open microphone stream
            print("🎙️  Opening transcription audio stream...")
            try:
                self.stream = self.audio.open(
                    input=True,
                    frames_per_buffer=self.frames_per_buffer,
                    channels=self.channels,
                    format=self.format,
                    rate=self.sample_rate,
                )
                print("✅ Transcription audio stream opened successfully")
            except Exception as e:
                print(f"❌ Failed to open transcription audio stream: {e}")
                raise
            
            # Create WebSocket app
            print("🌐 Creating WebSocket connection...")
            self.ws_app = websocket.WebSocketApp(
                self.api_endpoint,
                header={"Authorization": self.api_key},
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            print("✅ WebSocket app created")
            
            # Start WebSocket in thread
            print("🔄 Starting WebSocket thread...")
            self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            print("✅ WebSocket thread started")

            self._is_active = True
            print("🎯 Starting transcription result loop...")

            # Yield transcription results as they arrive
            while self._is_active:
                try:
                    # Use wait_for to allow interruption
                    result = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=0.1
                    )
                    yield result
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error getting transcription result: {e}")
                    break
                    
        except Exception as e:
            print(f"Error starting AssemblyAI streaming: {e}")
            self._is_active = False
            raise
    
    async def stop_streaming(self) -> None:
        """Stop the streaming transcription."""
        if not self._is_active:
            return
        
        # Clear audio callback immediately to prevent further data flow
        self.audio_callback = None
        
        with self._cleanup_lock:
            # Set flags first to stop all loops
            self.stop_event.set()
            self._is_active = False
            
            # Stop audio stream (before closing WebSocket).
            try:
                if self.stream:
                    print("🎤 Stopping AssemblyAI audio stream")
                    if hasattr(self.stream, 'is_active') and self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                    print("✅ AssemblyAI audio stream stopped")
            except Exception as e:
                print(f"⚠️  Error stopping AssemblyAI stream: {e}")
                self.stream = None
            
            # Send termination message if connected
            try:
                if self.ws_app and hasattr(self.ws_app, 'sock') and self.ws_app.sock and self.ws_app.sock.connected:
                    terminate_message = {"type": "Terminate"}
                    self.ws_app.send(json.dumps(terminate_message))
                    # Small delay to allow message to be sent
                    await asyncio.sleep(0.05)
            except Exception:
                pass
            
            # Close WebSocket
            try:
                if self.ws_app:
                    self.ws_app.close()
            except Exception:
                pass
            
            # Wait for threads with short timeout
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=0.2)
            
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.2)
            
            # Release audio resources only if we own them
            if self.audio:
                try:
                    # Check if we still own the audio before releasing
                    status = self.audio_manager.get_status()
                    if status['current_owner'] == "transcription":
                        self.audio_manager.release_audio("transcription", force_cleanup=False)
                        print("✅ Transcription audio resources released")
                    elif status['current_owner'] is None:
                        print("ℹ️  Audio already released (no owner)")
                    else:
                        print(f"ℹ️  Audio owned by {status['current_owner']}, skipping release")
                except Exception as e:
                    print(f"⚠️  Error releasing AssemblyAI audio: {e}")
                finally:
                    self.audio = None
            
            # Clear any remaining results from the queue
            try:
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except Exception:
                        break
            except Exception:
                pass
            
            # Reset state for next session
            self.stop_event.clear()  # Reset the threading event
            self.ws_app = None
            self.ws_thread = None
            self.audio_thread = None
            self._loop = None
            self.session_id = None
        
        # Session reset complete
    
    def set_audio_callback(self, callback):
        """Set callback to receive audio chunks."""
        self.audio_callback = callback
    
    @property
    def is_active(self) -> bool:
        """Check if transcription is currently active."""
        return self._is_active
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_streaming()
        
        # Audio cleanup is handled by the audio manager
        self.audio = None
    
    def _on_open(self, ws):
        """WebSocket opened callback."""
        def stream_audio():
            while not self.stop_event.is_set():
                try:
                    audio_data = self.stream.read(
                        self.frames_per_buffer, 
                        exception_on_overflow=False
                    )
                    
                    if self.stop_event.is_set():
                        break
                    
                    # Send to VAD if callback is set
                    if self.audio_callback:
                        try:
                            self.audio_callback(audio_data)
                        except Exception as e:
                            print(f"Audio callback error: {e}")
                    
                    # Check if WebSocket is still open before sending
                    if ws.sock and ws.sock.connected:
                        ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                    else:
                        break  # Exit if connection is closed
                except Exception as e:
                    print(f"Error streaming audio: {e}")
                    break
        
        self.audio_thread = threading.Thread(target=stream_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _on_message(self, ws, message):
        """WebSocket message received callback."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == "Begin":
                self.session_id = data.get('id')
                
            elif msg_type == "Turn":
                transcript = data.get('transcript', '')
                is_formatted = data.get('turn_is_formatted', False)
                
                if transcript:
                    # Create transcription result
                    result = TranscriptionResult(
                        text=transcript,
                        is_final=is_formatted,
                        timestamp=datetime.now().timestamp(),
                        metadata={'session_id': self.session_id}
                    )
                    
                    # Put result in async queue using the stored loop
                    if hasattr(self, '_loop') and self._loop:
                        asyncio.run_coroutine_threadsafe(
                            self.result_queue.put(result),
                            self._loop
                        )
                        
            elif msg_type == "Termination":
                self._is_active = False
                
        except json.JSONDecodeError as e:
            print(f"Error decoding AssemblyAI message: {e}")
        except Exception as e:
            print(f"Error handling AssemblyAI message: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error callback."""
        print(f"AssemblyAI WebSocket error: {error}")
        self.stop_event.set()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed callback."""
        # Do not close the audio stream here to avoid double-close races.
        self.stop_event.set()
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'batch': False,
            'languages': ['en-US'],
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'features': ['turn_formatting', 'real_time']
        }