"""
OpenAI Whisper transcription provider.

Uses OpenAI's Whisper API for speech-to-text transcription.
Implements chunked transcription to simulate streaming behavior.

ARCHITECTURE:
- Uses callback-based audio capture (same as AssemblyAI provider)
- Buffers audio and sends to Whisper API in chunks
- Returns partial results as audio accumulates, final on silence/send
"""

import asyncio
import io
import threading
import wave
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

import numpy as np
import sounddevice as sd

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from ...interfaces.transcription import TranscriptionInterface
    from ...models.data_models import TranscriptionResult
    from ...utils.audio_manager import get_audio_manager
    from ..base import StreamingProviderBase
except ImportError:
    from assistant_framework.interfaces.transcription import TranscriptionInterface
    from assistant_framework.models.data_models import TranscriptionResult
    from assistant_framework.utils.audio_manager import get_audio_manager
    from assistant_framework.providers.base import StreamingProviderBase


class OpenAIWhisperProvider(StreamingProviderBase, TranscriptionInterface):
    """
    OpenAI Whisper transcription provider.
    
    Uses chunked transcription approach:
    - Audio captured via callback (same as AssemblyAI)
    - Buffered and sent to Whisper API periodically
    - Returns results as they become available
    
    Configuration options:
    - api_key: OpenAI API key
    - model: Whisper model (default: "whisper-1")
    - sample_rate: Audio sample rate (default: 16000)
    - chunk_duration: Seconds of audio per API call (default: 3.0)
    - language: Language code (default: "en")
    - silence_threshold: Energy threshold for silence detection
    - silence_duration: Seconds of silence to trigger final result
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._component_name = "transcription"
        
        # Validate OpenAI library
        if AsyncOpenAI is None:
            raise ImportError("openai library is required. Install with: pip install openai")
        
        # Configuration
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = config.get('model', 'whisper-1')
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 3.0)  # Seconds per API call
        self.language = config.get('language', 'en')
        self.silence_threshold = config.get('silence_threshold', 0.01)
        self.silence_duration = config.get('silence_duration', 1.5)  # Seconds of silence for final
        
        # Audio settings
        self.frames_per_buffer = 3200  # 200ms at 16kHz
        self.channels = 1
        
        # State
        self.audio_manager = get_audio_manager()
        self._audio_stream: Optional[sd.InputStream] = None
        self._client: Optional[AsyncOpenAI] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._result_queue: Optional[asyncio.Queue] = None
        
        # Audio buffer for chunking
        self._audio_buffer: bytes = b""
        self._accumulated_text: str = ""
        
        # Processing task
        self._process_task: Optional[asyncio.Task] = None
        
        # Thread-safe shutdown coordination
        self._shutdown_flag = threading.Event()
        self._callback_active = threading.Event()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Prefill audio support
        self._prefill_audio: Optional[bytes] = None
        
        # Silence detection
        self._silence_start: Optional[float] = None
        self._last_speech_time: float = 0
        
        # Session tracking
        self.session_id: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            print("✅ OpenAI Whisper client initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {e}")
            return False
    
    def set_prefill_audio(self, audio_bytes: Optional[bytes]) -> None:
        """
        Set audio to transcribe before starting live capture.
        Used for barge-in captured audio.
        """
        self._prefill_audio = audio_bytes
        if audio_bytes:
            duration = len(audio_bytes) / 2 / self.sample_rate
            print(f"📼 Prefill audio set: {duration:.2f}s")
    
    async def preconnect(self) -> None:
        """
        Pre-initialize for faster startup.
        For Whisper, this just ensures the client is ready.
        """
        if not self._client:
            await self.initialize()
        print("⚡ OpenAI Whisper ready for transcription")
    
    @property
    def is_preconnected(self) -> bool:
        """Check if client is ready."""
        return self._client is not None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """
        Audio callback - runs in sounddevice's audio thread.
        Same safe callback approach as AssemblyAI provider.
        """
        if self._shutdown_flag.is_set():
            return
        
        if status:
            print(f"⚠️  Audio callback status: {status}")
        
        try:
            audio_copy = indata.copy()
            
            if self._event_loop and self._audio_queue:
                try:
                    self._event_loop.call_soon_threadsafe(
                        self._queue_audio_threadsafe, audio_copy
                    )
                except RuntimeError:
                    pass
        except Exception as e:
            if not self._shutdown_flag.is_set():
                print(f"⚠️  Audio callback error: {e}")
    
    def _queue_audio_threadsafe(self, audio_data: np.ndarray):
        """Thread-safe queue put."""
        if self._audio_queue and not self._shutdown_flag.is_set():
            try:
                self._audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(audio_data)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
    
    async def _initialize_stream(self):
        """Initialize audio capture."""
        self._event_loop = asyncio.get_running_loop()
        self._shutdown_flag.clear()
        
        # Ensure client is initialized
        if not self._client:
            await self.initialize()
        
        # Acquire audio
        acquired = self.audio_manager.acquire_audio("transcription", force_cleanup=True)
        if not acquired:
            raise RuntimeError("Failed to acquire audio for transcription")
        
        # Create queues
        self._audio_queue = asyncio.Queue(maxsize=50)
        self._result_queue = asyncio.Queue(maxsize=20)
        
        # Reset state
        self._audio_buffer = b""
        self._accumulated_text = ""
        self._silence_start = None
        self._last_speech_time = asyncio.get_event_loop().time()
        self.session_id = f"whisper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Open audio stream
        print("🎙️  Opening Whisper transcription audio stream...")
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.frames_per_buffer,
            callback=self._audio_callback
        )
        self._audio_stream.start()
        self._callback_active.set()
        print("✅ Audio stream opened for Whisper transcription")
        
        # Process prefill audio
        if self._prefill_audio:
            self._audio_buffer = self._prefill_audio
            self._prefill_audio = None
            duration = len(self._audio_buffer) / 2 / self.sample_rate
            print(f"📼 Added {duration:.2f}s prefill audio to buffer")
        
        # Start processing task
        self._process_task = asyncio.create_task(self._process_audio())
    
    async def _cleanup_stream(self, full_cleanup: bool = False):
        """Clean up resources."""
        if (self._process_task is None and
            self._audio_stream is None and
            (full_cleanup or self._client is None)):
            return
        
        print("🧹 Cleaning up Whisper transcription...")
        self._shutdown_flag.set()
        
        # Cancel processing task
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._process_task, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                print("⚠️  Process task cancellation timed out")
        self._process_task = None
        
        # Stop audio stream
        if self._audio_stream:
            try:
                if hasattr(self._audio_stream, 'active') and self._audio_stream.active:
                    self._audio_stream.stop()
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"⚠️  Audio stream stop error: {e}")
        
        self._callback_active.clear()
        
        # Close audio stream
        if self._audio_stream:
            try:
                self._audio_stream.close()
                await asyncio.sleep(0.1)
                print("✅ Audio stream closed")
            except Exception as e:
                print(f"⚠️  Audio stream close error: {e}")
            finally:
                self._audio_stream = None
        
        # Clear queues
        self._audio_queue = None
        self._result_queue = None
        self._event_loop = None
        
        # Release audio
        self.audio_manager.release_audio("transcription", force_cleanup=False)
        
        # Full cleanup - close client
        if full_cleanup and self._client:
            self._client = None
            print("✅ OpenAI client closed")
        
        self.session_id = None
        print("✅ Whisper transcription cleanup complete")
    
    def _detect_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio chunk contains speech based on energy."""
        energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
        return energy > self.silence_threshold
    
    def _audio_to_wav_bytes(self, audio_bytes: bytes) -> bytes:
        """Convert raw PCM bytes to WAV format for Whisper API."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    async def _transcribe_chunk(self, audio_bytes: bytes) -> Optional[str]:
        """Send audio chunk to Whisper API and get transcription."""
        if not audio_bytes or len(audio_bytes) < 1000:  # Skip tiny chunks
            return None
        
        try:
            # Convert to WAV
            wav_bytes = self._audio_to_wav_bytes(audio_bytes)
            
            # Create file-like object for API
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            
            # Call Whisper API
            response = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                response_format="text"
            )
            
            text = response.strip() if response else None
            return text if text else None
            
        except Exception as e:
            print(f"⚠️  Whisper API error: {e}")
            return None
    
    async def _process_audio(self):
        """Process audio chunks and generate transcription results."""
        chunk_bytes_threshold = int(self.chunk_duration * self.sample_rate * 2)  # 2 bytes per sample
        
        try:
            while not self._stop_event.is_set() and not self._shutdown_flag.is_set():
                try:
                    # Get audio from queue
                    if not self._audio_queue:
                        break
                    
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.3
                    )
                    
                    # Check for speech
                    has_speech = self._detect_speech(audio_data)
                    current_time = asyncio.get_event_loop().time()
                    
                    if has_speech:
                        self._last_speech_time = current_time
                        self._silence_start = None
                    else:
                        if self._silence_start is None:
                            self._silence_start = current_time
                    
                    # Add to buffer
                    self._audio_buffer += audio_data.tobytes()
                    
                    # Check if we should transcribe
                    buffer_duration = len(self._audio_buffer) / 2 / self.sample_rate
                    silence_elapsed = (current_time - self._silence_start) if self._silence_start else 0
                    
                    should_transcribe = (
                        len(self._audio_buffer) >= chunk_bytes_threshold or
                        (silence_elapsed >= self.silence_duration and len(self._audio_buffer) > 0)
                    )
                    
                    is_final = silence_elapsed >= self.silence_duration
                    
                    if should_transcribe and len(self._audio_buffer) > 0:
                        # Transcribe current buffer
                        text = await self._transcribe_chunk(self._audio_buffer)
                        
                        if text:
                            # Update accumulated text
                            if is_final:
                                # For final, use the full transcription
                                self._accumulated_text = text
                            else:
                                # For partial, append new text
                                self._accumulated_text = text
                            
                            # Queue result
                            if self._result_queue:
                                result = TranscriptionResult(
                                    text=self._accumulated_text,
                                    is_final=is_final,
                                    timestamp=datetime.now().timestamp(),
                                    metadata={'session_id': self.session_id}
                                )
                                await self._result_queue.put(result)
                        
                        # Clear buffer after transcription
                        if is_final:
                            self._audio_buffer = b""
                            self._accumulated_text = ""
                        else:
                            # Keep recent audio for context overlap
                            overlap_bytes = int(0.5 * self.sample_rate * 2)  # 0.5s overlap
                            if len(self._audio_buffer) > overlap_bytes:
                                self._audio_buffer = self._audio_buffer[-overlap_bytes:]
                
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        print(f"⚠️  Audio processing error: {e}")
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._shutdown_flag.is_set():
                print(f"❌ Audio processing failed: {e}")
    
    async def start_streaming(self) -> AsyncIterator[TranscriptionResult]:
        """Start streaming transcription."""
        await self.start_safe()
        
        try:
            print(f"📝 Whisper session started: {self.session_id}")
            
            while not self._stop_event.is_set() and not self._shutdown_flag.is_set():
                try:
                    if not self._result_queue:
                        break
                    
                    # Get result with timeout
                    result = await asyncio.wait_for(
                        self._result_queue.get(),
                        timeout=0.5
                    )
                    yield result
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
        
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("🛑 Whisper transcription interrupted")
            raise
        except Exception as e:
            print(f"❌ Whisper streaming error: {e}")
            raise
        finally:
            try:
                await self.stop_safe()
            except Exception as cleanup_error:
                print(f"⚠️  Error during cleanup: {cleanup_error}")
    
    async def stop_streaming(self) -> None:
        """Stop streaming (keeps client for fast reconnect)."""
        await self.stop_safe()
    
    async def cleanup(self) -> None:
        """Full cleanup including client."""
        await self._cleanup_stream(full_cleanup=True)
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,  # Simulated via chunking
            'batch': True,
            'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ja', 'zh', 'ko'],  # Whisper supports many
            'audio_formats': ['pcm16', 'wav'],
            'sample_rates': [16000],
            'features': ['chunked_streaming', 'silence_detection', 'multi_language']
        }
