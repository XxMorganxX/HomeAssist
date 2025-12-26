"""
Fully async AssemblyAI transcription provider.
Eliminates threading complexity from original implementation.

CRITICAL: Uses callback-based audio capture to avoid segfaults from
executor threads blocking in native sounddevice C code during cleanup.
"""

import asyncio
import json
import threading
from typing import AsyncIterator, Dict, Any, Optional
from urllib.parse import urlencode
from datetime import datetime

import aiohttp
import numpy as np
import sounddevice as sd

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


class AssemblyAIAsyncProvider(StreamingProviderBase, TranscriptionInterface):
    """
    Fully async AssemblyAI implementation.
    
    ARCHITECTURE (segfault-safe):
    - Uses CALLBACK-BASED audio capture instead of blocking reads
    - Callbacks run in sounddevice's audio thread (not executor threads)
    - No blocking calls in executors = no zombie threads during cleanup
    - Stream can be stopped/closed safely without crashing
    
    Key improvements:
    - No threading - callback feeds async queue
    - Natural async flow with aiohttp WebSocket  
    - Safe cleanup - no native code blocking during shutdown
    - Better error handling and cancellation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._component_name = "transcription"
        
        # Configuration
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required")
        
        self.sample_rate = config.get('sample_rate', 16000)
        self.format_turns = config.get('format_turns', True)
        # frames_per_buffer = blocksize for sd.InputStream
        # Default 1024 = 64ms at 16kHz (good for Bluetooth)
        self.frames_per_buffer = config.get('frames_per_buffer', 1024)
        self.channels = 1
        self._audio_queue: Optional[asyncio.Queue] = None  # Queue for decoupling audio callback from WebSocket send
        
        # Device-specific settings from config
        self._device_index = config.get('device_index')
        self._latency = config.get('latency', 'high')  # 'high' handles Bluetooth bursts better
        self._is_bluetooth = config.get('is_bluetooth', False)
        
        # WebSocket URL
        connection_params = {
            "sample_rate": self.sample_rate,
            "format_turns": self.format_turns,
        }
        self.api_endpoint = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(connection_params)}"
        
        # State
        self.audio_manager = get_audio_manager()
        self._audio_stream: Optional[sd.InputStream] = None
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None  # Persistent session
        self._send_task = None  # Only need send task now (no read task - using callback)
        self.session_id = None
        
        # Thread-safe shutdown coordination
        self._shutdown_flag = threading.Event()
        self._callback_active = threading.Event()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Prefill audio support (for barge-in captured audio)
        self._prefill_audio: Optional[bytes] = None
        
        # Pre-connection support (warm connection before needed)
        self._preconnect_task: Optional[asyncio.Task] = None
        self._ws_ready = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        self._preconnected = False
    
    async def initialize(self) -> bool:
        """One-time initialization - creates persistent HTTP session."""
        # Create persistent aiohttp session (reused across connections)
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            print("✅ Persistent HTTP session created for transcription")
        
        # Initialize the event for pre-connection signaling
        self._ws_ready = asyncio.Event()
        return True
    
    def set_prefill_audio(self, audio_bytes: Optional[bytes]) -> None:
        """
        Set audio to send before starting live capture.
        
        This is used for barge-in: the audio that triggered the interrupt
        is captured and fed to transcription so user doesn't repeat themselves.
        
        Args:
            audio_bytes: PCM int16 audio bytes, or None to clear
        """
        self._prefill_audio = audio_bytes
        if audio_bytes:
            duration = len(audio_bytes) / 2 / self.sample_rate  # 2 bytes per int16 sample
            print(f"📼 Prefill audio set: {duration:.2f}s")
    
    async def preconnect(self) -> None:
        """
        Pre-establish WebSocket connection before transcription is needed.
        
        Call this during TTS playback so the connection is ready
        when the user starts speaking (barge-in or after TTS).
        
        This is non-blocking - starts connection in background.
        """
        # Skip if already connected or connecting
        if self._ws and not self._ws.closed:
            print("⚡ WebSocket already connected")
            return
        
        if self._preconnect_task and not self._preconnect_task.done():
            print("⚡ Pre-connection already in progress")
            return
        
        # Start pre-connection in background
        self._preconnect_task = asyncio.create_task(self._do_preconnect())
    
    async def _do_preconnect(self) -> None:
        """Background task to establish WebSocket connection."""
        try:
            print("🔌 Pre-connecting to AssemblyAI...")
            
            # Ensure we have a session
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Close any existing dead connection
            if self._ws and self._ws.closed:
                self._ws = None
            
            # Connect WebSocket
            if not self._ws:
                self._ws = await self._session.ws_connect(
                    self.api_endpoint,
                    headers={"Authorization": self.api_key},
                    heartbeat=30
                )
                self._preconnected = True
                if self._ws_ready:
                    self._ws_ready.set()
                print("⚡ Pre-connected to AssemblyAI (ready for instant transcription)")
                
        except Exception as e:
            print(f"⚠️  Pre-connection failed (will retry on start): {e}")
            self._preconnected = False
            if self._ws_ready:
                self._ws_ready.clear()
    
    @property
    def is_preconnected(self) -> bool:
        """Check if WebSocket is pre-connected and ready."""
        return self._preconnected and self._ws and not self._ws.closed
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """
        Audio callback - runs in sounddevice's audio thread.
        
        This is MUCH safer than blocking reads in executor threads because:
        1. The callback is managed by sounddevice/PortAudio
        2. When stream.stop() is called, callbacks stop cleanly
        3. No zombie threads blocking in native code
        
        CRITICAL: This runs in a different thread, so we use thread-safe
        mechanisms to communicate with the async event loop.
        """
        if self._shutdown_flag.is_set():
            return  # Don't process if shutting down
        
        if status:
            if status.input_overflow:
                # Audio arriving faster than we can process - usually sample rate mismatch
                print(f"⚠️  Audio OVERFLOW - try increasing frames_per_buffer in config or check device sample rate")
            else:
                print(f"⚠️  Audio callback status: {status}")
        
        try:
            # Copy audio data (callback data is only valid during callback)
            audio_copy = indata.copy()
            
            # Put into queue using thread-safe method
            # Use call_soon_threadsafe to schedule queue.put_nowait from audio thread
            if self._event_loop and self._audio_queue:
                try:
                    self._event_loop.call_soon_threadsafe(
                        self._queue_audio_threadsafe, audio_copy
                    )
                except RuntimeError:
                    # Event loop closed or shutting down
                    pass
        except Exception as e:
            if not self._shutdown_flag.is_set():
                print(f"⚠️  Audio callback error: {e}")
    
    def _queue_audio_threadsafe(self, audio_data: np.ndarray):
        """Thread-safe queue put - called from event loop thread."""
        if self._audio_queue and not self._shutdown_flag.is_set():
            try:
                self._audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                # Queue full, drop oldest frame to prevent backpressure
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(audio_data)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
    
    async def _initialize_stream(self):
        """Initialize audio and WebSocket."""
        # Store event loop reference for thread-safe callback communication
        self._event_loop = asyncio.get_running_loop()
        
        # Clear shutdown flag
        self._shutdown_flag.clear()
        
        # Acquire audio
        acquired = self.audio_manager.acquire_audio("transcription", force_cleanup=True)
        if not acquired:
            raise RuntimeError("Failed to acquire audio for transcription")
        
        # Create audio queue BEFORE opening stream (callback may fire immediately)
        # Larger queue for Bluetooth devices which can have bursty delivery
        self._audio_queue = asyncio.Queue(maxsize=50)  # Buffer up to 50 chunks
        
        # Open audio stream with CALLBACK (not blocking reads!)
        bt_mode = " [Bluetooth]" if self._is_bluetooth else ""
        print(f"🎙️  Opening transcription audio stream (callback mode){bt_mode}...")
        print(f"   Device: {self._device_index or 'default'}, Rate: {self.sample_rate}Hz")
        print(f"   Blocksize: {self.frames_per_buffer} frames ({self.frames_per_buffer/self.sample_rate*1000:.0f}ms), Latency: '{self._latency}'")
        self._audio_stream = sd.InputStream(
            device=self._device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            blocksize=self.frames_per_buffer,
            latency=self._latency,
            callback=self._audio_callback  # CRITICAL: Use callback instead of blocking reads
        )
        self._audio_stream.start()
        self._callback_active.set()
        print("✅ Audio stream opened (callback mode - no blocking threads)")
        
        # Use pre-connected WebSocket if available, otherwise connect now
        if self.is_preconnected:
            print("⚡ Using pre-connected WebSocket (instant start!)")
        else:
            # Wait briefly for pre-connect task if it's in progress
            if self._preconnect_task and not self._preconnect_task.done():
                try:
                    await asyncio.wait_for(self._preconnect_task, timeout=0.5)
                except asyncio.TimeoutError:
                    print("⏳ Pre-connect still in progress, connecting fresh...")
            
            # Connect if still not connected
            if not self.is_preconnected:
                print("🌐 Connecting to AssemblyAI...")
                # Ensure we have a session
                if not self._session or self._session.closed:
                    self._session = aiohttp.ClientSession()
                try:
                    self._ws = await self._session.ws_connect(
                        self.api_endpoint,
                        headers={"Authorization": self.api_key},
                        heartbeat=30
                    )
                    print(f"✅ WebSocket connected")
                except Exception as e:
                    print(f"❌ WebSocket connection failed: {e}")
                    raise
            
            # Clear pre-connect state (connection is now in use)
        self._preconnected = False
        self._preconnect_task = None
        
        # Send prefill audio if available (barge-in captured audio)
        if self._prefill_audio:
            await self._send_prefill_audio()
        
        # Start send task (no read task needed - callback handles audio capture)
        self._send_task = asyncio.create_task(self._send_audio())
    
    async def _send_prefill_audio(self):
        """Send prefill audio to WebSocket before starting live capture."""
        if not self._prefill_audio or not self._ws:
            return
        
        prefill_bytes = self._prefill_audio
        self._prefill_audio = None  # Clear after sending
        
        duration = len(prefill_bytes) / 2 / self.sample_rate
        print(f"📼 Sending {duration:.2f}s of prefill audio to transcription...")
        
        try:
            # Send in chunks to avoid overwhelming the connection
            chunk_size = self.frames_per_buffer * 2  # bytes (2 bytes per int16 sample)
            offset = 0
            chunks_sent = 0
            
            while offset < len(prefill_bytes):
                chunk = prefill_bytes[offset:offset + chunk_size]
                if self._ws and not self._ws.closed:
                    await self._ws.send_bytes(chunk)
                    chunks_sent += 1
                    # Small delay between chunks to avoid buffering issues
                    if chunks_sent % 5 == 0:
                        await asyncio.sleep(0.01)
                offset += chunk_size
            
            print(f"✅ Prefill audio sent ({chunks_sent} chunks)")
            
        except Exception as e:
            print(f"⚠️  Error sending prefill audio: {e}")
    
    async def _cleanup_stream(self, full_cleanup: bool = False):
        """
        Cleanup resources - safe to call multiple times.
        
        CRITICAL: Uses callback-based audio, so cleanup is much safer:
        1. Set shutdown flag (callbacks will stop processing)
        2. Stop stream (callbacks will stop firing)
        3. Close stream (safe because no threads blocking in native code)
        
        Args:
            full_cleanup: If True, also closes persistent session. If False, keeps
                         session alive for faster reconnection.
        """
        
        # Skip if nothing to cleanup (idempotent)
        if (self._send_task is None and
            self._audio_stream is None and 
            self._ws is None and 
            (full_cleanup or self._session is None)):
            return
        
        print("🧹 Cleaning up transcription stream...")
        
        # 1. Signal shutdown FIRST - this tells callbacks to stop processing
        self._shutdown_flag.set()
        
        # 2. Cancel any pre-connect task
        if self._preconnect_task and not self._preconnect_task.done():
            self._preconnect_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._preconnect_task, return_exceptions=True),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                pass
        self._preconnect_task = None
        self._preconnected = False
        
        # 3. Stop audio stream - callbacks will stop firing
        if self._audio_stream:
            try:
                if hasattr(self._audio_stream, 'active') and self._audio_stream.active:
                    self._audio_stream.stop()
                    print("✅ Audio stream stopped (callbacks ceased)")
                    
                # Brief wait for any in-flight callbacks to complete
                # Callbacks are fast (no blocking), so this is quick
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"⚠️  Audio stream stop error: {e}")
        
        self._callback_active.clear()
        
        # 4. Cancel send task (only task now - no reader task with callbacks)
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._send_task, return_exceptions=True),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                print("⚠️  Send task cancellation timed out")
        
        self._send_task = None
        
        # 5. Close audio stream (safe - no threads blocking in native code)
        if self._audio_stream:
            try:
                self._audio_stream.close()
                # Brief wait for audio device release
                await asyncio.sleep(0.1)
                print("✅ Audio stream closed")
            except Exception as e:
                print(f"⚠️  Audio stream close error: {e}")
            finally:
                self._audio_stream = None
        
        # 6. Close WebSocket gracefully (always close - need fresh for new session)
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({"type": "Terminate"})
                await asyncio.sleep(0.05)
                await self._ws.close()
                print("✅ WebSocket closed")
            except Exception as e:
                print(f"⚠️  WebSocket close error: {e}")
            self._ws = None
        
        # 7. Close aiohttp session only on full cleanup (keep for fast reconnect)
        if full_cleanup and self._session and not self._session.closed:
            try:
                await self._session.close()
                await asyncio.sleep(0.05)
                print("✅ HTTP session closed")
            except Exception as e:
                print(f"⚠️  Session close error: {e}")
            self._session = None
        
        # 8. Clear queue and event loop reference
        self._audio_queue = None
        self._event_loop = None
        
        # 9. Release audio manager
        self.audio_manager.release_audio("transcription", force_cleanup=False)
        
        self.session_id = None
        print("✅ Transcription cleanup complete")
    
    async def _send_audio(self):
        """Send audio from queue to WebSocket."""
        try:
            while not self._stop_event.is_set() and not self._shutdown_flag.is_set():
                try:
                    # Check if queue still exists
                    if not self._audio_queue:
                        break
                    
                    # Get from queue with timeout
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.3  # Shorter timeout for faster shutdown response
                    )
                    
                    # Send to WebSocket
                    if self._ws and not self._ws.closed and not self._shutdown_flag.is_set():
                        await self._ws.send_bytes(audio_data.tobytes())
                    else:
                        break
                        
                except asyncio.TimeoutError:
                    # No audio in queue, continue
                    continue
                except asyncio.CancelledError:
                    raise  # Propagate cancellation
                except Exception as e:
                    if not self._shutdown_flag.is_set():
                        print(f"⚠️  Audio send error: {e}")
                    break
                
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            if not self._shutdown_flag.is_set():
                print(f"❌ Audio sending error: {e}")
                import traceback
                traceback.print_exc()
    
    async def start_streaming(self) -> AsyncIterator[TranscriptionResult]:
        """Start streaming transcription."""
        await self.start_safe()
        
        try:
            # Receive messages from WebSocket
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get('type')
                        
                        if msg_type == "Begin":
                            self.session_id = data.get('id')
                            print(f"📝 Session started: {self.session_id}")
                        
                        elif msg_type == "Turn":
                            transcript = data.get('transcript', '')
                            is_formatted = data.get('turn_is_formatted', False)
                            
                            if transcript:
                                result = TranscriptionResult(
                                    text=transcript,
                                    is_final=is_formatted,
                                    timestamp=datetime.now().timestamp(),
                                    metadata={'session_id': self.session_id}
                                )
                                yield result
                        
                        elif msg_type == "Termination":
                            print(f"📝 Session terminated")
                            break
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error: {e}")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"❌ WebSocket error: {msg.data}")
                    break
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print(f"🔌 WebSocket closed")
                    break
        
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("🛑 Transcription interrupted")
            raise
        except Exception as e:
            print(f"❌ Transcription streaming error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Always cleanup, even on exception
            try:
                await self.stop_safe()
            except Exception as cleanup_error:
                print(f"⚠️  Error during final cleanup: {cleanup_error}")
    
    async def stop_streaming(self) -> None:
        """Stop streaming (keeps persistent session for fast reconnect)."""
        await self.stop_safe()
    
    async def cleanup(self) -> None:
        """Full cleanup including persistent session."""
        await self._cleanup_stream(full_cleanup=True)
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'batch': False,
            'languages': ['en-US'],
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'features': ['turn_formatting', 'real_time', 'fully_async']
        }

