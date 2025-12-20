"""
Local Text-to-Speech provider using pyttsx3.
Fast, offline TTS with interruption support.
"""

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import wave
import pyaudio

import subprocess
import platform

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("⚠️  pyttsx3 not installed - will use macOS 'say' command as fallback")

try:
    # Try relative imports first (when used as package)
    from ...interfaces.text_to_speech import TextToSpeechInterface
    from ...models.data_models import AudioOutput, AudioFormat
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.text_to_speech import TextToSpeechInterface
    from models.data_models import AudioOutput, AudioFormat


class LocalTTSProvider(TextToSpeechInterface):
    """
    Local TTS implementation using pyttsx3 and PyAudio.
    
    Features:
    - Zero latency (no API calls)
    - Offline operation
    - Instant interruption support
    - Cross-platform (macOS, Windows, Linux)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Local TTS provider.
        
        Args:
            config: Configuration dictionary containing:
                - voice_id: System voice index (default: 0)
                - rate: Words per minute (default: 175)
                - volume: Volume 0.0-1.0 (default: 0.9)
        """
        # Prefer macOS 'say' command on macOS (more stable than pyttsx3)
        # pyttsx3 causes segfaults due to audio device conflicts
        self.use_macos_say = (platform.system() == 'Darwin')
        if not self.use_macos_say and pyttsx3 is None:
            raise ImportError("pyttsx3 not installed and not on macOS. Run: pip install pyttsx3")
        
        self.voice_id = config.get('voice_id', 0)
        self.rate = config.get('rate', 175)
        self.volume = config.get('volume', 0.9)
        
        # TTS engine (runs in separate dedicated thread)
        self._engine = None
        self._engine_lock = threading.Lock()
        self._engine_thread = None
        self._engine_ready = threading.Event()
        
        # Playback control
        self._is_playing = False
        self._stop_playback = threading.Event()
        self._speech_finished = threading.Event()  # Signals when pyttsx3 finishes
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_stream = None
        self._pyaudio = None
    
    async def initialize(self) -> bool:
        """Initialize the Local TTS provider."""
        try:
            if self.use_macos_say:
                print("✅ Local TTS provider initialized (using macOS 'say' command)")
                return True
            
            # DON'T initialize pyttsx3 here - defer until actual playback
            # This avoids audio device conflicts during initialization
            # The engine will be created fresh in _play_direct when needed
            
            print("✅ Local TTS provider initialized (engine will be created on first use)")
            return True
            
        except Exception as e:
            print(f"Failed to initialize Local TTS provider: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _ensure_pyaudio(self):
        """Ensure PyAudio is initialized (lazy initialization)."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
    
    def _init_engine(self):
        """Initialize pyttsx3 engine (must run in thread)."""
        with self._engine_lock:
            self._engine = pyttsx3.init()
            
            # Set voice with validation
            voices = self._engine.getProperty('voices')
            if voices:
                if 0 <= self.voice_id < len(voices):
                    try:
                        self._engine.setProperty('voice', voices[self.voice_id].id)
                        print(f"🎤 Using voice: {voices[self.voice_id].name}")
                    except Exception as e:
                        print(f"⚠️  Failed to set voice {self.voice_id}: {e}")
                        print(f"   Using default voice instead")
                else:
                    print(f"⚠️  Invalid voice_id {self.voice_id} (available: 0-{len(voices)-1})")
                    print(f"   Using default voice (id: 0)")
                    # Use first voice as fallback
                    if len(voices) > 0:
                        self._engine.setProperty('voice', voices[0].id)
            
            # Set rate and volume
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            
            # Register callback to detect when speech finishes
            self._engine.connect('finished-utterance', self._on_speech_finished)
    
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """
        Synthesize speech from text.
        
        Note: For local TTS, we store the text and synthesize on-the-fly during playback.
        The engine is created fresh when needed (not during initialization).
        """
        # No need to check for engine - we create it on-demand in _play_direct
        # This allows deferred initialization to avoid audio device conflicts
        
        # Override rate if speed provided
        rate = self.rate
        if speed is not None:
            # Convert speed multiplier to words per minute
            rate = int(self.rate * speed)
        
        # Store text and metadata for playback
        return AudioOutput(
            audio_data=text.encode('utf-8'),  # Store text, not audio
            format=AudioFormat.WAV,
            sample_rate=22050,  # pyttsx3 default
            voice=f"system_voice_{self.voice_id}",
            language="en-US",
            metadata={
                'rate': rate,
                'volume': self.volume,
                'engine': 'macos_say' if self.use_macos_say else 'pyttsx3',
                'text': text,  # Store original text
                'use_direct_playback': True
            }
        )
    
    def _synthesize_to_bytes(self, text: str, rate: int) -> bytes:
        """Synthesize text to WAV bytes (runs in thread)."""
        with self._engine_lock:
            # Set rate for this synthesis
            self._engine.setProperty('rate', rate)
            
            # Create temporary file with explicit name (not auto-delete)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            
            try:
                # Close the file descriptor, pyttsx3 will write to the path
                os.close(tmp_fd)
                
                # Save to file
                self._engine.save_to_file(text, tmp_path)
                self._engine.runAndWait()
                
                # Wait a bit for file to be fully written
                import time
                time.sleep(0.1)
                
                # Verify file exists and has content
                if not Path(tmp_path).exists():
                    raise RuntimeError(f"TTS file not created: {tmp_path}")
                
                file_size = Path(tmp_path).stat().st_size
                if file_size == 0:
                    raise RuntimeError(f"TTS file is empty: {tmp_path}")
                
                # Read file content
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                if len(audio_data) < 44:  # WAV header is at least 44 bytes
                    raise RuntimeError(f"TTS audio data too small: {len(audio_data)} bytes")
                
                return audio_data
                
            finally:
                # Cleanup temp file
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
    
    def play_audio(self, audio: AudioOutput) -> None:
        """Play synthesized audio (synchronous version)."""
        asyncio.run(self.play_audio_async(audio))
    
    async def play_audio_async(self, audio: AudioOutput) -> None:
        """Play synthesized audio with interruption support."""
        if self._is_playing:
            # Stop any existing playback
            self.stop_audio()
            await asyncio.sleep(0.1)
        
        self._is_playing = True
        self._stop_playback.clear()
        self._speech_finished.clear()  # Reset completion flag
        
        try:
            # Check if we should use direct playback (for local TTS)
            use_direct = audio.metadata.get('use_direct_playback', False)
            
            if use_direct:
                # Use pyttsx3 direct playback (better quality, simpler)
                text = audio.metadata.get('text', audio.audio_data.decode('utf-8'))
                rate = audio.metadata.get('rate', self.rate)
                
                # Start playback in thread (doesn't block the async loop)
                # The thread will poll for callback and set _speech_finished when done
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._play_direct, text, rate)
                
                # Poll for completion in async context
                # This allows interruption and proper async flow
                max_wait = 30.0  # Maximum 30 seconds timeout
                poll_interval = 0.05  # Check every 50ms
                elapsed = 0.0
                
                print("🎤 Waiting for audio to complete...")
                while elapsed < max_wait:
                    if self._speech_finished.is_set():
                        print("🎤 Speech completed (detected via callback)")
                        break
                    if self._stop_playback.is_set():
                        print("🎤 Speech interrupted")
                        break
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                else:
                    # Timeout - this shouldn't happen with our 2s callback timeout
                    print("⚠️  Audio completion timeout (30s) - may be stuck")
                
                # Small additional buffer for audio device cleanup
                if not self._stop_playback.is_set():
                    await asyncio.sleep(0.1)
            else:
                # Play pre-generated audio file
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._play_audio_thread, audio.audio_data)
                
                # Wait for buffer flush
                if not self._stop_playback.is_set():
                    await asyncio.sleep(0.3)
        
        except Exception as e:
            print(f"Error playing audio: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self._is_playing = False
    
    def _on_speech_finished(self, name, completed):
        """
        Callback when pyttsx3 finishes speaking.
        NOTE: This fires when synthesis completes, NOT when audio finishes!
        Do NOT set _speech_finished here - let _play_direct handle it after buffer wait.
        """
        print(f"🔔 pyttsx3 callback: synthesis finished (completed={completed})")
        # Don't set _speech_finished here! Audio buffer still playing.
    
    def _play_direct(self, text: str, rate: int):
        """
        Play audio directly using pyttsx3 or macOS 'say' command.
        
        This runs in a separate thread and blocks until complete.
        The _speech_finished event is set when audio actually finishes.
        """
        word_count = len(text.split())
        estimated_time = (word_count / 175.0) * 60.0  # ~175 WPM
        print(f"🔊 [_play_direct] Starting playback: {word_count} words (~{estimated_time:.1f}s)")
        print(f"   Text: '{text[:60]}...'")
        
        # Use macOS say command if pyttsx3 not available
        if self.use_macos_say:
            print(f"🔊 [_play_direct] Using macOS 'say' command")
            try:
                # subprocess.run blocks until 'say' completes (includes audio playback)
                subprocess.run(['say', '-r', str(rate), text], check=True)
                print(f"🔊 [_play_direct] macOS say completed (audio finished)")
                self._speech_finished.set()
                return
            except Exception as e:
                print(f"❌ macOS say failed: {e}")
                self._speech_finished.set()
                return
        
        # Create fresh engine in this thread for macOS compatibility
        # macOS NSSpeechSynthesizer doesn't work across threads
        print(f"🔊 [_play_direct] Creating fresh engine in thread {threading.current_thread().name}")
        engine = pyttsx3.init()
        
        # Set voice
        voices = engine.getProperty('voices')
        if 0 <= self.voice_id < len(voices):
            engine.setProperty('voice', voices[self.voice_id].id)
            print(f"🔊 [_play_direct] Using voice: {voices[self.voice_id].name}")
        
        # Set rate and volume
        engine.setProperty('rate', rate)
        engine.setProperty('volume', self.volume)
        print(f"🔊 [_play_direct] Rate set to: {rate}")
        
        # Reset the finished flag before speaking
        self._speech_finished.clear()
        
        # Register callback for this engine instance
        # Note: callback may not fire reliably on all platforms
        engine.connect('finished-utterance', self._on_speech_finished)
        
        # Speak the text directly
        engine.say(text)
        print(f"🔊 [_play_direct] Called engine.say()")
        
        # Run and wait for speech to complete
        # This blocks until pyttsx3 finishes synthesis and playback
        print(f"🔊 [_play_direct] Calling engine.runAndWait()...")
        engine.runAndWait()
        print(f"🔊 [_play_direct] engine.runAndWait() completed")
        
        # CRITICAL: runAndWait() returns when synthesis completes,
        # but audio buffer may still be playing. We must wait for the buffer.
        
        import time
        
        # Wait for audio buffer to finish playing
        # Scale based on text length for accurate timing
        word_count = len(text.split())
        # At 175 WPM: 60s/175words = 0.343s per word
        # Add buffer overhead: 500ms base + actual speech time + 200ms safety
        estimated_speech_time = (word_count / 175.0) * 60.0
        buffer_time = 0.5 + estimated_speech_time + 0.2
        
        print(f"🔊 [_play_direct] Waiting {buffer_time:.2f}s for audio buffer ({word_count} words, ~{estimated_speech_time:.1f}s speech)")
        time.sleep(buffer_time)
        print(f"🔊 [_play_direct] Audio playback complete")
        
        # Cleanup engine
        del engine
        
        # NOW signal completion (after audio truly finished)
        self._speech_finished.set()
    
    def _play_audio_thread(self, audio_data: bytes):
        """Play audio in thread with chunk-based interruption support."""
        # Ensure PyAudio is initialized (lazy init to avoid conflicts)
        self._ensure_pyaudio()
        
        # Write audio data to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            # Open WAV file
            with wave.open(tmp_path, 'rb') as wf:
                # Open audio stream
                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                try:
                    # Play in chunks for responsive interruption
                    chunk_size = 1024
                    data = wf.readframes(chunk_size)
                    
                    while data and not self._stop_playback.is_set():
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                    
                finally:
                    stream.stop_stream()
                    stream.close()
                    
        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    def stop_audio(self) -> None:
        """Stop audio playback immediately."""
        self._stop_playback.set()
        self._speech_finished.set()  # Signal completion to unblock waits
        
        # Kill macOS 'say' command if running
        if self.use_macos_say:
            try:
                import signal
                # Kill all 'say' processes started by this user
                subprocess.run(['pkill', '-9', 'say'], capture_output=True)
                print("🛑 Stopped macOS 'say' command")
            except Exception as e:
                print(f"⚠️  Error stopping 'say': {e}")
        
        # Try to stop pyttsx3 engine
        with self._engine_lock:
            try:
                if self._engine:
                    self._engine.stop()
            except Exception:
                pass
        
        # Wait for playback thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
        
        self._is_playing = False
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    async def save_audio(self, audio: AudioOutput, path: Path) -> None:
        """Save audio to file."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio data
            await asyncio.get_event_loop().run_in_executor(
                None,
                path.write_bytes,
                audio.audio_data
            )
            
        except Exception as e:
            raise Exception(f"Failed to save audio: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop any playing audio
        if self._is_playing:
            self.stop_audio()
        
        # Cleanup engine
        with self._engine_lock:
            if self._engine:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = None
        
        # Cleanup PyAudio
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
    
    async def speak_streaming(self, text_generator, on_sentence_start=None, on_complete=None) -> bool:
        """
        EXPERIMENTAL: Stream TTS - start speaking as text arrives.
        
        Buffers text until sentence boundaries, then speaks each sentence
        while the next one is being generated.
        
        Args:
            text_generator: Async generator yielding text chunks
            on_sentence_start: Optional callback when a sentence starts speaking
            on_complete: Optional callback when all speech finishes
            
        Returns:
            True if completed normally, False if interrupted
        """
        import re
        import asyncio
        
        self._is_playing = True
        self._stop_playback.clear()
        
        buffer = ""
        sentence_queue = asyncio.Queue()
        sentences_complete = asyncio.Event()
        was_interrupted = False
        
        # Sentence boundary pattern
        sentence_end = re.compile(r'[.!?]+\s*')
        
        async def sentence_producer():
            """Collect text and extract sentences."""
            nonlocal buffer
            chunk_count = 0
            try:
                print("[TTS DEBUG] sentence_producer starting...")
                async for chunk in text_generator:
                    chunk_count += 1
                    print(f"[TTS DEBUG] Received chunk {chunk_count}: {len(chunk)} chars")
                    
                    if self._stop_playback.is_set():
                        print("[TTS DEBUG] Stop playback set, breaking")
                        break
                    
                    buffer += chunk
                    print(f"[TTS DEBUG] Buffer now: {len(buffer)} chars")
                    
                    # Extract complete sentences
                    while True:
                        match = sentence_end.search(buffer)
                        if not match:
                            break
                        
                        # Extract sentence up to and including punctuation
                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        buffer = buffer[end_pos:]
                        
                        if sentence:
                            print(f"[TTS DEBUG] Extracted sentence: {sentence[:50]}...")
                            await sentence_queue.put(sentence)
                
                # Handle remaining buffer (incomplete sentence)
                if buffer.strip() and not self._stop_playback.is_set():
                    print(f"[TTS DEBUG] Final buffer: {buffer[:50]}...")
                    await sentence_queue.put(buffer.strip())
                
                print(f"[TTS DEBUG] sentence_producer done. Received {chunk_count} chunks")
                    
            finally:
                # Signal no more sentences
                await sentence_queue.put(None)
        
        async def sentence_speaker():
            """Speak sentences as they arrive."""
            nonlocal was_interrupted
            sentence_count = 0
            timeout_count = 0
            
            print("[TTS DEBUG] sentence_speaker starting...")
            
            while True:
                if self._stop_playback.is_set():
                    was_interrupted = True
                    print("[TTS DEBUG] Stop playback set in speaker")
                    break
                
                try:
                    # Wait for next sentence with timeout
                    sentence = await asyncio.wait_for(
                        sentence_queue.get(), 
                        timeout=0.1
                    )
                    timeout_count = 0  # Reset on successful get
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count % 50 == 0:  # Log every 5 seconds
                        print(f"[TTS DEBUG] Still waiting for sentences... ({timeout_count} timeouts)")
                    continue
                
                if sentence is None:
                    # No more sentences
                    print(f"[TTS DEBUG] Received None, ending speaker. Spoke {sentence_count} sentences")
                    break
                
                sentence_count += 1
                print(f"[TTS DEBUG] Speaking sentence {sentence_count}: {sentence[:40]}...")
                if on_sentence_start:
                    on_sentence_start(sentence_count, sentence)
                
                # Speak this sentence (blocking call in executor)
                if self._stop_playback.is_set():
                    was_interrupted = True
                    break
                
                print(f"🔊 [{sentence_count}] Speaking: {sentence[:40]}...")
                
                # Use macOS say command (non-blocking via subprocess)
                if self.use_macos_say:
                    proc = await asyncio.create_subprocess_exec(
                        'say', '-r', str(self.rate), sentence,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    
                    # Wait for speech to complete, checking for interruption every 25ms
                    while proc.returncode is None:
                        # Check for stop FIRST before any waiting
                        if self._stop_playback.is_set():
                            print(f"[TTS DEBUG] Stop detected mid-sentence {sentence_count}, killing process...")
                            try:
                                proc.kill()
                                await proc.wait()  # Ensure process terminates
                            except Exception as e:
                                print(f"[TTS DEBUG] Kill error: {e}")
                            was_interrupted = True
                            print(f"[TTS DEBUG] Process killed, breaking out of sentence loop")
                            break
                        
                        # Short poll interval for responsive interruption
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=0.025)
                            # Process finished naturally
                            break
                        except asyncio.TimeoutError:
                            # Still running, loop again to check stop flag
                            pass
                    
                    # Break out of outer sentence loop if interrupted
                    if was_interrupted:
                        print(f"[TTS DEBUG] Breaking out of sentence loop after interruption")
                        break
                else:
                    # Fallback: use synchronous playback in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, 
                        self._play_direct, 
                        sentence, 
                        self.rate
                    )
                    
                    # Also check for interruption after fallback playback
                    if self._stop_playback.is_set():
                        was_interrupted = True
                        break
            
            sentences_complete.set()
            if on_complete:
                on_complete(was_interrupted)
        
        try:
            # Run producer and speaker concurrently
            producer_task = asyncio.create_task(sentence_producer())
            speaker_task = asyncio.create_task(sentence_speaker())
            
            await asyncio.gather(producer_task, speaker_task)
            
            return not was_interrupted
            
        except Exception as e:
            print(f"❌ Streaming TTS error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self._is_playing = False
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        voices = []
        if self._engine:
            with self._engine_lock:
                try:
                    voices = [
                        f"system_voice_{i}" 
                        for i in range(len(self._engine.getProperty('voices')))
                    ]
                except Exception:
                    pass
        
        return {
            'streaming': True,  # Now supports streaming!
            'batch': True,
            'voices': voices,
            'languages': ['en-US'],  # Depends on system voices
            'audio_formats': ['wav'],
            'speed_range': (0.5, 2.0),
            'pitch_range': (0, 0),  # No pitch control with pyttsx3
            'features': ['offline', 'fast', 'interruptible', 'streaming'],
            'latency': 'low',
            'requires_api': False
        }

