"""
Barge-in detection for voice interruption during TTS playback.

This allows users to interrupt the assistant's speech by speaking,
which triggers immediate transition to transcription mode.

FEATURE: Audio buffering - captures speech that triggered barge-in
so it can be fed to transcription (user doesn't need to repeat).
"""

import asyncio
import threading
from collections import deque
from typing import Optional, Callable, List

import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from enum import Enum


class BargeInMode(Enum):
    """Barge-in detection modes."""
    ENERGY = "energy"          # Simple energy threshold (fast, may false positive)
    WAKE_WORD = "wake_word"    # Requires wake word (more accurate, but slower)
    DISABLED = "disabled"      # No barge-in


@dataclass
class BargeInConfig:
    """Configuration for barge-in detection."""
    mode: BargeInMode = BargeInMode.ENERGY
    energy_threshold: float = 0.02  # RMS threshold for energy mode (0-1 scale)
    min_speech_duration: float = 0.15  # Minimum seconds of speech to trigger
    cooldown_after_tts_start: float = 0.5  # Don't detect for first N seconds (avoid TTS feedback)
    sample_rate: int = 16000
    chunk_size: int = 1600  # 100ms at 16kHz
    # Audio buffer settings
    buffer_seconds: float = 2.0  # Keep last N seconds of audio for prefill
    capture_after_trigger: float = 0.3  # Continue capturing for N seconds after trigger


class BargeInDetector:
    """
    Detects user speech during TTS playback to enable interruption.
    
    Uses callback-based audio capture (same safe approach as transcription)
    to monitor for voice activity without blocking threads.
    
    FEATURE: Captures audio that triggered barge-in so it can be fed
    to transcription - user doesn't need to repeat themselves!
    """
    
    def __init__(self, config: Optional[BargeInConfig] = None):
        self.config = config or BargeInConfig()
        
        # State
        self._is_listening = False
        self._detected = asyncio.Event()
        self._shutdown = threading.Event()
        
        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        
        # Detection state
        self._speech_frames = 0
        self._required_frames = int(
            self.config.min_speech_duration * self.config.sample_rate / self.config.chunk_size
        )
        self._cooldown_frames = int(
            self.config.cooldown_after_tts_start * self.config.sample_rate / self.config.chunk_size
        )
        self._frame_count = 0
        
        # Audio buffer for capturing speech (circular buffer)
        # Keep last N seconds of audio so we can feed it to transcription
        self._buffer_max_chunks = int(
            self.config.buffer_seconds * self.config.sample_rate / self.config.chunk_size
        )
        self._audio_buffer: deque = deque(maxlen=self._buffer_max_chunks)
        self._captured_audio: Optional[np.ndarray] = None  # Final captured audio after trigger
        self._capture_after_frames = int(
            self.config.capture_after_trigger * self.config.sample_rate / self.config.chunk_size
        )
        self._post_trigger_frames = 0  # Count frames after trigger
        self._trigger_time: Optional[float] = None
        self._buffer_lock = threading.Lock()
        
        # Event loop reference for thread-safe signaling
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Callback for external notification
        self._on_barge_in: Optional[Callable[[], None]] = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """
        Audio callback for barge-in detection.
        
        Runs in sounddevice's audio thread - keeps processing fast and non-blocking.
        Also buffers audio for prefill to transcription.
        """
        if self._shutdown.is_set():
            return
        
        self._frame_count += 1
        
        # Always buffer audio (for prefill to transcription)
        audio_copy = indata.copy().flatten()
        with self._buffer_lock:
            self._audio_buffer.append(audio_copy)
        
        # If already triggered, continue capturing for a bit then stop
        if self._detected.is_set():
            self._post_trigger_frames += 1
            if self._post_trigger_frames >= self._capture_after_frames:
                # Done capturing post-trigger audio
                self._finalize_captured_audio()
                self._is_listening = False
            return
        
        # Skip cooldown period (avoid detecting TTS playback feedback)
        if self._frame_count < self._cooldown_frames:
            return
        
        if not self._is_listening:
            return
        
        try:
            # Calculate RMS energy
            audio = audio_copy.astype(np.float32) / 32768.0  # Normalize int16 to float
            rms = np.sqrt(np.mean(audio ** 2))
            
            # Check if above threshold
            if rms > self.config.energy_threshold:
                self._speech_frames += 1
                
                # Check if enough consecutive speech frames
                if self._speech_frames >= self._required_frames:
                    import time
                    self._trigger_time = time.time()
                    print(f"🎤 Barge-in detected! (RMS: {rms:.4f}, threshold: {self.config.energy_threshold})")
                    print(f"📼 Buffered {len(self._audio_buffer)} audio chunks for transcription prefill")
                    
                    # Mark as detected but keep listening for a bit more audio
                    self._detected.set()
                    self._post_trigger_frames = 0
                    
                    # Signal detection via event loop
                    if self._event_loop:
                        try:
                            self._event_loop.call_soon_threadsafe(self._signal_detection)
                        except RuntimeError:
                            pass  # Event loop closed
            else:
                # Reset consecutive count if silence
                self._speech_frames = max(0, self._speech_frames - 1)
                
        except Exception as e:
            if not self._shutdown.is_set():
                print(f"⚠️  Barge-in callback error: {e}")
    
    def _finalize_captured_audio(self):
        """Finalize the captured audio buffer into a single array."""
        with self._buffer_lock:
            if self._audio_buffer:
                # Concatenate all buffered chunks into single array
                self._captured_audio = np.concatenate(list(self._audio_buffer))
                duration = len(self._captured_audio) / self.config.sample_rate
                print(f"📼 Captured {duration:.2f}s of audio for transcription prefill")
    
    def _signal_detection(self):
        """Signal barge-in detection (called from event loop thread)."""
        self._detected.set()
        if self._on_barge_in:
            try:
                self._on_barge_in()
            except Exception as e:
                print(f"⚠️  Barge-in callback error: {e}")
    
    async def start(self, on_barge_in: Optional[Callable[[], None]] = None) -> None:
        """
        Start listening for barge-in.
        
        Args:
            on_barge_in: Optional callback when barge-in is detected
        """
        if self._is_listening:
            return
        
        if self.config.mode == BargeInMode.DISABLED:
            return
        
        # Store callback and event loop
        self._on_barge_in = on_barge_in
        self._event_loop = asyncio.get_running_loop()
        
        # Reset state
        self._detected.clear()
        self._shutdown.clear()
        self._speech_frames = 0
        self._frame_count = 0
        self._post_trigger_frames = 0
        self._trigger_time = None
        self._captured_audio = None
        with self._buffer_lock:
            self._audio_buffer.clear()
        
        # Open audio stream with callback
        print("👂 Starting barge-in detection (with audio buffering)...")
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.config.chunk_size,
                callback=self._audio_callback
            )
            self._stream.start()
            self._is_listening = True
            print(f"✅ Barge-in detection active (threshold: {self.config.energy_threshold}, buffer: {self.config.buffer_seconds}s)")
        except Exception as e:
            print(f"❌ Failed to start barge-in detection: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop listening for barge-in."""
        if not self._is_listening and self._stream is None:
            return
        
        print("🛑 Stopping barge-in detection...")
        self._shutdown.set()
        self._is_listening = False
        
        # Finalize captured audio if not done yet
        if self._detected.is_set() and self._captured_audio is None:
            self._finalize_captured_audio()
        
        if self._stream:
            try:
                self._stream.stop()
                await asyncio.sleep(0.05)  # Brief wait for callback to finish
                self._stream.close()
                await asyncio.sleep(0.1)  # Wait for device release
                print("✅ Barge-in detection stopped")
            except Exception as e:
                print(f"⚠️  Barge-in stop error: {e}")
            finally:
                self._stream = None
        
        self._event_loop = None
        self._on_barge_in = None
    
    def get_captured_audio(self) -> Optional[np.ndarray]:
        """
        Get the audio that triggered barge-in (for transcription prefill).
        
        Returns:
            numpy array of int16 audio samples, or None if no audio captured
        """
        return self._captured_audio
    
    def get_captured_audio_bytes(self) -> Optional[bytes]:
        """
        Get the captured audio as bytes (for sending to transcription API).
        
        Returns:
            bytes of int16 PCM audio, or None if no audio captured
        """
        if self._captured_audio is not None:
            return self._captured_audio.astype(np.int16).tobytes()
        return None
    
    async def wait_for_barge_in(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for barge-in detection.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            True if barge-in detected, False if timeout or stopped
        """
        try:
            await asyncio.wait_for(self._detected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening for barge-in."""
        return self._is_listening
    
    @property
    def detected(self) -> bool:
        """Check if barge-in was detected."""
        return self._detected.is_set()


# Convenience function
def create_barge_in_detector(
    energy_threshold: float = 0.02,
    min_speech_duration: float = 0.15,
    cooldown: float = 0.5
) -> BargeInDetector:
    """Create a barge-in detector with custom settings."""
    config = BargeInConfig(
        mode=BargeInMode.ENERGY,
        energy_threshold=energy_threshold,
        min_speech_duration=min_speech_duration,
        cooldown_after_tts_start=cooldown
    )
    return BargeInDetector(config)

