"""
Shared Audio Bus for zero-latency audio component transitions.

This module provides a persistent audio input stream that multiple consumers
can subscribe to, eliminating device acquisition delays during state transitions.

Key benefits:
- No stream teardown between transcription/barge-in transitions
- Ring buffer provides instant prefill access for barge-in captured audio
- Thread-safe subscriber management with priority support
- Backward compatible - components can fall back to standalone streams

Architecture:
- Wake word and termination detection remain process-isolated (unchanged)
- Transcription and barge-in subscribe to the shared bus during conversations
- Bus starts after wake word detection, stops when conversation ends
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any

import numpy as np
import sounddevice as sd


@dataclass
class SharedAudioBusConfig:
    """Configuration for the shared audio bus."""
    enabled: bool = True
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024  # 64ms at 16kHz - good for Bluetooth
    buffer_seconds: float = 3.0  # Ring buffer size for prefill
    device_index: Optional[int] = None  # None = default device
    latency: str = 'high'  # 'high' for Bluetooth devices
    is_bluetooth: bool = False


@dataclass
class AudioSubscriber:
    """Represents a subscriber to the audio bus."""
    name: str
    callback: Callable[[np.ndarray, int, dict, sd.CallbackFlags], None]
    priority: int = 0  # Higher priority = called first
    active: bool = True


class RingBuffer:
    """
    Thread-safe ring buffer for audio data.
    
    Stores the last N seconds of audio for instant prefill access.
    """
    
    def __init__(self, max_seconds: float, sample_rate: int, chunk_size: int):
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        # Calculate max chunks to store
        self._max_chunks = int(max_seconds * sample_rate / chunk_size)
        self._buffer: deque = deque(maxlen=self._max_chunks)
        self._lock = threading.Lock()
    
    def append(self, audio_chunk: np.ndarray) -> None:
        """Append an audio chunk to the buffer (thread-safe)."""
        with self._lock:
            self._buffer.append(audio_chunk.copy())
    
    def get_last(self, seconds: float) -> Optional[np.ndarray]:
        """
        Get the last N seconds of audio as a contiguous array.
        
        Args:
            seconds: Number of seconds to retrieve
            
        Returns:
            numpy array of int16 audio samples, or None if buffer is empty
        """
        with self._lock:
            if not self._buffer:
                return None
            
            # Calculate how many chunks we need
            chunks_needed = int(seconds * self._sample_rate / self._chunk_size)
            chunks_to_get = min(chunks_needed, len(self._buffer))
            
            if chunks_to_get == 0:
                return None
            
            # Get the most recent chunks
            recent_chunks = list(self._buffer)[-chunks_to_get:]
            return np.concatenate(recent_chunks)
    
    def get_last_bytes(self, seconds: float) -> Optional[bytes]:
        """
        Get the last N seconds of audio as bytes.
        
        Args:
            seconds: Number of seconds to retrieve
            
        Returns:
            bytes of int16 PCM audio, or None if buffer is empty
        """
        audio = self.get_last(seconds)
        if audio is not None:
            return audio.astype(np.int16).tobytes()
        return None
    
    def clear(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self._lock:
            self._buffer.clear()
    
    @property
    def duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) * self._chunk_size / self._sample_rate


class SharedAudioBus:
    """
    Persistent audio input stream with subscriber pattern.
    
    Eliminates device acquisition delays by keeping a single InputStream
    alive during conversations. Components subscribe/unsubscribe to 
    receive audio data without stream teardown.
    
    Usage:
        bus = SharedAudioBus(config)
        await bus.start()
        
        # Instant subscription (no device setup)
        bus.subscribe("transcription", my_callback)
        
        # Instant unsubscription (no device teardown)
        bus.unsubscribe("transcription")
        
        # Get buffered audio for prefill
        audio = bus.get_buffer(seconds=2.0)
        
        await bus.stop()
    """
    
    def __init__(self, config: Optional[SharedAudioBusConfig] = None):
        self.config = config or SharedAudioBusConfig()
        
        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        
        # Subscribers
        self._subscribers: Dict[str, AudioSubscriber] = {}
        self._subscribers_lock = threading.Lock()
        self._sorted_subscribers: List[AudioSubscriber] = []  # Cached sorted list
        
        # Ring buffer for prefill
        self._ring_buffer = RingBuffer(
            max_seconds=self.config.buffer_seconds,
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size
        )
        
        # Shutdown coordination
        self._shutdown_flag = threading.Event()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """
        Audio callback - distributes audio to all subscribers.
        
        Runs in sounddevice's audio thread. Must be fast and non-blocking.
        """
        if self._shutdown_flag.is_set():
            return
        
        # Make a copy of the audio data (safe for all subscribers)
        try:
            audio_copy = indata.copy().flatten()
        except Exception:
            return  # Can't process invalid audio
        
        # Always buffer audio (for prefill access)
        self._ring_buffer.append(audio_copy)
        
        # Distribute to all active subscribers (in priority order)
        # Use cached sorted list to avoid sorting in hot path
        for subscriber in self._sorted_subscribers:
            if subscriber.active:
                try:
                    subscriber.callback(audio_copy, frames, time_info, status)
                except Exception as e:
                    # Don't let one subscriber's error affect others
                    print(f"⚠️  Audio bus subscriber '{subscriber.name}' error: {e}")
    
    def _update_sorted_subscribers(self) -> None:
        """Update the cached sorted subscriber list (call with lock held)."""
        self._sorted_subscribers = sorted(
            [s for s in self._subscribers.values() if s.active],
            key=lambda s: -s.priority  # Higher priority first
        )
    
    async def start(self) -> None:
        """
        Start the persistent input stream.
        
        Call this when entering conversation mode (after wake word detection).
        """
        if self._is_running:
            print("⚠️  Shared audio bus already running")
            return
        
        if not self.config.enabled:
            print("ℹ️  Shared audio bus is disabled")
            return
        
        self._event_loop = asyncio.get_running_loop()
        self._shutdown_flag.clear()
        self._ring_buffer.clear()
        
        bt_mode = " [Bluetooth]" if self.config.is_bluetooth else ""
        print(f"🚌 Starting shared audio bus{bt_mode}...")
        print(f"   Device: {self.config.device_index or 'default'}, Rate: {self.config.sample_rate}Hz")
        print(f"   Chunk: {self.config.chunk_size} frames ({self.config.chunk_size/self.config.sample_rate*1000:.0f}ms)")
        print(f"   Buffer: {self.config.buffer_seconds}s ring buffer")
        
        try:
            self._stream = sd.InputStream(
                device=self.config.device_index,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='int16',
                blocksize=self.config.chunk_size,
                latency=self.config.latency,
                callback=self._audio_callback
            )
            self._stream.start()
            self._is_running = True
            print("✅ Shared audio bus started")
            
        except Exception as e:
            print(f"❌ Failed to start shared audio bus: {e}")
            raise
    
    async def stop(self) -> None:
        """
        Stop the input stream.
        
        Call this when exiting conversation mode (before resuming wake word).
        """
        if not self._is_running and self._stream is None:
            return
        
        print("🛑 Stopping shared audio bus...")
        
        # Signal shutdown
        self._shutdown_flag.set()
        
        # Give callbacks time to complete
        await asyncio.sleep(0.05)
        
        # Stop and close stream
        if self._stream:
            try:
                self._stream.stop()
                await asyncio.sleep(0.05)
                self._stream.close()
                await asyncio.sleep(0.03)
                print("✅ Shared audio bus stopped")
            except Exception as e:
                print(f"⚠️  Error stopping shared audio bus: {e}")
            finally:
                self._stream = None
        
        self._is_running = False
        self._event_loop = None
        
        # Clear all subscribers
        with self._subscribers_lock:
            self._subscribers.clear()
            self._sorted_subscribers.clear()
    
    def subscribe(self, name: str, callback: Callable, priority: int = 0) -> None:
        """
        Add a subscriber to receive audio callbacks.
        
        This is instant - no device setup required.
        
        Args:
            name: Unique identifier for the subscriber (e.g., "transcription", "barge_in")
            callback: Function matching sounddevice callback signature:
                      (indata: np.ndarray, frames: int, time_info: dict, status: CallbackFlags)
            priority: Higher priority subscribers are called first (default 0)
                     Use priority=10 for barge-in to detect before transcription processes
        """
        with self._subscribers_lock:
            if name in self._subscribers:
                # Reactivate existing subscriber
                self._subscribers[name].callback = callback
                self._subscribers[name].priority = priority
                self._subscribers[name].active = True
                print(f"⚡ Reactivated subscriber: {name} (priority: {priority})")
            else:
                # Add new subscriber
                self._subscribers[name] = AudioSubscriber(
                    name=name,
                    callback=callback,
                    priority=priority,
                    active=True
                )
                print(f"⚡ Added subscriber: {name} (priority: {priority})")
            
            self._update_sorted_subscribers()
    
    def unsubscribe(self, name: str) -> None:
        """
        Remove a subscriber.
        
        This is instant - no device teardown required.
        
        Args:
            name: Identifier of the subscriber to remove
        """
        with self._subscribers_lock:
            if name in self._subscribers:
                self._subscribers[name].active = False
                self._update_sorted_subscribers()
                print(f"⚡ Unsubscribed: {name}")
            else:
                print(f"ℹ️  Subscriber not found: {name}")
    
    def pause_subscriber(self, name: str) -> None:
        """
        Temporarily pause a subscriber without removing it.
        
        Args:
            name: Identifier of the subscriber to pause
        """
        with self._subscribers_lock:
            if name in self._subscribers:
                self._subscribers[name].active = False
                self._update_sorted_subscribers()
    
    def resume_subscriber(self, name: str) -> None:
        """
        Resume a paused subscriber.
        
        Args:
            name: Identifier of the subscriber to resume
        """
        with self._subscribers_lock:
            if name in self._subscribers:
                self._subscribers[name].active = True
                self._update_sorted_subscribers()
    
    def get_buffer(self, seconds: float) -> Optional[np.ndarray]:
        """
        Get recent audio from the ring buffer.
        
        Use this for prefill when transitioning from barge-in to transcription.
        
        Args:
            seconds: Number of seconds of audio to retrieve
            
        Returns:
            numpy array of int16 audio samples, or None if buffer is empty
        """
        return self._ring_buffer.get_last(seconds)
    
    def get_buffer_bytes(self, seconds: float) -> Optional[bytes]:
        """
        Get recent audio as bytes.
        
        Use this for APIs that expect PCM bytes.
        
        Args:
            seconds: Number of seconds of audio to retrieve
            
        Returns:
            bytes of int16 PCM audio, or None if buffer is empty
        """
        return self._ring_buffer.get_last_bytes(seconds)
    
    def clear_buffer(self) -> None:
        """Clear the ring buffer."""
        self._ring_buffer.clear()
    
    @property
    def is_running(self) -> bool:
        """Check if the bus is currently running."""
        return self._is_running
    
    @property
    def buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return self._ring_buffer.duration
    
    @property
    def subscriber_count(self) -> int:
        """Get number of active subscribers."""
        with self._subscribers_lock:
            return len([s for s in self._subscribers.values() if s.active])
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bus status for debugging."""
        with self._subscribers_lock:
            subscribers = {
                name: {"priority": s.priority, "active": s.active}
                for name, s in self._subscribers.items()
            }
        
        return {
            "is_running": self._is_running,
            "enabled": self.config.enabled,
            "buffer_duration": self._ring_buffer.duration,
            "subscribers": subscribers,
            "sample_rate": self.config.sample_rate,
            "chunk_size": self.config.chunk_size,
        }


# Global instance (optional - orchestrator can also create its own)
_shared_bus: Optional[SharedAudioBus] = None


def get_shared_audio_bus() -> Optional[SharedAudioBus]:
    """Get the global shared audio bus instance."""
    return _shared_bus


def set_shared_audio_bus(bus: Optional[SharedAudioBus]) -> None:
    """Set the global shared audio bus instance."""
    global _shared_bus
    _shared_bus = bus

