"""
Audio processing utilities for speech detection and chunking.
Handles VAD (Voice Activity Detection) and audio frame management.
"""

import io
import time
import wave
from collections import deque
from typing import Optional, Tuple

import numpy as np
import webrtcvad


def wav_bytes_from_frames(frames: list, sample_rate: int = 16000) -> io.BytesIO:
    """
    Combine audio frames into an in-memory WAV file.
    
    Args:
        frames: List of raw int16 audio bytes
        sample_rate: Sample rate in Hz
        
    Returns:
        BytesIO buffer containing WAV file
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    return buf


def calculate_rms(audio_data: bytes) -> float:
    """
    Calculate Root Mean Square (volume level) of audio data.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        RMS value
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    return np.sqrt(np.mean(audio_array**2))


class VADChunker:
    """
    Voice Activity Detection chunker that collects mic frames until silence is detected.
    Yields complete speech chunks for transcription.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 frame_ms: int,
                 vad_mode: int,
                 silence_end_sec: float,
                 max_utterance_sec: float):
        """
        Initialize VAD chunker.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds (10, 20, or 30)
            vad_mode: VAD aggressiveness (0-3, 3 = most aggressive)
            silence_end_sec: Seconds of silence that ends a speech chunk
            max_utterance_sec: Maximum utterance length before force flush
        """
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_utterance_sec = max_utterance_sec
        self.vad = webrtcvad.Vad(vad_mode)
        self.speech_buf = []
        self.ring = deque(maxlen=int(silence_end_sec * 1000 // frame_ms))
        self.last_voice_t = None
        self.is_speaking = False
        
    def process(self, frame_bytes: bytes) -> Optional[bytes]:
        """
        Process an audio frame and return a complete chunk if ready.
        
        Args:
            frame_bytes: Raw audio frame bytes
            
        Returns:
            Complete speech chunk bytes if silence detected, None otherwise
        """
        now = time.monotonic()
        voiced = self.vad.is_speech(frame_bytes, self.sample_rate)
        
        if voiced:
            if not self.is_speaking:
                print("🟢 [CHUNK START] Speech detected")
                self.is_speaking = True
            self.speech_buf.append(frame_bytes)
            self.last_voice_t = now
            self.ring.clear()
        else:
            self.ring.append(frame_bytes)
            # Only flush if we have speech buffer AND ring buffer is full (indicating sustained silence)
            if self.speech_buf and len(self.ring) == self.ring.maxlen:
                return self._flush()
        
        # Safety flush for very long utterances
        if len(self.speech_buf) * self.frame_ms / 1000 > self.max_utterance_sec:
            return self._flush()
        
        return None
    
    def _flush(self) -> Optional[bytes]:
        """Flush accumulated speech buffer and return as chunk."""
        if not self.speech_buf:
            return None
        print("🔴 [CHUNK END] Silence detected, flushing chunk")
        self.is_speaking = False
        chunk = b"".join(self.speech_buf)
        self.speech_buf.clear()
        self.ring.clear()
        self.last_voice_t = None
        return chunk
    
    def is_speech(self, frame_bytes: bytes) -> bool:
        """Check if frame contains speech."""
        return self.vad.is_speech(frame_bytes, self.sample_rate)