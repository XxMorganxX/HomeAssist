"""
Audio processing utilities for speech detection and chunking.
Handles VAD (Voice Activity Detection) and audio frame management.
Includes Acoustic Echo Cancellation (AEC) for noise suppression.
"""

import io
import time
import wave
from collections import deque
from typing import Optional, Tuple

import numpy as np
import webrtcvad
from .aec_processor import AECProcessor, AudioCaptureManager


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
    Includes optional Acoustic Echo Cancellation (AEC).
    """
    
    def __init__(self, 
                 sample_rate: int,
                 frame_ms: int,
                 vad_mode: int,
                 silence_end_sec: float,
                 max_utterance_sec: float,
                 aec_enabled: bool = False,
                 aec_config: Optional[dict] = None):
        """
        Initialize VAD chunker.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds (10, 20, or 30)
            vad_mode: VAD aggressiveness (0-3, 3 = most aggressive)
            silence_end_sec: Seconds of silence that ends a speech chunk
            max_utterance_sec: Maximum utterance length before force flush
            aec_enabled: Enable Acoustic Echo Cancellation
            aec_config: AEC configuration dictionary
        """
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_utterance_sec = max_utterance_sec
        self.vad = webrtcvad.Vad(vad_mode)
        self.speech_buf = []
        self.ring = deque(maxlen=int(silence_end_sec * 1000 // frame_ms))
        self.last_voice_t = None
        self.is_speaking = False
        
        # Initialize AEC if enabled
        self.aec_enabled = aec_enabled
        if aec_enabled:
            if aec_config is None:
                aec_config = {}
            self.aec_processor = AECProcessor(
                filter_length=aec_config.get('filter_length', 300),
                step_size=aec_config.get('step_size', 0.05),
                sample_rate=sample_rate,
                frame_size=sample_rate * frame_ms // 1000,
                delay_samples=aec_config.get('delay_samples', 800),
                reference_buffer_sec=aec_config.get('reference_buffer_sec', 5.0)
            )
            self.audio_capture = AudioCaptureManager(
                strategy=aec_config.get('capture_strategy', 'file_based')
            )
            print("🔊 VADChunker initialized with AEC enabled")
        else:
            self.aec_processor = None
            self.audio_capture = None
            print("🔊 VADChunker initialized without AEC")
        
    def process(self, frame_bytes: bytes) -> Optional[bytes]:
        """
        Process an audio frame and return a complete chunk if ready.
        
        Args:
            frame_bytes: Raw audio frame bytes
            
        Returns:
            Complete speech chunk bytes if silence detected, None otherwise
        """
        # Apply AEC if enabled
        processed_frame = frame_bytes
        if self.aec_enabled and self.aec_processor is not None:
            processed_frame = self.aec_processor.process_frame(frame_bytes)
        
        now = time.monotonic()
        voiced = self.vad.is_speech(processed_frame, self.sample_rate)
        
        if voiced:
            if not self.is_speaking:
                print("🟢 [CHUNK START] Speech detected")
                self.is_speaking = True
            self.speech_buf.append(processed_frame)
            self.last_voice_t = now
            self.ring.clear()
        else:
            self.ring.append(processed_frame)
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
    
    def add_reference_audio_file(self, audio_file_path: str) -> None:
        """
        Add reference audio from a file that's about to be played.
        
        Args:
            audio_file_path: Path to the audio file being played
        """
        if self.aec_enabled and self.audio_capture is not None:
            audio_data = self.audio_capture.capture_audio_file(audio_file_path)
            if len(audio_data) > 0 and self.aec_processor is not None:
                self.aec_processor.add_reference_audio(audio_data)
                print(f"🎵 Added reference audio: {audio_file_path}")
    
    def get_aec_status(self) -> dict:
        """Get AEC processor status."""
        if self.aec_enabled and self.aec_processor is not None:
            return self.aec_processor.get_status()
        else:
            return {"aec_enabled": False}
    
    def reset_aec(self) -> None:
        """Reset AEC processor."""
        if self.aec_enabled and self.aec_processor is not None:
            self.aec_processor.reset()