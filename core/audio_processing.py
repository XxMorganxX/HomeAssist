import io
import time
import wave
from collections import deque
from typing import Optional, Tuple
import numpy as np
import webrtcvad



"""
Acoustic Echo Cancellation (AEC) processor for noise suppression.
Uses NLMS (Normalized Least Mean Squares) adaptive filtering to remove
speaker output from microphone input.
"""


class AECProcessor:
    """
    Acoustic Echo Cancellation processor using NLMS adaptive filtering.
    Removes speaker audio from microphone input in real-time.
    """
    
    def __init__(self,
                 filter_length: int = 300,
                 step_size: float = 0.05,
                 sample_rate: int = 16000,
                 frame_size: int = 480,
                 delay_samples: int = 600,
                 reference_buffer_sec: float = 5.0):
        """
        Initialize AEC processor.
        
        Args:
            filter_length: Number of filter taps (200-500 typical)
            step_size: NLMS step size (0.01-0.1, smaller = more stable)
            sample_rate: Audio sample rate in Hz
            frame_size: Audio frame size in samples
            delay_samples: Estimated delay between speaker and mic in samples
            reference_buffer_sec: How long to keep reference audio (seconds)
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.delay_samples = delay_samples
        
        # Initialize NLMS filter coefficients
        self.filter_coeffs = np.zeros(filter_length, dtype=np.float32)
        self.eps = 1e-6  # Small value to prevent division by zero
        
        # Reference signal buffer (speaker output)
        buffer_samples = int(reference_buffer_sec * sample_rate)
        self.reference_buffer = deque(maxlen=buffer_samples)
        
        # Delay compensation buffer
        self.delay_buffer = deque(maxlen=delay_samples)
        
        # Statistics
        self.total_frames_processed = 0
        self.echo_reduction_db = 0.0
        self.last_stats_time = time.time()
        
        print(f"🔧 AEC Initialized: filter_len={filter_length}, step={step_size}, delay={delay_samples}")
    
    def add_reference_audio(self, audio_data: np.ndarray) -> None:
        """
        Add reference audio (what's being played through speakers).
        
        Args:
            audio_data: Audio data as numpy array (float32, normalized)
        """
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Add to reference buffer
        self.reference_buffer.extend(audio_data)
    
    def process_microphone_audio(self, mic_audio: np.ndarray) -> np.ndarray:
        """
        Process microphone audio to remove echo.
        
        Args:
            mic_audio: Raw microphone audio as numpy array (int16 or float32)
            
        Returns:
            Echo-cancelled audio as numpy array (same type as input)
        """
        input_dtype = mic_audio.dtype
        
        # Convert to float32 for processing
        if mic_audio.dtype == np.int16:
            mic_float = mic_audio.astype(np.float32) / 32768.0
        else:
            mic_float = mic_audio.astype(np.float32)
        
        # Ensure we have enough reference audio
        if len(self.reference_buffer) < self.filter_length:
            # Not enough reference audio yet, return original
            return mic_audio
        
        # Get reference signal with delay compensation
        ref_start_idx = max(0, len(self.reference_buffer) - len(mic_float) - self.delay_samples)
        ref_end_idx = ref_start_idx + len(mic_float)
        
        if ref_end_idx > len(self.reference_buffer):
            # Not enough reference audio, return original
            return mic_audio
        
        reference_signal = np.array(list(self.reference_buffer))[ref_start_idx:ref_end_idx]
        
        if len(reference_signal) != len(mic_float):
            # Length mismatch, return original
            return mic_audio
        
        # Apply NLMS filtering
        try:
            output_audio = self._apply_nlms_filter(reference_signal, mic_float)
            
            # Update statistics
            self._update_statistics(mic_float, output_audio)
            
        except Exception as e:
            print(f"⚠️ AEC processing error: {e}")
            output_audio = mic_float
        
        # Convert back to original data type
        if input_dtype == np.int16:
            # Clip to prevent overflow
            output_audio = np.clip(output_audio, -1.0, 1.0)
            output_audio = (output_audio * 32767.0).astype(np.int16)
        
        return output_audio
    
    def process_frame(self, mic_frame: bytes, reference_frame: Optional[bytes] = None) -> bytes:
        """
        Process a single audio frame (convenience method for existing pipeline).
        
        Args:
            mic_frame: Microphone audio frame as bytes (int16)
            reference_frame: Reference audio frame as bytes (int16), optional
            
        Returns:
            Processed audio frame as bytes
        """
        # Convert bytes to numpy array
        mic_array = np.frombuffer(mic_frame, dtype=np.int16)
        
        # Add reference audio if provided
        if reference_frame is not None:
            ref_array = np.frombuffer(reference_frame, dtype=np.int16)
            ref_float = ref_array.astype(np.float32) / 32768.0
            self.add_reference_audio(ref_float)
        
        # Process audio
        processed_array = self.process_microphone_audio(mic_array)
        
        # Convert back to bytes
        return processed_array.tobytes()
    
    def _apply_nlms_filter(self, reference: np.ndarray, microphone: np.ndarray) -> np.ndarray:
        """
        Apply NLMS (Normalized Least Mean Squares) filtering.
        
        Args:
            reference: Reference signal (speaker output)
            microphone: Microphone signal (input with echo)
            
        Returns:
            Echo-cancelled microphone signal
        """
        output = np.zeros_like(microphone)
        
        # Process sample by sample for real-time operation
        for i in range(len(microphone)):
            # Get reference window
            if i < self.filter_length:
                # Pad with zeros at the beginning
                ref_window = np.concatenate([
                    np.zeros(self.filter_length - i - 1),
                    reference[:i+1]
                ])
            else:
                ref_window = reference[i-self.filter_length+1:i+1]
            
            # Reverse for convolution (most recent sample first)
            ref_window = ref_window[::-1]
            
            # Predict echo using current filter coefficients
            predicted_echo = np.dot(self.filter_coeffs, ref_window)
            
            # Calculate error (echo-cancelled signal)
            error = microphone[i] - predicted_echo
            output[i] = error
            
            # Update filter coefficients using NLMS
            ref_power = np.dot(ref_window, ref_window) + self.eps
            self.filter_coeffs += (self.step_size * error / ref_power) * ref_window
        
        return output
    
    def _update_statistics(self, original: np.ndarray, processed: np.ndarray) -> None:
        """Update processing statistics."""
        self.total_frames_processed += 1
        
        # Calculate echo reduction every 100 frames
        if self.total_frames_processed % 100 == 0:
            original_power = np.mean(original ** 2)
            processed_power = np.mean(processed ** 2)
            
            if original_power > 0 and processed_power > 0:
                self.echo_reduction_db = 10 * np.log10(original_power / processed_power)
            
            # Print stats every 10 seconds
            current_time = time.time()
            if current_time - self.last_stats_time > 10.0:
                print(f"🔊 AEC Stats: {self.echo_reduction_db:.1f}dB reduction, "
                      f"{self.total_frames_processed} frames processed")
                self.last_stats_time = current_time
    
    def reset(self) -> None:
        """Reset the AEC processor."""
        self.filter_coeffs.fill(0.0)
        self.reference_buffer.clear()
        self.delay_buffer.clear()
        self.total_frames_processed = 0
        print("🔄 AEC processor reset")
    
    def get_status(self) -> dict:
        """Get current AEC status."""
        return {
            "filter_length": self.filter_length,
            "step_size": self.step_size,
            "frames_processed": self.total_frames_processed,
            "echo_reduction_db": self.echo_reduction_db,
            "reference_buffer_size": len(self.reference_buffer),
            "reference_buffer_sec": len(self.reference_buffer) / self.sample_rate
        }


class AudioCaptureManager:
    """
    Manages capturing audio that's being played through speakers.
    Provides multiple strategies for different platforms.
    """
    
    def __init__(self, strategy: str = "file_based"):
        """
        Initialize audio capture manager.
        
        Args:
            strategy: "file_based", "virtual_device", or "system_monitor"
        """
        self.strategy = strategy
        self.captured_audio_buffer = deque(maxlen=16000 * 10)  # 10 seconds
        
        if strategy == "file_based":
            print("🎵 Using file-based audio capture strategy")
        elif strategy == "virtual_device":
            print("🎵 Using virtual device audio capture strategy")
        elif strategy == "system_monitor":
            print("🎵 Using system monitor audio capture strategy")
        else:
            raise ValueError(f"Unknown capture strategy: {strategy}")
    
    def capture_audio_file(self, file_path: str) -> np.ndarray:
        """
        Capture audio from a file that's about to be played.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio data as numpy array
        """
        try:
            import librosa
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            return audio_data.astype(np.float32)
        except ImportError:
            print("⚠️ librosa not installed, using basic file reading")
            # Fallback to basic wave reading
            try:
                import wave
                with wave.open(file_path, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    return audio_array.astype(np.float32) / 32768.0
            except Exception as e:
                print(f"⚠️ Could not read audio file {file_path}: {e}")
                return np.array([], dtype=np.float32)
    
    def add_played_audio(self, audio_data: np.ndarray) -> None:
        """Add audio that was just played to the capture buffer."""
        self.captured_audio_buffer.extend(audio_data)
    
    def get_recent_audio(self, duration_sec: float) -> np.ndarray:
        """Get recently played audio."""
        samples_needed = int(duration_sec * 16000)
        if len(self.captured_audio_buffer) >= samples_needed:
            return np.array(list(self.captured_audio_buffer)[-samples_needed:])
        else:
            return np.array(list(self.captured_audio_buffer))



"""
Audio processing utilities for speech detection and chunking.
Handles VAD (Voice Activity Detection) and audio frame management.
Includes Acoustic Echo Cancellation (AEC) for noise suppression.
"""

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
        
        # Validate frame parameters for WebRTC VAD compatibility
        frame_samples = len(processed_frame) // 2  # 16-bit = 2 bytes per sample
        expected_samples = self.sample_rate * self.frame_ms // 1000
        
        # WebRTC VAD requires exact frame sizes
        if frame_samples != expected_samples:
            # Pad or truncate to expected size
            if frame_samples < expected_samples:
                # Pad with zeros
                padding = (expected_samples - frame_samples) * 2
                processed_frame = processed_frame + b'\x00' * padding
            else:
                # Truncate to expected size
                processed_frame = processed_frame[:expected_samples * 2]
        
        # Ensure frame is properly aligned for int16
        if len(processed_frame) % 2 != 0:
            processed_frame = processed_frame[:-1]  # Remove odd byte
        
        now = time.monotonic()
        
        # Process with WebRTC VAD with error handling
        try:
            voiced = self.vad.is_speech(processed_frame, self.sample_rate)
        except Exception as vad_error:
            # If VAD fails, assume silence to prevent crashes
            if config.DEBUG_MODE:
                print(f"⚠️ VAD processing error: {vad_error}")
            voiced = False
        
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
        # Validate frame parameters for WebRTC VAD compatibility
        frame_samples = len(frame_bytes) // 2  # 16-bit = 2 bytes per sample
        expected_samples = self.sample_rate * self.frame_ms // 1000
        
        # WebRTC VAD requires exact frame sizes
        if frame_samples != expected_samples:
            # Pad or truncate to expected size
            if frame_samples < expected_samples:
                # Pad with zeros
                padding = (expected_samples - frame_samples) * 2
                frame_bytes = frame_bytes + b'\x00' * padding
            else:
                # Truncate to expected size
                frame_bytes = frame_bytes[:expected_samples * 2]
        
        # Ensure frame is properly aligned for int16
        if len(frame_bytes) % 2 != 0:
            frame_bytes = frame_bytes[:-1]  # Remove odd byte
        
        # Process with WebRTC VAD with error handling
        try:
            return self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as vad_error:
            # If VAD fails, assume silence to prevent crashes
            if config.DEBUG_MODE:
                print(f"⚠️ VAD is_speech error: {vad_error}")
            return False
    
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

