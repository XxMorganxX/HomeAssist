"""
Piper TTS provider - Fast local neural TTS using ONNX models.

Features:
- Near-instant generation (~50x realtime)
- High-quality neural voices
- Small model sizes (15-60MB each)
- CPU-only (no GPU required)
- Offline operation
- Local model storage

Install: pip install piper-tts
"""

import asyncio
import io
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from ...interfaces.text_to_speech import TextToSpeechInterface
    from ...models.data_models import AudioOutput, AudioFormat
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.text_to_speech import TextToSpeechInterface
    from models.data_models import AudioOutput, AudioFormat

# Lazy imports for piper
_piper = None
_PiperVoice = None

# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "audio_data" / "piper_models"

# Popular voice models with download URLs
# Format: (name, quality, language, url_base)
AVAILABLE_VOICES = {
    # English US
    "en_US-lessac-medium": {
        "quality": "medium",
        "language": "en-US",
        "description": "Clear American English voice",
        "size_mb": 65,
    },
    "en_US-lessac-high": {
        "quality": "high",
        "language": "en-US",
        "description": "High quality American English",
        "size_mb": 100,
    },
    "en_US-ryan-medium": {
        "quality": "medium",
        "language": "en-US",
        "description": "Male American English voice",
        "size_mb": 65,
    },
    "en_US-amy-medium": {
        "quality": "medium",
        "language": "en-US",
        "description": "Female American English voice",
        "size_mb": 65,
    },
    # English GB
    "en_GB-alan-medium": {
        "quality": "medium",
        "language": "en-GB",
        "description": "Male British English voice",
        "size_mb": 65,
    },
    "en_GB-jenny_dioco-medium": {
        "quality": "medium",
        "language": "en-GB",
        "description": "Female British English voice",
        "size_mb": 65,
    },
}

# Base URL for Piper voice models
# Format: https://huggingface.co/rhasspy/piper-voices/resolve/main/<lang>/<locale>/<voice>/<quality>/<filename>
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _load_piper():
    """Lazy load piper module."""
    global _piper, _PiperVoice
    
    if _PiperVoice is None:
        try:
            from piper import PiperVoice
            _PiperVoice = PiperVoice
        except ImportError:
            raise ImportError(
                "piper-tts not installed. Run: pip install piper-tts"
            )


def download_voice_model(voice_name: str, model_dir: Path) -> Path:
    """
    Download a Piper voice model if not present.
    
    Args:
        voice_name: Name of the voice (e.g., "en_US-lessac-medium")
        model_dir: Directory to store models
        
    Returns:
        Path to the .onnx model file
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected file paths
    onnx_path = model_dir / f"{voice_name}.onnx"
    json_path = model_dir / f"{voice_name}.onnx.json"
    
    # Check if already downloaded
    if onnx_path.exists() and json_path.exists():
        return onnx_path
    
    print(f"📥 Downloading Piper voice: {voice_name}...")
    
    # Parse voice name: en_US-lessac-medium -> lang=en, locale=en_US, voice=lessac, quality=medium
    parts = voice_name.split("-")
    if len(parts) >= 3:
        locale = parts[0]  # e.g., "en_US"
        voice = parts[1]   # e.g., "lessac"
        quality = parts[2] # e.g., "medium"
        
        # Extract language code from locale (en_US -> en)
        lang = locale.split("_")[0]
        
        # Construct URLs matching HuggingFace structure:
        # https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
        base_url = f"{PIPER_VOICES_URL}/{lang}/{locale}/{voice}/{quality}"
        onnx_url = f"{base_url}/{voice_name}.onnx"
        json_url = f"{base_url}/{voice_name}.onnx.json"
    else:
        raise ValueError(f"Invalid voice name format: {voice_name}")
    
    try:
        import urllib.request
        
        # Download ONNX model
        print(f"   Downloading model (~{AVAILABLE_VOICES.get(voice_name, {}).get('size_mb', '?')}MB)...")
        urllib.request.urlretrieve(onnx_url, onnx_path)
        
        # Download JSON config
        print(f"   Downloading config...")
        urllib.request.urlretrieve(json_url, json_path)
        
        print(f"✅ Voice downloaded: {voice_name}")
        return onnx_path
        
    except Exception as e:
        # Cleanup partial downloads
        onnx_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download voice {voice_name}: {e}")


class PiperTTSProvider(TextToSpeechInterface):
    """
    Piper TTS provider - fast local neural speech synthesis.
    
    Uses ONNX models for near-instant speech generation (~50x realtime).
    Models are small (15-100MB) and run on CPU.
    
    Configuration options:
    - voice: Voice model name (default: "en_US-lessac-medium")
    - model_dir: Directory for local model storage
    - speed: Speech rate multiplier (default: 1.0)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Piper TTS provider.
        
        Args:
            config: Configuration dictionary containing:
                - voice: Voice name (default: "en_US-lessac-medium")
                - model_dir: Local model storage (default: audio_data/piper_models)
                - speed: Rate multiplier 0.5-2.0 (default: 1.0)
        """
        self.voice_name = config.get('voice', 'en_US-lessac-medium')
        self.model_dir = Path(config.get('model_dir', DEFAULT_MODEL_DIR))
        self.speed = config.get('speed', 1.0)
        
        # Voice model (loaded on initialize)
        self._voice = None
        self._voice_lock = threading.Lock()
        self._sample_rate = None
        
        # Playback state
        self._is_playing = False
        self._stop_playback = threading.Event()
        self._current_process = None
    
    async def initialize(self) -> bool:
        """Initialize the Piper TTS provider."""
        try:
            print(f"🔄 Initializing Piper TTS...")
            print(f"📁 Model storage: {self.model_dir}")
            
            # Load piper module
            _load_piper()
            
            # Download voice model if needed
            model_path = await asyncio.get_event_loop().run_in_executor(
                None, download_voice_model, self.voice_name, self.model_dir
            )
            
            # Load voice model
            print(f"🔄 Loading voice: {self.voice_name}...")
            self._voice = _PiperVoice.load(str(model_path))
            self._sample_rate = self._voice.config.sample_rate
            
            print(f"✅ Piper TTS ready")
            print(f"   Voice: {self.voice_name}")
            print(f"   Sample rate: {self._sample_rate} Hz")
            print(f"   Speed: {self.speed}x")
            
            return True
            
        except ImportError as e:
            print(f"❌ Piper TTS not available: {e}")
            print("   Install with: pip install piper-tts")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize Piper TTS: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice name (ignored, use config)
            speed: Rate multiplier (default: config speed)
            pitch: Not supported
            
        Returns:
            AudioOutput with WAV audio data
        """
        if self._voice is None:
            raise RuntimeError("Piper TTS not initialized. Call initialize() first.")
        
        effective_speed = speed if speed is not None else self.speed
        
        # Generate audio in executor (CPU-bound)
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._generate_speech, text, effective_speed
        )
        
        return AudioOutput(
            audio_data=audio_bytes,
            format=AudioFormat.WAV,
            sample_rate=self._sample_rate,
            voice=self.voice_name,
            language="en-US",
            metadata={
                'engine': 'piper',
                'speed': effective_speed,
                'text': text
            }
        )
    
    def _generate_speech(self, text: str, speed: float) -> bytes:
        """Generate speech audio (runs in thread)."""
        from piper.config import SynthesisConfig
        
        with self._voice_lock:
            audio_buffer = io.BytesIO()
            
            # Configure synthesis with speed adjustment
            # Piper uses length_scale for speed (inverse: 0.5 = 2x speed)
            length_scale = 1.0 / speed if speed > 0 else 1.0
            syn_config = SynthesisConfig(length_scale=length_scale)
            
            # Write WAV with piper's built-in method
            with wave.open(audio_buffer, 'wb') as wav_file:
                self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            
            return audio_buffer.getvalue()
    
    async def play_audio_async(self, audio: AudioOutput) -> None:
        """Play synthesized audio."""
        if self._is_playing:
            self.stop_audio()
            await asyncio.sleep(0.05)
        
        self._is_playing = True
        self._stop_playback.clear()
        
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio.audio_data)
                tmp_path = f.name
            
            try:
                # Play with system command
                import platform
                if platform.system() == 'Darwin':
                    self._current_process = await asyncio.create_subprocess_exec(
                        'afplay', tmp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                else:
                    self._current_process = await asyncio.create_subprocess_exec(
                        'aplay', tmp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                
                # Wait for playback, checking for interruption
                while self._current_process.returncode is None:
                    if self._stop_playback.is_set():
                        self._current_process.kill()
                        break
                    try:
                        await asyncio.wait_for(
                            self._current_process.wait(),
                            timeout=0.05
                        )
                    except asyncio.TimeoutError:
                        pass
                        
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
        finally:
            self._is_playing = False
            self._current_process = None
    
    def stop_audio(self) -> None:
        """Stop audio playback immediately."""
        self._stop_playback.set()
        
        if self._current_process:
            try:
                self._current_process.kill()
            except Exception:
                pass
        
        # Also kill any afplay/aplay processes
        try:
            import platform
            if platform.system() == 'Darwin':
                subprocess.run(['pkill', '-9', 'afplay'], capture_output=True)
            else:
                subprocess.run(['pkill', '-9', 'aplay'], capture_output=True)
        except Exception:
            pass
        
        self._is_playing = False
    
    def play_audio(self, audio: AudioOutput) -> None:
        """Play synthesized audio (synchronous version)."""
        asyncio.run(self.play_audio_async(audio))
    
    async def save_audio(self, audio: AudioOutput, path: Path) -> None:
        """Save audio to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_event_loop().run_in_executor(
            None, path.write_bytes, audio.audio_data
        )
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._is_playing:
            self.stop_audio()
        
        self._voice = None
        print("✅ Piper TTS cleaned up")
    
    async def speak_streaming(self, text_generator, on_sentence_start=None, on_complete=None) -> bool:
        """
        Stream TTS - speak sentences as they arrive.
        
        Piper is fast enough that we can generate and play sentence-by-sentence
        with minimal latency.
        """
        import re
        
        self._is_playing = True
        self._stop_playback.clear()
        
        buffer = ""
        sentence_end = re.compile(r'[.!?]+\s*')
        sentence_count = 0
        was_interrupted = False
        
        try:
            async for chunk in text_generator:
                if self._stop_playback.is_set():
                    was_interrupted = True
                    break
                
                buffer += chunk
                
                # Extract and speak complete sentences
                while True:
                    match = sentence_end.search(buffer)
                    if not match:
                        break
                    
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    
                    if sentence:
                        sentence_count += 1
                        
                        if self._stop_playback.is_set():
                            was_interrupted = True
                            break
                        
                        if on_sentence_start:
                            on_sentence_start(sentence_count, sentence)
                        
                        # Generate and play
                        audio = await self.synthesize(sentence)
                        await self.play_audio_async(audio)
                
                if was_interrupted:
                    break
            
            # Speak remaining buffer
            if buffer.strip() and not self._stop_playback.is_set():
                sentence_count += 1
                if on_sentence_start:
                    on_sentence_start(sentence_count, buffer.strip())
                audio = await self.synthesize(buffer.strip())
                await self.play_audio_async(audio)
            
            if on_complete:
                on_complete(was_interrupted)
            
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
        return {
            'streaming': True,
            'batch': True,
            'voices': list(AVAILABLE_VOICES.keys()),
            'languages': ['en-US', 'en-GB'],
            'audio_formats': ['wav'],
            'speed_range': (0.5, 2.0),
            'pitch_range': (0, 0),  # No pitch control
            'features': ['offline', 'fast', 'interruptible', 'streaming', 'neural'],
            'latency': 'very_low',
            'requires_api': False,
            'model_size': f"{AVAILABLE_VOICES.get(self.voice_name, {}).get('size_mb', '?')}MB"
        }
    
    @staticmethod
    def list_available_voices() -> Dict[str, dict]:
        """List all available voice models."""
        return AVAILABLE_VOICES.copy()

