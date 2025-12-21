"""
Chatterbox TTS provider - Local neural TTS from Resemble AI.

Features:
- High-quality neural TTS running locally
- Voice cloning from audio samples
- Paralinguistic tags (Turbo mode): [chuckle], [sigh], [laugh], etc.
- Apple Silicon (MPS) acceleration with CPU fallback
- Creative controls (cfg, exaggeration)
- Local model storage in audio_data/chatterbox_models/

Install: pip install chatterbox-tts torchaudio
Requires: huggingface-cli login (first time only)
"""

import asyncio
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from ...interfaces.text_to_speech import TextToSpeechInterface
    from ...models.data_models import AudioOutput, AudioFormat
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.text_to_speech import TextToSpeechInterface
    from models.data_models import AudioOutput, AudioFormat

# Lazy imports for heavy dependencies
_scipy_wavfile = None
_ChatterboxTTS = None
_ChatterboxTurboTTS = None
_torch = None
_np = None

# Default model directory (relative to project root)
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "audio_data" / "chatterbox_models"


def _set_hf_cache(model_dir: Path):
    """
    Redirect HuggingFace model cache to local project directory.
    This ensures models are stored in audio_data/chatterbox_models/ instead of ~/.cache/huggingface/
    Note: We only redirect the hub cache, not HF_HOME, so the token lookup still works from ~/.cache/huggingface/
    """
    model_dir = Path(model_dir).resolve()
    hub_dir = model_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    
    # Only redirect the model cache, not the entire HF_HOME (preserves token location)
    os.environ['HF_HUB_CACHE'] = str(hub_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(hub_dir)
    
    print(f"📁 Model storage: {model_dir}")


def _load_dependencies():
    """Lazy load heavy dependencies."""
    global _scipy_wavfile, _ChatterboxTTS, _ChatterboxTurboTTS, _torch, _np
    
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("torch not installed. Run: pip install torch")
    
    if _np is None:
        import numpy as np
        _np = np
    
    if _scipy_wavfile is None:
        try:
            import scipy.io.wavfile as wavfile
            _scipy_wavfile = wavfile
        except ImportError:
            raise ImportError("scipy not installed. Run: pip install scipy")
    
    if _ChatterboxTTS is None:
        try:
            from chatterbox.tts import ChatterboxTTS
            _ChatterboxTTS = ChatterboxTTS
        except ImportError:
            pass  # May only have Turbo
    
    if _ChatterboxTurboTTS is None:
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            _ChatterboxTurboTTS = ChatterboxTurboTTS
        except ImportError:
            pass  # May only have standard
    
    if _ChatterboxTTS is None and _ChatterboxTurboTTS is None:
        raise ImportError(
            "chatterbox-tts not installed. Run: pip install chatterbox-tts torchaudio"
        )


def check_models_downloaded(model_dir: Path, model_type: str = "turbo") -> bool:
    """
    Check if Chatterbox models are already downloaded.
    
    Returns:
        True if models exist locally, False otherwise
    """
    hub_dir = Path(model_dir) / "hub"
    if not hub_dir.exists():
        return False
    
    # Look for model directories
    model_dirs = list(hub_dir.glob("models--ResembleAI--chatterbox*"))
    return len(model_dirs) > 0


class ChatterboxTTSProvider(TextToSpeechInterface):
    """
    Chatterbox TTS provider - local neural speech synthesis.
    
    Supports two model variants:
    - Standard (ChatterboxTTS): 500M params, creative controls
    - Turbo (ChatterboxTurboTTS): 350M params, faster, paralinguistic tags
    
    Configuration options:
    - model_type: "turbo" or "standard" (default: "turbo")
    - model_dir: Directory for local model storage (default: audio_data/chatterbox_models)
    - device: "mps", "cuda", or "cpu" (default: auto-detect)
    - voice_prompt_path: Path to ~10s reference audio for voice cloning
    - cfg: Classifier-free guidance scale (default: 0.5)
    - exaggeration: Expressiveness control (default: 0.5)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Chatterbox TTS provider.
        
        Args:
            config: Configuration dictionary containing:
                - model_type: "turbo" or "standard" (default: "turbo")
                - model_dir: Local directory for model storage (default: audio_data/chatterbox_models)
                - device: "mps", "cuda", or "cpu" (default: auto-detect)
                - voice_prompt_path: Optional path to reference voice audio
                - cfg: Classifier-free guidance (0.0-1.0, default: 0.5) - standard only
                - exaggeration: Expressiveness (0.0-1.0, default: 0.5) - standard only
        """
        self.model_type = config.get('model_type', 'turbo')
        self.model_dir = Path(config.get('model_dir', DEFAULT_MODEL_DIR))
        self.device = config.get('device', 'auto')
        self.voice_prompt_path = config.get('voice_prompt_path')
        self.cfg = config.get('cfg', 0.5)
        self.exaggeration = config.get('exaggeration', 0.5)
        
        # Model state (stored locally in model_dir)
        self._model = None
        self._model_lock = threading.Lock()
        self._sample_rate = None
        
        # Playback state
        self._is_playing = False
        self._stop_playback = threading.Event()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        import torch
        
        if self.device != 'auto':
            return self.device
        
        # Try MPS (Apple Silicon) first
        if torch.backends.mps.is_available():
            try:
                # Test MPS actually works
                test_tensor = torch.zeros(1, device='mps')
                del test_tensor
                print("🍎 Using MPS (Apple Silicon) acceleration")
                return 'mps'
            except Exception as e:
                print(f"⚠️  MPS available but failed test: {e}")
        
        # Try CUDA
        if torch.cuda.is_available():
            print("🎮 Using CUDA (NVIDIA GPU) acceleration")
            return 'cuda'
        
        # Fallback to CPU
        print("💻 Using CPU (no GPU acceleration)")
        return 'cpu'
    
    async def initialize(self) -> bool:
        """Initialize the Chatterbox model."""
        try:
            print(f"🔄 Initializing Chatterbox {self.model_type.upper()} TTS...")
            
            # Set HuggingFace cache to local project directory BEFORE loading
            _set_hf_cache(self.model_dir)
            
            # Check if models already exist locally
            models_exist = check_models_downloaded(self.model_dir, self.model_type)
            if models_exist:
                print(f"✅ Models found locally")
            else:
                print(f"📥 First run - downloading models (~1-2GB)...")
                print(f"   (Requires: huggingface-cli login)")
            
            # Load dependencies
            _load_dependencies()
            
            # Detect device
            device = self._detect_device()
            
            # Load model in thread (blocking operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model, device)
            
            print(f"✅ Chatterbox TTS ready")
            print(f"   Model: {self.model_type}, Device: {device}")
            if self.voice_prompt_path:
                print(f"   Voice cloning: {self.voice_prompt_path}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Chatterbox TTS not available: {e}")
            print("   Install with: pip install chatterbox-tts torchaudio")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize Chatterbox TTS: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model(self, device: str):
        """Load the Chatterbox model (runs in thread). HuggingFace handles caching."""
        with self._model_lock:
            print(f"🔄 Loading Chatterbox {self.model_type} model...")
            
            if self.model_type == 'turbo':
                if _ChatterboxTurboTTS is None:
                    raise ImportError("ChatterboxTurboTTS not available")
                # HuggingFace caches models automatically in ~/.cache/huggingface/
                self._model = _ChatterboxTurboTTS.from_pretrained(device=device)
            else:
                if _ChatterboxTTS is None:
                    raise ImportError("ChatterboxTTS not available")
                self._model = _ChatterboxTTS.from_pretrained(device=device)
            
            self._sample_rate = self._model.sr
            print(f"✅ Model loaded (sample rate: {self._sample_rate} Hz)")
    
    async def synthesize(self,
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize (Turbo mode supports tags like [chuckle])
            voice: Optional path to voice reference audio (overrides config)
            speed: Not directly supported (use cfg to affect pacing)
            pitch: Not directly supported
            
        Returns:
            AudioOutput containing synthesized audio
        """
        if self._model is None:
            raise RuntimeError("Chatterbox model not initialized. Call initialize() first.")
        
        # Determine voice prompt
        voice_prompt = voice or self.voice_prompt_path
        
        # Generate in thread (blocking operation)
        loop = asyncio.get_event_loop()
        wav_tensor = await loop.run_in_executor(
            None, 
            self._generate_speech, 
            text, 
            voice_prompt
        )
        
        # Convert tensor to bytes
        audio_bytes = self._tensor_to_wav_bytes(wav_tensor)
        
        return AudioOutput(
            audio_data=audio_bytes,
            format=AudioFormat.WAV,
            sample_rate=self._sample_rate,
            voice=f"chatterbox_{self.model_type}",
            language="en-US",
            metadata={
                'model_type': self.model_type,
                'voice_cloned': voice_prompt is not None,
                'cfg': self.cfg,
                'exaggeration': self.exaggeration,
                'text': text
            }
        )
    
    def _generate_speech(self, text: str, voice_prompt: Optional[str]) -> 'torch.Tensor':
        """Generate speech from text (runs in thread)."""
        import inspect
        
        with self._model_lock:
            # Build kwargs based on what the model's generate() accepts
            generate_sig = inspect.signature(self._model.generate)
            kwargs = {}
            
            # Add voice prompt if cloning
            if voice_prompt and 'audio_prompt_path' in generate_sig.parameters:
                kwargs['audio_prompt_path'] = voice_prompt
            
            # Add creative controls if supported
            if 'cfg' in generate_sig.parameters:
                kwargs['cfg'] = self.cfg
            if 'exaggeration' in generate_sig.parameters:
                kwargs['exaggeration'] = self.exaggeration
            
            # Generate audio
            wav = self._model.generate(text, **kwargs)
            
            return wav
    
    def _tensor_to_wav_bytes(self, wav_tensor) -> bytes:
        """Convert PyTorch tensor to WAV bytes using scipy."""
        # Ensure tensor is on CPU
        if wav_tensor.device.type != 'cpu':
            wav_tensor = wav_tensor.cpu()
        
        # Convert to numpy and squeeze to 1D
        wav_np = wav_tensor.numpy().squeeze()
        
        # Convert float to int16 for WAV
        wav_int16 = (_np.clip(wav_np, -1.0, 1.0) * 32767).astype(_np.int16)
        
        # Save to bytes buffer using scipy
        buffer = io.BytesIO()
        _scipy_wavfile.write(buffer, self._sample_rate, wav_int16)
        buffer.seek(0)
        return buffer.read()
    
    def play_audio(self, audio: AudioOutput) -> None:
        """Play synthesized audio (synchronous)."""
        asyncio.run(self.play_audio_async(audio))
    
    async def play_audio_async(self, audio: AudioOutput) -> None:
        """Play synthesized audio with interruption support."""
        import subprocess
        import platform
        
        if self._is_playing:
            self.stop_audio()
            await asyncio.sleep(0.1)
        
        self._is_playing = True
        self._stop_playback.clear()
        
        try:
            # Write audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio.audio_data)
                temp_path = f.name
            
            try:
                # Play using afplay on macOS, aplay on Linux
                if platform.system() == 'Darwin':
                    proc = await asyncio.create_subprocess_exec(
                        'afplay', temp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                else:
                    proc = await asyncio.create_subprocess_exec(
                        'aplay', temp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                
                # Wait for playback, checking for interruption
                while proc.returncode is None:
                    if self._stop_playback.is_set():
                        proc.kill()
                        await proc.wait()
                        break
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass
                        
            finally:
                # Cleanup temp file
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"❌ Playback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_playing = False
    
    def stop_audio(self) -> None:
        """Stop audio playback immediately."""
        self._stop_playback.set()
        self._is_playing = False
        
        # Kill any running afplay/aplay processes
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['pkill', '-9', 'afplay'], capture_output=True)
            else:
                subprocess.run(['pkill', '-9', 'aplay'], capture_output=True)
        except Exception:
            pass
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    async def save_audio(self, audio: AudioOutput, path: Path) -> None:
        """Save audio to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio.audio_data)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._is_playing:
            self.stop_audio()
        
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
        
        # Clear GPU memory if applicable
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have empty_cache, but we can try
                pass
        except Exception:
            pass
        
        print("✅ Chatterbox TTS cleaned up")
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': False,
            'batch': True,
            'voices': ['default', 'cloned'],
            'languages': ['en-US'] if self.model_type != 'multilingual' else ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'ja', 'ko', 'zh'],
            'audio_formats': ['wav'],
            'speed_range': (0.5, 2.0),  # Via cfg
            'pitch_range': (0, 0),  # Not directly supported
            'features': [
                'neural_tts',
                'voice_cloning',
                'local',
                'offline',
                'paralinguistic_tags' if self.model_type == 'turbo' else 'creative_controls',
                'mps_acceleration',
            ],
            'latency': 'medium',  # ~1-3s depending on text length
            'requires_api': False
        }


# Convenience function to test the provider
async def test_chatterbox():
    """Quick test of Chatterbox TTS."""
    config = {
        'model_type': 'turbo',
        'device': 'auto',
    }
    
    provider = ChatterboxTTSProvider(config)
    
    if await provider.initialize():
        # Test with paralinguistic tag
        text = "Hey there! [chuckle] This is Chatterbox running locally on your Mac."
        print(f"🎤 Synthesizing: {text}")
        
        audio = await provider.synthesize(text)
        print(f"✅ Generated {len(audio.audio_data)} bytes of audio")
        
        # Play it
        print("🔊 Playing...")
        await provider.play_audio_async(audio)
        
        await provider.cleanup()
    else:
        print("❌ Failed to initialize")


if __name__ == "__main__":
    asyncio.run(test_chatterbox())
