"""
Google Cloud Text-to-Speech provider.
"""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import re

from google.cloud import texttospeech

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


class GoogleTTSProvider(TextToSpeechInterface):
    """Google Cloud TTS batch implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google TTS provider.
        
        Args:
            config: Configuration dictionary containing:
                - voice: Voice name (e.g., "en-US-Chirp3-HD-Sadachbia")
                - speed: Speed modifier (default: 1.3)
                - pitch: Pitch modifier in semitones (default: -1.2)
                - language_code: Language code (default: "en-US")
                - audio_encoding: Audio encoding (default: "MP3")
        """
        self.voice_name = config.get('voice', 'en-US-Chirp3-HD-Sadachbia')
        self.speed = config.get('speed', 1.3)
        self.pitch = config.get('pitch', -1.2)
        self.language_code = config.get('language_code', 'en-US')
        self.audio_encoding = config.get('audio_encoding', 'MP3')
        
        # TTS client (lazy initialization)
        self._client = None
        self._is_playing = False
        self._stop_playback = False
        self._afplay_process = None
        
        # Map encoding strings to Google enums
        self.encoding_map = {
            'MP3': texttospeech.AudioEncoding.MP3,
            'WAV': texttospeech.AudioEncoding.LINEAR16,
            'OGG': texttospeech.AudioEncoding.OGG_OPUS,
        }
    
    async def initialize(self) -> bool:
        """Initialize the Google TTS provider with gRPC→REST fallback."""
        try:
            # Prefer transport override via env
            import os
            preferred_transport = os.getenv("GOOGLE_TTS_TRANSPORT", "grpc").lower()

            if preferred_transport == "rest":
                self._client = texttospeech.TextToSpeechClient(transport="rest")
            else:
                try:
                    # Try gRPC first
                    self._client = texttospeech.TextToSpeechClient()
                except Exception as grpc_err:
                    print(f"⚠️  TTS gRPC init failed ({grpc_err}); falling back to REST transport")
                    self._client = texttospeech.TextToSpeechClient(transport="rest")
            
            # Pre-warm the connection with a minimal request
            await self._prewarm_client()
            return True
        except Exception as e:
            print(f"Failed to initialize Google TTS provider: {e}")
            return False
    
    async def _prewarm_client(self):
        """Pre-warm the TTS client to reduce first-call latency."""
        try:
            test_input = texttospeech.SynthesisInput(text=".")
            test_voice = texttospeech.VoiceSelectionParams(
                name="en-US-Neural2-C", 
                language_code="en-US"
            )
            test_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._client.synthesize_speech,
                texttospeech.SynthesizeSpeechRequest(
                    input=test_input, 
                    voice=test_voice, 
                    audio_config=test_config
                )
            )
        except Exception:
            pass  # Ignore pre-warming errors
    
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """Synthesize speech from text."""
        if not self._client:
            raise RuntimeError("Google TTS provider not initialized")
        
        # Use provided parameters or defaults
        voice_name = voice or self.voice_name
        speed_rate = speed or self.speed
        pitch_semitones = pitch or self.pitch
        
        # Determine if text is SSML
        is_ssml = text.lstrip().startswith("<speak>")
        
        # Sanitize plain text to avoid reading ASCII markup/punctuation literally
        if not is_ssml:
            text = self._sanitize_text_for_tts(text)
        
        # Create synthesis input
        synthesis_input = (
            texttospeech.SynthesisInput(ssml=text)
            if is_ssml
            else texttospeech.SynthesisInput(text=text)
        )
        
        # Extract language code from voice name
        language_code = self._extract_language_code(voice_name)
        
        # Create voice selection
        voice_params = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language_code
        )
        
        # Determine if we need post-processing
        is_hd = "HD" in voice_name.upper()
        needs_post_processing = (is_hd and abs(speed_rate - 1.0) >= 1e-6) or (abs(pitch_semitones) >= 1e-6)
        
        # Create audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=self.encoding_map.get(self.audio_encoding, texttospeech.AudioEncoding.MP3),
            speaking_rate=(1.0 if needs_post_processing else speed_rate),
        )
        
        # Synthesize speech (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                self._client.synthesize_speech,
                texttospeech.SynthesizeSpeechRequest(
                    input=synthesis_input,
                    voice=voice_params,
                    audio_config=audio_config
                )
            )
        except Exception as e:
            # Retry once using REST transport if gRPC was in use
            try:
                # If current client is not REST, try REST fallback
                client_is_rest = getattr(getattr(self._client, '_transport', None), '__class__', None)
                # Blindly create a REST client for fallback
                rest_client = texttospeech.TextToSpeechClient(transport="rest")
                response = await loop.run_in_executor(
                    None,
                    rest_client.synthesize_speech,
                    texttospeech.SynthesizeSpeechRequest(
                        input=synthesis_input,
                        voice=voice_params,
                        audio_config=audio_config
                    )
                )
                # Swap client for subsequent calls
                self._client = rest_client
            except Exception:
                raise
        
        audio_data = response.audio_content
        
        # Post-process if needed
        if needs_post_processing:
            audio_data = await self._post_process_audio(
                audio_data, 
                speed_rate, 
                pitch_semitones
            )
        
        # Create audio output
        return AudioOutput(
            audio_data=audio_data,
            format=AudioFormat.MP3 if self.audio_encoding == 'MP3' else AudioFormat.WAV,
            sample_rate=24000,  # Google TTS default
            voice=voice_name,
            language=language_code,
            metadata={
                'speed': speed_rate,
                'pitch': pitch_semitones,
                'is_ssml': is_ssml
            }
        )

    def _sanitize_text_for_tts(self, raw: str) -> str:
        """Remove or normalize ASCII markup so it isn't read literally.

        - Remove asterisks used for bullets/markdown emphasis
        - Strip backticks and code fences
        - Replace colons used as punctuation with a short pause (comma)
          but keep time-like patterns (e.g., 12:30)
        - Collapse excessive punctuation/whitespace
        """
        try:
            text = raw
            # Remove code fences/backticks
            text = text.replace("```", " ").replace("`", "")
            # Remove asterisks
            text = text.replace("*", "")
            # Replace punctuation colons not between digits with a comma pause
            text = re.sub(r"(?<!\\d):(?!\\d)", ", ", text)
            # Normalize multiple commas/spaces
            text = re.sub(r",\s*,+", ", ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception:
            return raw
    
    def _extract_language_code(self, voice_name: str) -> str:
        """Extract language code from voice name."""
        import re
        match = re.match(r"^([a-zA-Z]{2,3}-[A-Za-z]{2})(?:-.+)?$", voice_name)
        return match.group(1) if match else self.language_code
    
    async def _post_process_audio(self, audio_data: bytes, speed: float, pitch: float) -> bytes:
        """Post-process audio with ffmpeg for speed and pitch adjustment."""
        try:
            # Use RAM disk if available
            tmp_dir = "/tmp" if Path("/tmp").exists() else tempfile.gettempdir()
            
            with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
                tmp_in = Path(td) / "input.mp3"
                tmp_out = Path(td) / "output.mp3"
                
                # Write input audio
                tmp_in.write_bytes(audio_data)
                
                # Build filter chain
                filter_chain = self._build_ffmpeg_filter(speed, pitch)
                
                # Run ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-threads", "0",
                    "-i", str(tmp_in),
                    "-filter:a", filter_chain,
                    "-c:a", "mp3", "-b:a", "128k",
                    "-preset", "ultrafast",
                    str(tmp_out)
                ]
                
                # Run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                )
                
                # Read processed audio
                return tmp_out.read_bytes()
                
        except (FileNotFoundError, subprocess.CalledProcessError):
            # ffmpeg not available or failed, return original
            return audio_data
    
    def _build_ffmpeg_filter(self, speed: float, pitch: float) -> str:
        """Build ffmpeg filter chain for speed and pitch adjustment."""
        eps = 1e-6
        
        if abs(pitch) < eps:
            # Speed only
            return self._build_atempo_chain(max(speed, eps))
        
        # Combined pitch and speed
        pitch_factor = 2.0 ** (pitch / 12.0)
        combined_tempo = max(speed / pitch_factor, eps)
        tempo_chain = self._build_atempo_chain(combined_tempo)
        
        return f"aresample=48000,asetrate=48000*{pitch_factor:.8f},{tempo_chain},aresample=48000"
    
    def _build_atempo_chain(self, speed: float) -> str:
        """Build atempo filter chain for speed adjustment."""
        if speed <= 0:
            speed = 1.0
        
        parts = []
        remaining = speed
        
        # Decompose into factors within [0.5, 2.0]
        while remaining > 2.0:
            parts.append("2.0")
            remaining /= 2.0
        while remaining < 0.5:
            parts.append("0.5")
            remaining /= 0.5
        parts.append(f"{remaining:.3f}")
        
        return ",".join(f"atempo={p}" for p in parts)
    
    def play_audio(self, audio: AudioOutput) -> None:
        """Play synthesized audio (synchronous version)."""
        asyncio.run(self.play_audio_async(audio))
    
    async def play_audio_async(self, audio: AudioOutput) -> None:
        """Play synthesized audio (async, interruptible) using macOS afplay."""
        tmp_path = None
        self._is_playing = True
        self._stop_playback = False
        try:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio.audio_data)
                tmp_path = tmp.name
            # Use macOS afplay (non-blocking, interruptible)
            await self._play_with_afplay(tmp_path)
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            self._is_playing = False
    
    def stop_audio(self) -> None:
        """Stop audio playback immediately."""
        self._stop_playback = True
        self._is_playing = False  # Mark as not playing immediately
        
        # Stop afplay if running
        try:
            process = self._afplay_process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except Exception:
                    process.kill()
                print("🛑 Stopped afplay audio")
        except Exception as e:
            print(f"⚠️  Error stopping afplay: {e}")
        finally:
            self._afplay_process = None
    
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
        # Ensure any afplay process is terminated
        try:
            process = self._afplay_process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except Exception:
                    process.kill()
        except Exception:
            pass
        finally:
            self._afplay_process = None
        
        self._client = None

    async def _play_with_afplay(self, file_path: str) -> None:
        """Play audio using macOS 'afplay' with interruption support."""
        try:
            # Start process
            self._afplay_process = subprocess.Popen(["afplay", file_path])
        except FileNotFoundError:
            raise RuntimeError("'afplay' not found. Ensure afplay is available on macOS.")
        except Exception as e:
            raise RuntimeError(f"Failed to start 'afplay': {e}")

        try:
            # Poll until finished or interrupted (robust to concurrent stop that clears the process)
            while not self._stop_playback:
                process = self._afplay_process
                if process is None or process.poll() is not None:
                    break
                await asyncio.sleep(0.05)

            # If interrupted and process still running, terminate it
            if self._stop_playback:
                process = self._afplay_process
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except Exception:
                        process.kill()
        finally:
            self._afplay_process = None
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': False,
            'batch': True,
            'voices': [
                'en-US-Chirp3-HD-Sadachbia',
                'en-US-Neural2-A', 'en-US-Neural2-C', 'en-US-Neural2-D',
                'en-US-Neural2-E', 'en-US-Neural2-F', 'en-US-Neural2-G',
                'en-US-Neural2-H', 'en-US-Neural2-I', 'en-US-Neural2-J',
            ],
            'languages': ['en-US', 'en-GB', 'en-AU', 'en-IN'],
            'audio_formats': ['mp3', 'wav', 'ogg'],
            'speed_range': (0.25, 4.0),
            'pitch_range': (-20, 20),
            'features': ['ssml_support', 'pitch_control', 'speed_control']
        }