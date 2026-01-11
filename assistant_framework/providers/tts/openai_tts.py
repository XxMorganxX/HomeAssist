"""
OpenAI Text-to-Speech provider with streaming support for low-latency playback.

Uses OpenAI's streaming API to begin audio playback as soon as the first chunks
arrive, significantly reducing perceived latency.
"""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import os

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai package is required. Install with: pip install openai")

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


class OpenAITTSProvider(TextToSpeechInterface):
    """
    OpenAI TTS provider with streaming support.
    
    Supports tts-1, tts-1-hd, and gpt-4o-mini-tts models with true streaming
    for minimal perceived latency. Audio playback begins as soon as the first
    chunks arrive from the API.
    """
    
    # Available voices: alloy, echo, fable, onyx, nova, shimmer
    AVAILABLE_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    
    # Available models
    AVAILABLE_MODELS = ['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts']
    
    # Available output formats
    AVAILABLE_FORMATS = ['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']
    
    # Streaming chunk size (bytes) - balance between latency and overhead
    STREAM_CHUNK_SIZE = 4096
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key (optional, can use env var)
                - model: Model to use ('tts-1', 'tts-1-hd', or 'gpt-4o-mini-tts')
                - voice: Voice name (default: 'alloy')
                - speed: Speed modifier 0.25-4.0 (default: 1.0)
                - response_format: Output format (default: 'mp3')
                - stream_chunk_size: Bytes per chunk for streaming (default: 4096)
        """
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass in config.")
        
        self.model = config.get('model', 'gpt-4o-mini-tts')
        if self.model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {self.model}. Available: {self.AVAILABLE_MODELS}")
        
        self.voice = config.get('voice', 'alloy')
        if self.voice not in self.AVAILABLE_VOICES:
            raise ValueError(f"Invalid voice: {self.voice}. Available: {self.AVAILABLE_VOICES}")
        
        self.speed = config.get('speed', 1.0)
        if not (0.25 <= self.speed <= 4.0):
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        self.response_format = config.get('response_format', 'mp3')
        if self.response_format not in self.AVAILABLE_FORMATS:
            raise ValueError(f"Invalid format: {self.response_format}. Available: {self.AVAILABLE_FORMATS}")
        
        self.stream_chunk_size = config.get('stream_chunk_size', self.STREAM_CHUNK_SIZE)
        
        # TTS client (lazy initialization)
        self._client = None
        self._is_playing = False
        self._stop_playback = False
        self._playback_process = None
        self._playback_lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI TTS provider."""
        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Failed to initialize OpenAI TTS provider: {e}")
            return False
    
    def _get_audio_format(self) -> AudioFormat:
        """Get the AudioFormat enum for the current response format."""
        format_map = {
            'mp3': AudioFormat.MP3,
            'opus': AudioFormat.OGG,
            'aac': AudioFormat.MP3,
            'flac': AudioFormat.FLAC,
            'wav': AudioFormat.WAV,
            'pcm': AudioFormat.PCM16
        }
        return format_map.get(self.response_format, AudioFormat.MP3)
    
    def _get_file_suffix(self) -> str:
        """Get file suffix for the current response format."""
        suffix_map = {
            'mp3': '.mp3',
            'opus': '.opus',
            'aac': '.aac',
            'flac': '.flac',
            'wav': '.wav',
            'pcm': '.pcm'
        }
        return suffix_map.get(self.response_format, '.mp3')
    
    async def synthesize(self, 
                        text: str,
                        voice: Optional[str] = None,
                        speed: Optional[float] = None,
                        pitch: Optional[float] = None) -> AudioOutput:
        """
        Synthesize speech from text (non-streaming, returns complete audio).
        
        For lower latency, use synthesize_and_play_streaming() instead.
        
        Note: OpenAI TTS does not support pitch adjustment.
        """
        if not self._client:
            raise RuntimeError("OpenAI TTS provider not initialized")
        
        voice_name = voice or self.voice
        speed_rate = speed or self.speed
        
        if voice_name not in self.AVAILABLE_VOICES:
            voice_name = self.voice
        
        speed_rate = max(0.25, min(4.0, speed_rate))
        
        try:
            response = await self._client.audio.speech.create(
                model=self.model,
                voice=voice_name,
                input=text,
                speed=speed_rate,
                response_format=self.response_format
            )
            
            audio_data = response.content
            
            return AudioOutput(
                audio_data=audio_data,
                format=self._get_audio_format(),
                sample_rate=24000,
                voice=voice_name,
                language="en-US",
                metadata={
                    'model': self.model,
                    'speed': speed_rate,
                    'provider': 'openai',
                    'format': self.response_format,
                    'streaming': False
                }
            )
            
        except Exception as e:
            raise Exception(f"OpenAI TTS synthesis failed: {e}")
    
    async def stream_synthesize(self,
                               text: str,
                               voice: Optional[str] = None,
                               speed: Optional[float] = None,
                               pitch: Optional[float] = None) -> AsyncIterator[AudioOutput]:
        """
        Stream synthesized speech as chunks arrive from OpenAI.
        
        Yields AudioOutput objects containing audio chunks as they become available.
        This enables playback to begin before the full response is received.
        
        Args:
            text: Text to synthesize
            voice: Optional voice identifier  
            speed: Optional speed modifier
            pitch: Optional pitch modifier (ignored - OpenAI doesn't support pitch)
            
        Yields:
            AudioOutput: Audio chunks as they become available
        """
        if not self._client:
            raise RuntimeError("OpenAI TTS provider not initialized")
        
        voice_name = voice or self.voice
        speed_rate = speed or self.speed
        
        if voice_name not in self.AVAILABLE_VOICES:
            voice_name = self.voice
        
        speed_rate = max(0.25, min(4.0, speed_rate))
        
        try:
            # Use streaming response
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_name,
                input=text,
                speed=speed_rate,
                response_format=self.response_format
            ) as response:
                chunk_index = 0
                async for chunk in response.iter_bytes(self.stream_chunk_size):
                    if self._stop_playback:
                        break
                    
                    yield AudioOutput(
                        audio_data=chunk,
                        format=self._get_audio_format(),
                        sample_rate=24000,
                        voice=voice_name,
                        language="en-US",
                        metadata={
                            'model': self.model,
                            'speed': speed_rate,
                            'provider': 'openai',
                            'format': self.response_format,
                            'streaming': True,
                            'chunk_index': chunk_index
                        }
                    )
                    chunk_index += 1
                    
        except Exception as e:
            raise Exception(f"OpenAI TTS streaming synthesis failed: {e}")
    
    async def synthesize_and_play_streaming(self,
                                           text: str,
                                           voice: Optional[str] = None,
                                           speed: Optional[float] = None) -> None:
        """
        Synthesize and play audio with minimal latency using true streaming.
        
        This method streams audio directly from OpenAI to the audio player,
        beginning playback as soon as the first chunks arrive. This provides
        the lowest perceived latency.
        
        Uses ffplay for streaming playback (supports stdin input).
        Falls back to buffered playback if ffplay is not available.
        
        Args:
            text: Text to synthesize
            voice: Optional voice identifier
            speed: Optional speed modifier
        """
        async with self._playback_lock:
            self._is_playing = True
            self._stop_playback = False
            
            voice_name = voice or self.voice
            speed_rate = speed or self.speed
            
            if voice_name not in self.AVAILABLE_VOICES:
                voice_name = self.voice
            
            speed_rate = max(0.25, min(4.0, speed_rate))
            
            start_time = time.time()
            first_chunk_time = None
            
            try:
                # Try streaming playback with ffplay first
                if await self._try_streaming_playback(text, voice_name, speed_rate):
                    return
                
                # Fallback: buffer and play with afplay
                await self._fallback_buffered_playback(text, voice_name, speed_rate)
                
            except Exception as e:
                print(f"❌ Streaming TTS error: {e}")
            finally:
                self._is_playing = False
                self._playback_process = None
    
    async def _try_streaming_playback(self, text: str, voice: str, speed: float) -> bool:
        """
        Attempt streaming playback using ffplay.
        
        Returns True if successful, False if ffplay is not available.
        """
        try:
            # Check if ffplay is available
            check = await asyncio.create_subprocess_exec(
                "which", "ffplay",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await check.wait()
            
            if check.returncode != 0:
                return False
            
            # Determine format-specific ffplay args
            format_args = self._get_ffplay_format_args()
            
            # Start ffplay process reading from stdin
            self._playback_process = await asyncio.create_subprocess_exec(
                "ffplay",
                "-nodisp",          # No video display
                "-autoexit",        # Exit when done
                "-loglevel", "quiet",  # Suppress output
                *format_args,
                "-i", "pipe:0",     # Read from stdin
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            first_chunk_time = None
            start_time = time.time()
            
            # Stream audio chunks directly to ffplay
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice,
                input=text,
                speed=speed,
                response_format=self.response_format
            ) as response:
                async for chunk in response.iter_bytes(self.stream_chunk_size):
                    if self._stop_playback:
                        break
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency_ms = (first_chunk_time - start_time) * 1000
                        print(f"🎵 TTS streaming started (latency: {latency_ms:.0f}ms)")
                    
                    if self._playback_process and self._playback_process.stdin:
                        try:
                            self._playback_process.stdin.write(chunk)
                            await self._playback_process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            break
            
            # Close stdin to signal end of stream
            if self._playback_process and self._playback_process.stdin:
                self._playback_process.stdin.close()
                await self._playback_process.stdin.wait_closed()
            
            # Wait for playback to complete (with timeout)
            if self._playback_process and not self._stop_playback:
                try:
                    await asyncio.wait_for(self._playback_process.wait(), timeout=60)
                except asyncio.TimeoutError:
                    self._playback_process.terminate()
            
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"⚠️  Streaming playback error: {e}")
            return False
    
    def _get_ffplay_format_args(self) -> list:
        """Get format-specific arguments for ffplay."""
        # Most formats are auto-detected, but some need hints
        if self.response_format == 'pcm':
            # PCM needs explicit format specification
            return ["-f", "s16le", "-ar", "24000", "-ac", "1"]
        return []
    
    async def _fallback_buffered_playback(self, text: str, voice: str, speed: float) -> None:
        """
        Fallback playback method that buffers audio then plays with afplay.
        
        Used when ffplay is not available.
        """
        print("⚠️  ffplay not found, using buffered playback (higher latency)")
        
        # Collect all chunks
        chunks = []
        async for audio_chunk in self.stream_synthesize(text, voice=voice, speed=speed):
            if self._stop_playback:
                return
            chunks.append(audio_chunk.audio_data)
        
        if not chunks:
            return
        
        # Combine and play
        full_audio = b''.join(chunks)
        audio_output = AudioOutput(
            audio_data=full_audio,
            format=self._get_audio_format(),
            sample_rate=24000,
            voice=voice,
            language="en-US",
            metadata={'streaming': False, 'fallback': True}
        )
        
        await self.play_audio_async(audio_output)
    
    def play_audio(self, audio: AudioOutput) -> None:
        """Play synthesized audio (synchronous version)."""
        asyncio.run(self.play_audio_async(audio))
    
    async def play_audio_async(self, audio: AudioOutput) -> None:
        """Play synthesized audio (async, interruptible) using macOS afplay."""
        tmp_path = None
        self._is_playing = True
        self._stop_playback = False
        
        try:
            suffix = self._get_file_suffix()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio.audio_data)
                tmp_path = tmp.name
            
            # Play with afplay
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
        self._is_playing = False
        
        # Stop playback process (ffplay or afplay)
        try:
            process = self._playback_process
            if process is not None:
                # For asyncio subprocess
                if hasattr(process, 'terminate'):
                    process.terminate()
                # For regular subprocess
                elif hasattr(process, 'poll') and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except Exception:
                        process.kill()
                print("🛑 Stopped audio playback")
        except Exception as e:
            print(f"⚠️  Error stopping playback: {e}")
        finally:
            self._playback_process = None
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    async def save_audio(self, audio: AudioOutput, path: Path) -> None:
        """Save audio to file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.get_event_loop().run_in_executor(
                None,
                path.write_bytes,
                audio.audio_data
            )
        except Exception as e:
            raise Exception(f"Failed to save audio: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_audio()
        
        if self._client:
            await self._client.close()
            self._client = None

    async def _play_with_afplay(self, file_path: str) -> None:
        """Play audio using macOS 'afplay' with interruption support."""
        try:
            self._playback_process = subprocess.Popen(["afplay", file_path])
        except FileNotFoundError:
            raise RuntimeError("'afplay' not found. Ensure afplay is available on macOS.")
        except Exception as e:
            raise RuntimeError(f"Failed to start 'afplay': {e}")

        try:
            while not self._stop_playback:
                process = self._playback_process
                if process is None or process.poll() is not None:
                    break
                await asyncio.sleep(0.05)

            if self._stop_playback:
                process = self._playback_process
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except Exception:
                        process.kill()
        finally:
            self._playback_process = None
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,  # Now supports streaming!
            'batch': True,
            'voices': self.AVAILABLE_VOICES,
            'languages': ['auto-detect'],
            'audio_formats': self.AVAILABLE_FORMATS,
            'speed_range': (0.25, 4.0),
            'pitch_range': (0, 0),
            'models': self.AVAILABLE_MODELS,
            'features': [
                'auto_language_detection',
                'high_quality',
                'low_latency',
                'streaming_playback',
                'interruptible'
            ]
        }
