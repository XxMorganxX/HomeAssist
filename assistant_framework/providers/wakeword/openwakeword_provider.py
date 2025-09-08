"""
OpenWakeWord-based wake word detection provider.
"""

import asyncio
import os
import time
from enum import Enum
from typing import AsyncIterator, Dict, Any, Optional

import numpy as np
import pyaudio

try:
    # Try relative imports first (when used as package)
    from ...interfaces.wake_word import WakeWordInterface
    from ...models.data_models import WakeWordEvent
    from ...utils.audio_manager import get_audio_manager
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.wake_word import WakeWordInterface
    from models.data_models import WakeWordEvent
    from utils.audio_manager import get_audio_manager


class _ListenerState(Enum):
    LISTENING = 1
    COOLDOWN = 2


class OpenWakeWordProvider(WakeWordInterface):
    """Wake word provider using openwakeword."""

    def __init__(self, config: Dict[str, Any]):
        # External import inside to avoid hard dependency when not used
        from openwakeword.model import Model  # type: ignore
        from openwakeword.utils import download_models  # type: ignore

        # Configuration with sensible defaults
        self.sample_rate: int = int(config.get('sample_rate', 16000))
        self.chunk: int = int(config.get('chunk', 1280))
        self.threshold: float = float(config.get('threshold', 0.3))
        self.cooldown_seconds: float = float(config.get('cooldown_seconds', 2.0))
        self.min_playback_interval: float = float(config.get('min_playback_interval', 0.5))
        self.verbose: bool = bool(config.get('verbose', False))

        # Model paths
        self.model_dir: str = str(config.get('model_dir', './audio_data/wake_word_models'))
        self.model_name: str = str(config.get('model_name', 'hey_monkey'))
        self.model_path_tflite: str = os.path.join(self.model_dir, f"{self.model_name}.tflite")
        self.model_path_onnx: str = os.path.join(self.model_dir, f"{self.model_name}.onnx")

        # Optional input device index (for stability on macOS if default device changes)
        self.input_device_index: Optional[int] = None
        try:
            if 'input_device_index' in config and config['input_device_index'] is not None:
                self.input_device_index = int(config['input_device_index'])
        except Exception:
            self.input_device_index = None

        # Ensure model exists
        if not os.path.exists(self.model_path_tflite) and not os.path.exists(self.model_path_onnx):
            os.makedirs(self.model_dir, exist_ok=True)
            download_models(target_directory=self.model_dir)

        # Initialize model (prefer ONNX if present). Explicitly pass feature model paths
        # so openwakeword doesn't rely on package-installed resources.
        melspec_onnx = os.path.join(self.model_dir, "melspectrogram.onnx")
        embed_onnx = os.path.join(self.model_dir, "embedding_model.onnx")
        melspec_tflite = os.path.join(self.model_dir, "melspectrogram.tflite")
        embed_tflite = os.path.join(self.model_dir, "embedding_model.tflite")

        if os.path.exists(self.model_path_onnx):
            self._oww_model = Model(
                wakeword_models=[self.model_path_onnx],
                inference_framework='onnx',
                melspec_model_path=melspec_onnx,
                embedding_model_path=embed_onnx,
            )
            if self.verbose:
                print(f"Using ONNX model: {self.model_path_onnx}")
        else:
            self._oww_model = Model(
                wakeword_models=[self.model_path_tflite],
                inference_framework='tflite',
                melspec_model_path=melspec_tflite,
                embedding_model_path=embed_tflite,
            )
            if self.verbose:
                print(f"Using TFLite model: {self.model_path_tflite}")

        # Internal state
        self._n_models: int = len(self._oww_model.models.keys())
        self._last_activation_time: Dict[str, float] = {name: 0.0 for name in self._oww_model.models.keys()}
        self._previous_scores: Dict[str, float] = {name: 0.0 for name in self._oww_model.models.keys()}
        self._state: Dict[str, _ListenerState] = {name: _ListenerState.LISTENING for name in self._oww_model.models.keys()}
        self._last_playback_time: float = 0.0

        # Audio IO
        self._audio: Optional[pyaudio.PyAudio] = None
        self._mic_stream = None
        self.audio_manager = get_audio_manager()

        # Async coordination
        self._is_listening: bool = False
        self._stop_event = asyncio.Event()

    async def initialize(self) -> bool:
        try:
            # Audio will be acquired when needed
            return True
        except Exception as e:
            print(f"Failed to initialize OpenWakeWordProvider: {e}")
            return False

    async def start_detection(self) -> AsyncIterator[WakeWordEvent]:
        if self._is_listening:
            return

        # Retry device acquisition once if first attempt fails due to transient device issues
        for attempt in range(2):
            # Acquire audio resources (soft handoff)
            self._audio = self.audio_manager.acquire_audio("wakeword", force_cleanup=False)
            if self._audio:
                break
            # Backoff before retry
            await asyncio.sleep(0.3)
        if not self._audio:
            raise RuntimeError("Failed to acquire audio resources for wake word detection")

        # Open microphone stream (deferred start to avoid CoreAudio race conditions)
        print("🎤 Opening wake word audio stream")
        open_kwargs = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': self.sample_rate,
            'input': True,
            'frames_per_buffer': self.chunk,
            'start': False,
        }
        if self.input_device_index is not None:
            open_kwargs['input_device_index'] = self.input_device_index

        self._mic_stream = self._audio.open(**open_kwargs)
        # Let CoreAudio settle briefly before starting the stream
        await asyncio.sleep(0.25)
        self._mic_stream.start_stream()
        print("✅ Wake word audio stream started")

        self._is_listening = True
        self._stop_event.clear()
        
        # Reset wake word model state
        self._oww_model.reset()
        
        # Reset internal state tracking
        for model_name in self._oww_model.models.keys():
            self._state[model_name] = _ListenerState.LISTENING
            self._previous_scores[model_name] = 0.0
            self._last_activation_time[model_name] = 0.0
        
        # Clear any buffered audio by reading and discarding a few chunks
        try:
            for i in range(10):  # Clear more chunks and add delays
                if self._mic_stream and not self._stop_event.is_set():
                    self._mic_stream.read(self.chunk, exception_on_overflow=False)
                    # Small delay between reads to prevent overwhelming the system
                    if i % 2 == 0:
                        await asyncio.sleep(0.01)
        except Exception as e:
            print(f"⚠️  Buffer clearing warning: {e}")  # More visible warning
            # Additional delay if buffer clearing fails
            await asyncio.sleep(0.1)

        try:
            # Non-blocking async loop yielding events
            while self._is_listening and not self._stop_event.is_set():
                try:
                    audio = np.frombuffer(
                        self._mic_stream.read(self.chunk, exception_on_overflow=False),
                        dtype=np.int16,
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"Audio read error: {e}")
                    await asyncio.sleep(0.01)
                    continue

                self._oww_model.predict(audio)

                current_time = time.time()
                for model_name in self._oww_model.prediction_buffer.keys():
                    scores = list(self._oww_model.prediction_buffer[model_name])
                    if not scores:
                        continue
                    curr_score = scores[-1]
                    prev_score = self._previous_scores.get(model_name, 0.0)

                    if self.verbose and curr_score > 0.05:
                        print(f"[{model_name}] State={self._state[model_name].name} Score={curr_score:.3f} Prev={prev_score:.3f}")

                    if self._state[model_name] == _ListenerState.LISTENING:
                        time_since_last_activation = current_time - self._last_activation_time.get(model_name, 0.0)
                        time_since_last_playback = current_time - self._last_playback_time
                        rising_edge = prev_score <= self.threshold and curr_score > self.threshold
                        if rising_edge and time_since_last_activation > self.cooldown_seconds and time_since_last_playback > self.min_playback_interval:
                            self._state[model_name] = _ListenerState.COOLDOWN
                            self._last_activation_time[model_name] = current_time
                            event = WakeWordEvent(model_name=model_name, score=float(curr_score))
                            yield event
                    else:
                        time_since_activation = current_time - self._last_activation_time.get(model_name, 0.0)
                        if time_since_activation > self.cooldown_seconds:
                            self._state[model_name] = _ListenerState.LISTENING

                    self._previous_scores[model_name] = float(curr_score)

                # Yield control to event loop
                await asyncio.sleep(0)
        finally:
            self._is_listening = False

    async def stop_detection(self) -> None:
        if not self._is_listening:
            return
        self._stop_event.set()
        self._is_listening = False
        
        # Close microphone stream
        try:
            if self._mic_stream:
                print("🎤 Stopping wake word audio stream")
                if hasattr(self._mic_stream, 'is_active') and self._mic_stream.is_active():
                    self._mic_stream.stop_stream()
                self._mic_stream.close()
                self._mic_stream = None
                print("✅ Wake word audio stream stopped")
        except Exception as e:
            print(f"⚠️  Error stopping wake word stream: {e}")
            self._mic_stream = None
        
        # Release audio resources (don't force cleanup here to avoid double cleanup)
        try:
            self.audio_manager.release_audio("wakeword", force_cleanup=False)
            self._audio = None
        except Exception as e:
            print(f"⚠️  Error releasing wake word audio: {e}")
            self._audio = None

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    async def cleanup(self) -> None:
        await self.stop_detection()
        
        # Audio cleanup is handled by the audio manager
        self._audio = None
        self._mic_stream = None

    @property
    def capabilities(self) -> dict:
        caps = super().capabilities
        caps.update({
            'model_dir': self.model_dir,
            'models_loaded': self._n_models,
            'threshold': self.threshold,
        })
        return caps


