"""
Process-isolated wake word detection provider.
Eliminates model corruption and segfault issues through complete process isolation.
"""

import asyncio
import multiprocessing as mp
import queue
import os
from typing import AsyncIterator, Dict, Any, Optional
from pathlib import Path

try:
    from ...interfaces.wake_word import WakeWordInterface
    from ...models.data_models import WakeWordEvent
    from ...utils.audio_manager import get_audio_manager
except ImportError:
    from assistant_framework.interfaces.wake_word import WakeWordInterface
    from assistant_framework.models.data_models import WakeWordEvent
    from assistant_framework.utils.audio_manager import get_audio_manager


def _wake_word_worker(config: Dict[str, Any], event_queue: mp.Queue, stop_event: mp.Event):
    """
    Worker function running in separate process.
    
    This runs in complete isolation from the main process, preventing
    any model corruption or memory issues from affecting the main application.
    
    Args:
        config: Wake word configuration
        event_queue: Queue for sending detection events to main process
        stop_event: Event to signal worker to stop
    """
    import numpy as np
    import sounddevice as sd
    from openwakeword.model import Model
    import time
    
    # Import tones for audio feedback
    try:
        from assistant_framework.utils.tones import beep_wake_model_ready
    except ImportError:
        beep_wake_model_ready = lambda: None  # Fallback if import fails
    
    # Extract config
    sample_rate = config['sample_rate']
    chunk = config['chunk']
    threshold = config['threshold']
    cooldown_seconds = config['cooldown_seconds']
    model_path = config['model_path']
    input_device_index = config.get('input_device_index')
    verbose = config.get('verbose', False)
    suppress_overflow = config.get('suppress_overflow_warnings', False)
    latency = config.get('latency', 'high')  # 'high' for Bluetooth devices
    inference_framework = config.get('inference_framework', 'onnx')
    melspec_model = config.get('melspec_model')
    embed_model = config.get('embed_model')
    
    try:
        # Initialize model in clean process
        print(f"🔄 [Worker] Initializing wake word model (PID: {os.getpid()})...")
        
        model_kwargs = {'wakeword_models': [model_path], 'inference_framework': inference_framework}
        if melspec_model:
            model_kwargs['melspec_model_path'] = melspec_model
        if embed_model:
            model_kwargs['embedding_model_path'] = embed_model
        
        model = Model(**model_kwargs)
        print(f"✅ [Worker] Model initialized")
        
        # Open audio stream
        # Use configurable latency - 'high' for Bluetooth devices to handle bursty audio
        bt_mode = " [Bluetooth mode]" if suppress_overflow else ""
        print(f"🎤 [Worker] Opening audio stream{bt_mode}")
        print(f"   Device: {input_device_index or 'default'}, Blocksize: {chunk}, Latency: '{latency}'")
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=chunk,
            device=input_device_index,
            latency=latency
        )
        stream.start()
        print(f"✅ [Worker] Audio stream started")
        beep_wake_model_ready()  # 🔔 Wake word model ready sound
        
        # Detection state
        last_detection = {}
        
        # Main detection loop
        while not stop_event.is_set():
            try:
                # Read audio
                audio_data, overflowed = stream.read(chunk)
                if overflowed and verbose and not suppress_overflow:
                    print("⚠️  [Worker] Audio overflow")
                
                audio = audio_data.flatten().astype(np.int16)
                
                # Predict
                model.predict(audio)
                
                current_time = time.time()
                
                # Check for detections
                for model_name, scores in model.prediction_buffer.items():
                    if not scores:
                        continue
                    
                    score = scores[-1]
                    
                    if verbose and score > 0.05:
                        print(f"[Worker] {model_name}: {score:.3f}")
                    
                    # Check threshold and cooldown
                    if score > threshold:
                        last_time = last_detection.get(model_name, 0.0)
                        if current_time - last_time > cooldown_seconds:
                            # Detection!
                            event_data = {
                                'model_name': model_name,
                                'score': float(score),
                                'timestamp': current_time
                            }
                            event_queue.put(event_data)
                            last_detection[model_name] = current_time
                            print(f"🔔 [Worker] Wake word detected: {model_name} ({score:.3f})")
            
            except Exception as e:
                if not stop_event.is_set():
                    print(f"❌ [Worker] Detection error: {e}")
        
        # Cleanup
        stream.stop()
        stream.close()
        print("✅ [Worker] Shutdown complete")
        
    except Exception as e:
        print(f"❌ [Worker] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        event_queue.put({'error': str(e)})


class IsolatedOpenWakeWordProvider(WakeWordInterface):
    """
    Wake word provider running in isolated process.
    
    Benefits:
    - Complete isolation prevents model corruption
    - Process crash doesn't affect main application
    - Clean state on each start
    - No model recreation workarounds needed
    
    This addresses the primary reliability issue with wake word detection
    by running OpenWakeWord in a completely separate process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_manager = get_audio_manager()
        
        # Process management
        self._process: Optional[mp.Process] = None
        self._event_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._is_listening = False
        
        # Prepare model paths
        model_dir = Path(config.get('model_dir', './audio_data/wake_word_models'))
        model_name = config.get('model_name', 'alexa_v0.1')
        
        # Try .onnx first, then .tflite
        model_path = model_dir / f"{model_name}.onnx"
        inference_framework = 'onnx'
        melspec_model = model_dir / "melspectrogram.onnx"
        embed_model = model_dir / "embedding_model.onnx"
        
        if not model_path.exists():
            model_path = model_dir / f"{model_name}.tflite"
            inference_framework = 'tflite'
            melspec_model = model_dir / "melspectrogram.tflite"
            embed_model = model_dir / "embedding_model.tflite"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Wake word model not found: {model_name}")
        
        self.config['model_path'] = str(model_path)
        self.config['inference_framework'] = inference_framework
        self.config['melspec_model'] = str(melspec_model) if melspec_model.exists() else None
        self.config['embed_model'] = str(embed_model) if embed_model.exists() else None
    
    async def initialize(self) -> bool:
        """Initialize provider (one-time setup)."""
        try:
            # Create model directory if needed
            model_dir = Path(self.config['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to initialize wake word provider: {e}")
            return False
    
    async def start_detection(self) -> AsyncIterator[WakeWordEvent]:
        """Start wake word detection in isolated process."""
        if self._is_listening:
            return
        
        # Acquire audio - don't force cleanup to avoid race conditions with transcription
        # The state machine should have already cleaned up the previous component
        acquired = self.audio_manager.acquire_audio("wakeword", force_cleanup=False)
        if not acquired:
            # Wait and retry once in case cleanup is still in progress (reduced from 1.0s)
            print("⏳ Audio busy, waiting for cleanup to complete...")
            await asyncio.sleep(0.3)
            acquired = self.audio_manager.acquire_audio("wakeword", force_cleanup=False)
            if not acquired:
                raise RuntimeError("Failed to acquire audio for wake word")
        
        # Create process communication
        self._event_queue = mp.Queue()
        self._stop_event = mp.Event()
        
        # Start worker process
        print("🚀 Starting wake word detection process...")
        self._process = mp.Process(
            target=_wake_word_worker,
            args=(self.config, self._event_queue, self._stop_event),
            daemon=False  # Explicit shutdown
        )
        self._process.start()
        print(f"✅ Wake word process started (PID: {self._process.pid})")
        
        self._is_listening = True
        
        try:
            # Yield events from queue
            while self._is_listening:
                try:
                    # Get event with timeout to allow checking stop condition
                    event_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._event_queue.get,
                        True,  # block
                        0.1    # timeout
                    )
                    
                    # Check for error
                    if 'error' in event_data:
                        print(f"❌ Worker error: {event_data['error']}")
                        break
                    
                    # Yield event
                    yield WakeWordEvent(**event_data)
                    
                except queue.Empty:
                    # Timeout, check if process is still alive
                    if self._process and not self._process.is_alive():
                        print("⚠️  Wake word process terminated unexpectedly")
                        break
                    continue
                    
        finally:
            self._is_listening = False
    
    async def stop_detection(self) -> None:
        """Stop wake word detection."""
        if not self._is_listening:
            return
        
        print("🛑 Stopping wake word detection...")
        self._is_listening = False
        
        # Signal process to stop
        if self._stop_event:
            self._stop_event.set()
        
        # Wait for process to terminate (run blocking join() in executor)
        if self._process:
            loop = asyncio.get_event_loop()
            
            # Try graceful shutdown first (short timeout)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._process.join, 1.0),
                    timeout=1.5
                )
            except asyncio.TimeoutError:
                pass
            
            # If still alive, terminate
            if self._process.is_alive():
                print("⚠️  Process didn't stop gracefully, terminating...")
                self._process.terminate()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self._process.join, 0.5),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    pass
            
            # If STILL alive, kill it
            if self._process.is_alive():
                print("⚠️  Process still alive, killing...")
                self._process.kill()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self._process.join, 0.5),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    print("⚠️  Process force kill timed out (this is unusual)")
            
            print(f"✅ Wake word process stopped")
            self._process = None
        
        # Cleanup
        self._event_queue = None
        self._stop_event = None
        
        # Release audio
        self.audio_manager.release_audio("wakeword", force_cleanup=False)
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening
    
    async def cleanup(self) -> None:
        """Cleanup provider."""
        await self.stop_detection()
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'threshold_configurable': True,
            'process_isolated': True,  # Key feature!
            'model': self.config.get('model_name', 'unknown')
        }



