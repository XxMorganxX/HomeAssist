"""
Process-isolated wake word detection provider.
Eliminates model corruption and segfault issues through complete process isolation.

Supports "warm mode" - keeping the subprocess alive between conversations for faster restart.
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
    from ...utils.audio.audio_manager import get_audio_manager
    from ...utils.logging.logging_config import vprint, eprint
except ImportError:
    from assistant_framework.interfaces.wake_word import WakeWordInterface
    from assistant_framework.models.data_models import WakeWordEvent
    from assistant_framework.utils.audio.audio_manager import get_audio_manager
    from assistant_framework.utils.logging.logging_config import vprint, eprint


def _wake_word_worker(
    config: Dict[str, Any], 
    event_queue: mp.Queue, 
    stop_event: mp.Event,
    pause_event: mp.Event,
    resume_event: mp.Event,
    verbose: bool = True
):
    """
    Worker function running in separate process.
    
    This runs in complete isolation from the main process, preventing
    any model corruption or memory issues from affecting the main application.
    
    Supports pause/resume for fast turnaround between conversations.
    
    Args:
        config: Wake word configuration
        event_queue: Queue for sending detection events to main process
        stop_event: Event to signal worker to fully stop (terminate)
        pause_event: Event to signal worker to pause detection (keep process alive)
        resume_event: Event to signal worker to resume detection
        verbose: Whether to print detailed status messages
    """
    # Local verbose print for worker process
    def wprint(msg):
        if verbose:
            print(msg)
    import numpy as np
    import sounddevice as sd
    from openwakeword.model import Model
    import time
    
    # Import tones for audio feedback
    try:
        from assistant_framework.utils.audio.tones import beep_wake_model_ready
    except ImportError:
        beep_wake_model_ready = lambda: None  # Fallback if import fails
    
    # Extract config
    sample_rate = config['sample_rate']
    chunk = config['chunk']
    threshold = config['threshold']
    cooldown_seconds = config['cooldown_seconds']
    # Support multiple models
    model_paths = config.get('model_paths', [config['model_path']])
    input_device_index = config.get('input_device_index')
    debug_verbose = config.get('verbose', False)  # Separate from logging verbose
    suppress_overflow = config.get('suppress_overflow_warnings', False)
    latency = config.get('latency', 'high')  # 'high' for Bluetooth devices
    inference_framework = config.get('inference_framework', 'onnx')
    melspec_model = config.get('melspec_model')
    embed_model = config.get('embed_model')
    
    stream = None
    
    try:
        # Initialize model(s) in clean process
        wprint(f"🔄 [Worker] Initializing {len(model_paths)} wake word model(s) (PID: {os.getpid()})...")
        for mp_path in model_paths:
            wprint(f"   - {mp_path}")
        
        model_kwargs = {'wakeword_models': model_paths, 'inference_framework': inference_framework}
        if melspec_model:
            model_kwargs['melspec_model_path'] = melspec_model
        if embed_model:
            model_kwargs['embedding_model_path'] = embed_model
        
        model = Model(**model_kwargs)
        wprint(f"✅ [Worker] {len(model_paths)} model(s) initialized")
        
        # Detection state
        last_detection = {}
        is_paused = False
        
        # Main loop - supports pause/resume
        while not stop_event.is_set():
            # Check for pause signal
            if pause_event.is_set() and not is_paused:
                is_paused = True
                pause_event.clear()
                
                # Stop and close audio stream during pause
                if stream:
                    try:
                        stream.stop()
                        stream.close()
                        wprint("⏸️  [Worker] Paused - audio stream closed")
                    except Exception as e:
                        print(f"⚠️  [Worker] Error closing stream on pause: {e}")  # Always show warnings
                    stream = None
                
                # Clear prediction buffer on pause to prevent stale data on resume
                for model_name in model.prediction_buffer:
                    model.prediction_buffer[model_name].clear()
                
                # Signal that we're paused
                event_queue.put({'status': 'paused'})
                continue
            
            # Check for resume signal
            if resume_event.is_set() and is_paused:
                is_paused = False
                resume_event.clear()
                
                # Reopen audio stream
                try:
                    bt_mode = " [Bluetooth mode]" if suppress_overflow else ""
                    wprint(f"▶️  [Worker] Resuming - reopening audio stream{bt_mode}")
                    stream = sd.InputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        blocksize=chunk,
                        device=input_device_index,
                        latency=latency
                    )
                    stream.start()
                    
                    # CRITICAL: Clear prediction buffer to prevent spurious detections
                    # The model accumulates scores over time, stale scores can trigger false positives
                    for model_name in model.prediction_buffer:
                        model.prediction_buffer[model_name].clear()
                    wprint(f"🧹 [Worker] Prediction buffer cleared")
                    
                    # Reset last detection times (stale cooldowns)
                    last_detection.clear()
                    
                    # Brief warmup: discard initial audio frames (may contain noise/transients)
                    warmup_frames = 3  # ~60ms at 16kHz with 320 chunk
                    for _ in range(warmup_frames):
                        try:
                            stream.read(chunk)
                        except Exception:
                            pass
                    wprint(f"✅ [Worker] Audio stream restarted (warmup complete)")
                    
                    beep_wake_model_ready()  # 🔔 Wake word model ready sound
                    event_queue.put({'status': 'resumed'})
                except Exception as e:
                    print(f"❌ [Worker] Failed to resume audio stream: {e}")  # Always show errors
                    event_queue.put({'error': f"Resume failed: {e}"})
                    break
                continue
            
            # If paused, just wait for resume or stop
            if is_paused:
                # Check events every 50ms while paused
                time.sleep(0.05)
                continue
            
            # Open audio stream if not already open (initial start)
            if stream is None:
                try:
                    bt_mode = " [Bluetooth mode]" if suppress_overflow else ""
                    wprint(f"🎤 [Worker] Opening audio stream{bt_mode}")
                    wprint(f"   Device: {input_device_index or 'default'}, Blocksize: {chunk}, Latency: '{latency}'")
                    stream = sd.InputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        blocksize=chunk,
                        device=input_device_index,
                        latency=latency
                    )
                    stream.start()
                    
                    # Brief warmup: discard initial audio frames (may contain noise/transients)
                    warmup_frames = 3  # ~60ms at 16kHz with 320 chunk
                    for _ in range(warmup_frames):
                        try:
                            stream.read(chunk)
                        except Exception:
                            pass
                    
                    wprint(f"✅ [Worker] Audio stream started")
                    beep_wake_model_ready()  # 🔔 Wake word model ready sound
                except Exception as e:
                    print(f"❌ [Worker] Failed to open audio stream: {e}")  # Always show errors
                    event_queue.put({'error': str(e)})
                    break
            
            # Normal detection loop
            try:
                # Read audio
                audio_data, overflowed = stream.read(chunk)
                if overflowed and debug_verbose and not suppress_overflow:
                    wprint("⚠️  [Worker] Audio overflow")
                
                audio = audio_data.flatten().astype(np.int16)
                
                # Predict
                model.predict(audio)
                
                current_time = time.time()
                
                # Check for detections
                for model_name, scores in model.prediction_buffer.items():
                    if not scores:
                        continue
                    
                    score = scores[-1]
                    
                    if debug_verbose and score > 0.05:
                        wprint(f"[Worker] {model_name}: {score:.3f}")
                    
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
                            print(f"🔔 [Worker] Wake word detected: {model_name} ({score:.3f})")  # Always show
            
            except Exception as e:
                if not stop_event.is_set() and not pause_event.is_set():
                    print(f"❌ [Worker] Detection error: {e}")  # Always show errors
        
        # Cleanup
        if stream:
            stream.stop()
            stream.close()
        wprint("✅ [Worker] Shutdown complete")
        
    except Exception as e:
        print(f"❌ [Worker] Fatal error: {e}")  # Always show errors
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
    - WARM MODE: Keep process alive between conversations for ~2s faster restart
    
    This addresses the primary reliability issue with wake word detection
    by running OpenWakeWord in a completely separate process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_manager = get_audio_manager()
        
        # Warm mode configuration - keep process alive between conversations
        self._warm_mode = config.get('warm_mode', True)
        
        # Verbose logging - get from config or environment
        try:
            from assistant_framework.config import VERBOSE_LOGGING
            self._verbose = VERBOSE_LOGGING
        except ImportError:
            self._verbose = True
        
        # Process management
        self._process: Optional[mp.Process] = None
        self._event_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._pause_event: Optional[mp.Event] = None
        self._resume_event: Optional[mp.Event] = None
        self._is_listening = False
        self._is_paused = False  # Track if process is paused (warm mode)
        
        # Prepare model paths - support multiple models
        model_dir = Path(config.get('model_dir', './audio_data/wake_word_models'))
        
        # Support both single model_name and multiple model_names
        model_names = config.get('model_names', [])
        if not model_names:
            model_names = [config.get('model_name', 'alexa_v0.1')]
        
        # Find all model paths
        model_paths = []
        inference_framework = 'onnx'  # Default
        
        for model_name in model_names:
            # Try .onnx first, then .tflite
            model_path = model_dir / f"{model_name}.onnx"
            if model_path.exists():
                model_paths.append(str(model_path))
                inference_framework = 'onnx'
            else:
                model_path = model_dir / f"{model_name}.tflite"
                if model_path.exists():
                    model_paths.append(str(model_path))
                    inference_framework = 'tflite'
                else:
                    print(f"⚠️  Wake word model not found: {model_name}")  # Always show warnings
        
        if not model_paths:
            raise FileNotFoundError(f"No wake word models found in {model_dir}")
        
        # Helper model paths
        melspec_model = model_dir / f"melspectrogram.{inference_framework}"
        embed_model = model_dir / f"embedding_model.{inference_framework}"
        
        self.config['model_paths'] = model_paths  # List of model paths
        self.config['model_path'] = model_paths[0]  # Backward compat
        self.config['inference_framework'] = inference_framework
        self.config['melspec_model'] = str(melspec_model) if melspec_model.exists() else None
        self.config['embed_model'] = str(embed_model) if embed_model.exists() else None
        
        warm_status = " [warm mode]" if self._warm_mode else ""
        vprint(f"📋 Wake word models configured{warm_status}: {[Path(p).stem for p in model_paths]}")
    
    async def initialize(self) -> bool:
        """Initialize provider (one-time setup)."""
        try:
            # Create model directory if needed
            model_dir = Path(self.config['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize wake word provider: {e}")  # Always show errors
            return False
    
    def _ensure_process_communication(self):
        """Ensure process communication objects exist."""
        if self._event_queue is None:
            self._event_queue = mp.Queue()
        if self._stop_event is None:
            self._stop_event = mp.Event()
        if self._pause_event is None:
            self._pause_event = mp.Event()
        if self._resume_event is None:
            self._resume_event = mp.Event()
    
    async def _start_process(self):
        """Start the worker process."""
        self._ensure_process_communication()
        
        vprint("🚀 Starting wake word detection process...")
        self._process = mp.Process(
            target=_wake_word_worker,
            args=(self.config, self._event_queue, self._stop_event, 
                  self._pause_event, self._resume_event, self._verbose),
            daemon=False  # Explicit shutdown
        )
        self._process.start()
        vprint(f"✅ Wake word process started (PID: {self._process.pid})")
    
    async def start_detection(self) -> AsyncIterator[WakeWordEvent]:
        """Start wake word detection in isolated process."""
        if self._is_listening:
            return
        
        # Acquire audio - don't force cleanup to avoid race conditions with transcription
        acquired = self.audio_manager.acquire_audio("wakeword", force_cleanup=False)
        if not acquired:
            # Wait and retry once in case cleanup is still in progress
            vprint("⏳ Audio busy, waiting for cleanup to complete...")
            await asyncio.sleep(0.1)  # Reduced from 0.3s
            acquired = self.audio_manager.acquire_audio("wakeword", force_cleanup=False)
            if not acquired:
                raise RuntimeError("Failed to acquire audio for wake word")
        
        # Check if we can resume from paused state (warm mode)
        if self._warm_mode and self._is_paused and self._process and self._process.is_alive():
            vprint("⚡ Resuming wake word detection (warm mode)...")
            self._resume_event.set()
            self._is_paused = False
            self._is_listening = True
            
            # Wait for resume confirmation
            try:
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        event_data = await asyncio.wait_for(
                            loop.run_in_executor(None, self._event_queue.get, True, 0.1),
                            timeout=0.5
                        )
                        if event_data.get('status') == 'resumed':
                            vprint("✅ Wake word detection resumed")
                            break
                        elif 'error' in event_data:
                            raise RuntimeError(event_data['error'])
                    except queue.Empty:
                        if not self._process.is_alive():
                            raise RuntimeError("Process died during resume")
                        continue
            except asyncio.TimeoutError:
                print("⚠️  Resume confirmation timed out, continuing anyway")  # Always show warnings
        else:
            # Cold start - need to start new process
            self._ensure_process_communication()
            
            # Clear any stale events
            self._stop_event.clear()
            self._pause_event.clear()
            self._resume_event.clear()
            
            await self._start_process()
            self._is_listening = True
            self._is_paused = False
        
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
                        print(f"❌ Worker error: {event_data['error']}")  # Always show errors
                        break
                    
                    # Skip status messages (paused/resumed)
                    if 'status' in event_data:
                        continue
                    
                    # Yield event
                    yield WakeWordEvent(**event_data)
                    
                except queue.Empty:
                    # Timeout, check if process is still alive
                    if self._process and not self._process.is_alive():
                        print("⚠️  Wake word process terminated unexpectedly")  # Always show warnings
                        break
                    continue
                    
        finally:
            self._is_listening = False
    
    async def pause_detection(self) -> None:
        """
        Pause wake word detection without terminating process (warm mode).
        
        This releases audio resources but keeps the process and models loaded
        for fast restart.
        """
        if not self._is_listening and not (self._process and self._process.is_alive()):
            return
        
        if self._is_paused:
            return  # Already paused
        
        vprint("⏸️  Pausing wake word detection (warm mode)...")
        self._is_listening = False
        
        # Signal process to pause
        if self._pause_event:
            self._pause_event.set()
        
        # Wait for pause confirmation
        if self._process and self._process.is_alive():
            try:
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        event_data = await asyncio.wait_for(
                            loop.run_in_executor(None, self._event_queue.get, True, 0.1),
                            timeout=0.5
                        )
                        if event_data.get('status') == 'paused':
                            vprint("✅ Wake word detection paused")
                            break
                    except queue.Empty:
                        if not self._process.is_alive():
                            break
                        continue
            except asyncio.TimeoutError:
                print("⚠️  Pause confirmation timed out")  # Always show warnings
        
        self._is_paused = True
        
        # Release audio
        self.audio_manager.release_audio("wakeword", force_cleanup=False)
    
    async def stop_detection(self) -> None:
        """
        Stop wake word detection.
        
        In warm mode, this pauses instead of terminating.
        Use cleanup() for full termination.
        """
        if not self._is_listening and not self._is_paused:
            # Nothing to stop
            if not (self._process and self._process.is_alive()):
                return
        
        # In warm mode, pause instead of full stop
        if self._warm_mode and self._process and self._process.is_alive():
            await self.pause_detection()
            return
        
        # Full stop (non-warm mode or explicit cleanup)
        await self._full_stop()
    
    async def _full_stop(self) -> None:
        """Fully stop and terminate the wake word process."""
        vprint("🛑 Stopping wake word detection...")
        self._is_listening = False
        self._is_paused = False
        
        # Signal process to stop
        if self._stop_event:
            self._stop_event.set()
        
        # Wait for process to terminate (run blocking join() in executor)
        if self._process:
            loop = asyncio.get_event_loop()
            
            # Try graceful shutdown first (short timeout)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._process.join, 0.5),
                    timeout=0.8
                )
            except asyncio.TimeoutError:
                pass
            
            # If still alive, terminate
            if self._process.is_alive():
                print("⚠️  Process didn't stop gracefully, terminating...")  # Always show warnings
                self._process.terminate()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self._process.join, 0.3),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    pass
            
            # If STILL alive, kill it
            if self._process.is_alive():
                print("⚠️  Process still alive, killing...")  # Always show warnings
                self._process.kill()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self._process.join, 0.2),
                        timeout=0.3
                    )
                except asyncio.TimeoutError:
                    print("⚠️  Process force kill timed out (this is unusual)")  # Always show warnings
            
            vprint(f"✅ Wake word process stopped")
            self._process = None
        
        # Cleanup communication objects
        self._event_queue = None
        self._stop_event = None
        self._pause_event = None
        self._resume_event = None
        
        # Release audio
        self.audio_manager.release_audio("wakeword", force_cleanup=False)
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening
    
    @property
    def is_warm(self) -> bool:
        """Check if process is warm (paused but ready to resume)."""
        return self._is_paused and self._process is not None and self._process.is_alive()
    
    async def cleanup(self) -> None:
        """Cleanup provider - always fully terminates process."""
        # Force full stop even in warm mode
        self._warm_mode = False  # Temporarily disable to ensure full stop
        await self._full_stop()
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'threshold_configurable': True,
            'process_isolated': True,  # Key feature!
            'warm_mode': self._warm_mode,  # Fast restart support
            'model': self.config.get('model_name', 'unknown')
        }


