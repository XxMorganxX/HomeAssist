"""
Process-isolated termination phrase detection provider.

Runs OpenWakeWord in a separate process to detect conversation-ending phrases
like "over out" in parallel with other operations (TTS, transcription, etc.).

Key differences from wake word detection:
- Runs in parallel with other audio operations (doesn't use audio_manager)
- Single model focus (termination phrase only)
- Designed to interrupt ongoing operations
- Lighter weight - optimized for running alongside main flow
"""

import asyncio
import multiprocessing as mp
import queue
import os
from typing import AsyncIterator, Dict, Any, Optional
from pathlib import Path

try:
    from ...interfaces.termination import TerminationInterface
    from ...models.data_models import TerminationEvent
    from ...utils.logging_config import vprint, eprint
except ImportError:
    from assistant_framework.interfaces.termination import TerminationInterface
    from assistant_framework.models.data_models import TerminationEvent
    from assistant_framework.utils.logging_config import vprint, eprint


def _termination_worker(
    config: Dict[str, Any], 
    event_queue: mp.Queue, 
    stop_event: mp.Event,
    pause_event: mp.Event,
    resume_event: mp.Event,
    current_state_value: mp.Value,  # Shared value for current state
    verbose: bool = False
):
    """
    Worker function running in separate process for termination phrase detection.
    
    This runs in complete isolation from the main process, allowing reliable
    detection even when other audio operations are happening.
    
    Args:
        config: Termination detection configuration
        event_queue: Queue for sending detection events to main process
        stop_event: Event to signal worker to fully stop (terminate)
        pause_event: Event to signal worker to pause detection
        resume_event: Event to signal worker to resume detection
        current_state_value: Shared value containing current AudioState (as int)
        verbose: Whether to print detailed status messages
    """
    # Local print helper
    def tprint(msg):
        if verbose:
            print(f"[Termination] {msg}")
    
    import numpy as np
    import sounddevice as sd
    from openwakeword.model import Model
    import time
    
    # State name mapping for metadata
    STATE_NAMES = {
        0: "IDLE",
        1: "WAKE_WORD_LISTENING", 
        2: "TRANSCRIBING",
        3: "PROCESSING_RESPONSE",
        4: "SYNTHESIZING",
        5: "TRANSITIONING",
        6: "ERROR"
    }
    
    # Extract config
    sample_rate = config['sample_rate']
    chunk = config['chunk']
    threshold = config['threshold']
    cooldown_seconds = config['cooldown_seconds']
    model_path = config['model_path']
    input_device_index = config.get('input_device_index')
    latency = config.get('latency', 'high')
    inference_framework = config.get('inference_framework', 'onnx')
    melspec_model = config.get('melspec_model')
    embed_model = config.get('embed_model')
    suppress_overflow = config.get('suppress_overflow_warnings', True)  # Default suppress for parallel
    
    stream = None
    
    try:
        # Initialize model in clean process
        tprint(f"Initializing termination model (PID: {os.getpid()})...")
        tprint(f"Model: {model_path}")
        
        model_kwargs = {'wakeword_models': [model_path], 'inference_framework': inference_framework}
        if melspec_model:
            model_kwargs['melspec_model_path'] = melspec_model
        if embed_model:
            model_kwargs['embedding_model_path'] = embed_model
        
        model = Model(**model_kwargs)
        tprint("Model initialized")
        
        # Detection state
        last_detection_time = 0.0
        is_paused = True  # Start paused, wait for explicit resume
        
        # Signal ready
        event_queue.put({'status': 'ready'})
        
        # Main loop - supports pause/resume
        while not stop_event.is_set():
            # Check for pause signal
            if pause_event.is_set():
                if not is_paused:
                    is_paused = True
                    pause_event.clear()
                    
                    # Close audio stream during pause
                    if stream:
                        try:
                            stream.stop()
                            stream.close()
                            tprint("Paused - audio stream closed")
                        except Exception as e:
                            print(f"[Termination] Error closing stream: {e}")
                        stream = None
                    
                    # Clear prediction buffer
                    for model_name in model.prediction_buffer:
                        model.prediction_buffer[model_name].clear()
                    
                    event_queue.put({'status': 'paused'})
                else:
                    pause_event.clear()
                continue
            
            # Check for resume signal
            if resume_event.is_set():
                if is_paused:
                    is_paused = False
                    resume_event.clear()
                    
                    # Open audio stream
                    try:
                        tprint(f"Resuming - opening audio stream")
                        stream = sd.InputStream(
                            samplerate=sample_rate,
                            channels=1,
                            dtype='int16',
                            blocksize=chunk,
                            device=input_device_index,
                            latency=latency
                        )
                        stream.start()
                        
                        # Clear prediction buffer for fresh start
                        for model_name in model.prediction_buffer:
                            model.prediction_buffer[model_name].clear()
                        
                        # Brief warmup
                        for _ in range(3):
                            try:
                                stream.read(chunk)
                            except Exception:
                                pass
                        
                        tprint("Audio stream started")
                        event_queue.put({'status': 'resumed'})
                    except Exception as e:
                        print(f"[Termination] Failed to open audio: {e}")
                        event_queue.put({'error': f"Audio open failed: {e}"})
                        is_paused = True
                else:
                    resume_event.clear()
                continue
            
            # If paused, wait
            if is_paused:
                time.sleep(0.05)
                continue
            
            # Open stream if needed (shouldn't happen, but safety)
            if stream is None:
                try:
                    stream = sd.InputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        blocksize=chunk,
                        device=input_device_index,
                        latency=latency
                    )
                    stream.start()
                    tprint("Audio stream opened (fallback)")
                except Exception as e:
                    print(f"[Termination] Failed to open stream: {e}")
                    event_queue.put({'error': str(e)})
                    break
            
            # Detection loop
            try:
                audio_data, overflowed = stream.read(chunk)
                if overflowed and verbose and not suppress_overflow:
                    tprint("Audio overflow")
                
                audio = audio_data.flatten().astype(np.int16)
                model.predict(audio)
                
                current_time = time.time()
                
                # Check for detections
                for model_name, scores in model.prediction_buffer.items():
                    if not scores:
                        continue
                    
                    score = scores[-1]
                    
                    if verbose and score > 0.1:
                        tprint(f"{model_name}: {score:.3f}")
                    
                    # Check threshold and cooldown
                    if score > threshold:
                        if current_time - last_detection_time > cooldown_seconds:
                            # Get current state for metadata
                            try:
                                state_int = current_state_value.value
                                state_name = STATE_NAMES.get(state_int, f"UNKNOWN_{state_int}")
                            except Exception:
                                state_name = "UNKNOWN"
                            
                            # Detection!
                            event_data = {
                                'phrase_name': model_name,
                                'score': float(score),
                                'timestamp': current_time,
                                'interrupted_state': state_name
                            }
                            event_queue.put(event_data)
                            last_detection_time = current_time
                            print(f"🛑 [Termination] Phrase detected: {model_name} ({score:.3f}) - interrupting {state_name}")
                
            except Exception as e:
                if not stop_event.is_set() and not pause_event.is_set():
                    print(f"[Termination] Detection error: {e}")
        
        # Cleanup
        if stream:
            stream.stop()
            stream.close()
        tprint("Shutdown complete")
        
    except Exception as e:
        print(f"[Termination] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        event_queue.put({'error': str(e)})


class IsolatedTerminationProvider(TerminationInterface):
    """
    Termination phrase detection provider running in isolated process.
    
    This provider detects conversation-ending phrases like "over out" in parallel
    with other operations, enabling instant conversation termination.
    
    Key features:
    - Process isolation prevents crashes affecting main app
    - Runs alongside TTS, transcription, and response generation
    - Lightweight single-model detection
    - Warm mode keeps process alive for fast resume
    
    Usage:
        provider = IsolatedTerminationProvider(config)
        await provider.initialize()
        
        # Start detection when conversation begins
        await provider.resume_detection()
        
        async for event in provider.start_detection():
            if event:
                # Termination phrase detected - cancel current operation
                break
        
        # Pause when conversation ends
        await provider.pause_detection()
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Verbose logging
        try:
            from assistant_framework.config import VERBOSE_LOGGING
            self._verbose = VERBOSE_LOGGING and config.get('verbose', False)
        except ImportError:
            self._verbose = config.get('verbose', False)
        
        # Process management
        self._process: Optional[mp.Process] = None
        self._event_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._pause_event: Optional[mp.Event] = None
        self._resume_event: Optional[mp.Event] = None
        self._current_state: Optional[mp.Value] = None  # Shared int for current state
        self._is_listening = False
        self._is_paused = True
        self._is_initialized = False
        
        # Prepare model path
        model_dir = Path(config.get('model_dir', './audio_data/wake_word_models'))
        model_name = config.get('model_name', 'over_out')
        
        # Find model file
        model_path = model_dir / f"{model_name}.onnx"
        inference_framework = 'onnx'
        
        if not model_path.exists():
            model_path = model_dir / f"{model_name}.tflite"
            inference_framework = 'tflite'
            
        if not model_path.exists():
            # Model doesn't exist yet - that's okay, we'll note it
            print(f"⚠️  Termination model not found: {model_name}")
            print(f"   Expected at: {model_dir / model_name}.onnx")
            print(f"   Termination detection will be disabled until model is trained")
            self._model_available = False
            return
        
        self._model_available = True
        
        # Helper model paths
        melspec_model = model_dir / f"melspectrogram.{inference_framework}"
        embed_model = model_dir / f"embedding_model.{inference_framework}"
        
        self.config['model_path'] = str(model_path)
        self.config['inference_framework'] = inference_framework
        self.config['melspec_model'] = str(melspec_model) if melspec_model.exists() else None
        self.config['embed_model'] = str(embed_model) if embed_model.exists() else None
        
        vprint(f"📋 Termination model configured: {model_name}")
    
    async def initialize(self) -> bool:
        """Initialize the provider and start the worker process."""
        if not self._model_available:
            vprint("⚠️  Termination detection disabled (model not found)")
            return False
        
        if self._is_initialized:
            return True
        
        try:
            # Create process communication objects
            self._event_queue = mp.Queue()
            self._stop_event = mp.Event()
            self._pause_event = mp.Event()
            self._resume_event = mp.Event()
            self._current_state = mp.Value('i', 0)  # Integer for state enum
            
            # Start worker process
            vprint("🚀 Starting termination detection process...")
            self._process = mp.Process(
                target=_termination_worker,
                args=(
                    self.config,
                    self._event_queue,
                    self._stop_event,
                    self._pause_event,
                    self._resume_event,
                    self._current_state,
                    self._verbose
                ),
                daemon=False
            )
            self._process.start()
            
            # Wait for ready signal
            try:
                loop = asyncio.get_event_loop()
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, self._event_queue.get, True, 5.0),
                    timeout=6.0
                )
                if event.get('status') == 'ready':
                    vprint(f"✅ Termination detection ready (PID: {self._process.pid})")
                    self._is_initialized = True
                    self._is_paused = True  # Starts paused
                    return True
                elif 'error' in event:
                    raise RuntimeError(event['error'])
            except (asyncio.TimeoutError, queue.Empty):
                print("❌ Termination process startup timed out")
                await self._terminate_process()
                return False
            
        except Exception as e:
            print(f"❌ Failed to initialize termination detection: {e}")
            return False
        
        return False
    
    async def start_detection(self) -> AsyncIterator[TerminationEvent]:
        """
        Start yielding termination events.
        
        This should be called in a task that runs alongside other operations.
        """
        if not self._is_initialized or not self._model_available:
            return
        
        self._is_listening = True
        
        try:
            while self._is_listening:
                try:
                    # Get event with timeout
                    event_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._event_queue.get,
                        True,
                        0.1
                    )
                    
                    # Check for errors
                    if 'error' in event_data:
                        print(f"❌ Termination worker error: {event_data['error']}")
                        break
                    
                    # Skip status messages
                    if 'status' in event_data:
                        continue
                    
                    # Yield termination event
                    yield TerminationEvent(**event_data)
                    
                except queue.Empty:
                    # Check if process is alive
                    if self._process and not self._process.is_alive():
                        print("⚠️  Termination process died unexpectedly")
                        break
                    continue
                    
        finally:
            self._is_listening = False
    
    async def stop_detection(self) -> None:
        """Stop listening for events (but keep process warm)."""
        self._is_listening = False
    
    async def pause_detection(self) -> None:
        """Pause detection (keeps process alive for fast resume)."""
        if not self._is_initialized or self._is_paused:
            return
        
        vprint("⏸️  Pausing termination detection...")
        self._pause_event.set()
        self._is_paused = True  # Set immediately - don't wait for confirmation
        
        # Quick confirmation check (non-blocking, fire-and-forget style)
        # The worker will pause regardless, this just clears the queue
        try:
            loop = asyncio.get_event_loop()
            try:
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, self._event_queue.get, True, 0.05),
                    timeout=0.15  # Very short timeout - don't block restart
                )
                if event.get('status') == 'paused':
                    vprint("✅ Termination detection paused")
            except queue.Empty:
                pass  # That's fine, worker will pause anyway
        except asyncio.TimeoutError:
            pass  # Don't log - this is expected for fast restart
    
    async def resume_detection(self) -> None:
        """Resume detection after pause."""
        if not self._is_initialized:
            return
        
        if not self._is_paused:
            return  # Already active
        
        vprint("▶️  Resuming termination detection...")
        self._resume_event.set()
        self._is_paused = False  # Set immediately for faster flow
        
        # Quick confirmation check - don't block conversation startup
        try:
            loop = asyncio.get_event_loop()
            try:
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, self._event_queue.get, True, 0.1),
                    timeout=0.5  # Reduced from 2s - audio opens fast
                )
                if event.get('status') == 'resumed':
                    vprint("✅ Termination detection resumed")
                elif 'error' in event:
                    print(f"⚠️  Termination resume error: {event['error']}")
            except queue.Empty:
                pass  # Worker will resume anyway
        except asyncio.TimeoutError:
            pass  # Don't block - detection will work even without confirmation
        
        self._is_paused = False
    
    def set_current_state(self, state: str) -> None:
        """Update the current conversation state for event metadata."""
        if self._current_state is None:
            return
        
        # Map state name to int
        state_map = {
            "IDLE": 0,
            "WAKE_WORD_LISTENING": 1,
            "TRANSCRIBING": 2,
            "PROCESSING_RESPONSE": 3,
            "SYNTHESIZING": 4,
            "TRANSITIONING": 5,
            "ERROR": 6
        }
        
        state_int = state_map.get(state, 0)
        self._current_state.value = state_int
    
    @property
    def is_listening(self) -> bool:
        """Check if actively listening for events."""
        return self._is_listening
    
    @property
    def is_paused(self) -> bool:
        """Check if detection is paused."""
        return self._is_paused
    
    @property
    def is_available(self) -> bool:
        """Check if termination detection is available (model exists)."""
        return self._model_available
    
    async def _terminate_process(self) -> None:
        """Terminate the worker process."""
        if not self._process:
            return
        
        self._stop_event.set()
        
        loop = asyncio.get_event_loop()
        
        # Try graceful shutdown
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._process.join, 0.5),
                timeout=0.8
            )
        except asyncio.TimeoutError:
            pass
        
        # Force terminate if needed
        if self._process.is_alive():
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._process.join, 0.3),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                self._process.kill()
        
        self._process = None
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        vprint("🧹 Cleaning up termination detection...")
        
        self._is_listening = False
        
        await self._terminate_process()
        
        # Cleanup communication objects
        self._event_queue = None
        self._stop_event = None
        self._pause_event = None
        self._resume_event = None
        self._current_state = None
        self._is_initialized = False
        
        vprint("✅ Termination detection cleaned up")
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'streaming': True,
            'audio_formats': ['pcm16'],
            'sample_rates': [16000],
            'threshold_configurable': True,
            'parallel_execution': True,
            'warm_mode': True,
            'process_isolated': True,
            'model': self.config.get('model_name', 'over_out'),
            'available': self._model_available
        }

