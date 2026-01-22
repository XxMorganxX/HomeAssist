"""
Refactored orchestrator with state machine integration.
Simplified, reliable pipeline execution.
"""

import asyncio
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from .interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        WakeWordInterface,
        ContextInterface,
        TerminationInterface
    )
    from .utils.state_machine import AudioStateMachine, AudioState
    from .utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity
    from .utils.audio.barge_in import BargeInDetector, BargeInConfig, BargeInMode
    from .utils.audio.shared_audio_bus import SharedAudioBus, SharedAudioBusConfig
    from .utils.logging.conversation_recorder import ConversationRecorder
    from .utils.audio.tones import (
        beep_send_detected, beep_shutdown
    )
    # Note: beep_wake_detected, beep_listening_start, beep_ready_to_listen
    # are now handled automatically by state machine transition beeps
    from .utils.logging.console_logger import (
        log_boot, log_shutdown, log_conversation_end, log_wake_word,
        log_user_message, log_assistant_response, log_tool_call,
        log_termination_detected
    )
    from .providers.wakeword import IsolatedOpenWakeWordProvider
    from .providers.transcription import AssemblyAIAsyncProvider
    from .providers.context import UnifiedContextProvider
    from .providers.termination import IsolatedTerminationProvider
    from .utils.briefing.briefing_manager import BriefingManager
    from .config import get_active_preset, TERMINATION_DETECTION_CONFIG, ENABLE_TTS_ANNOUNCEMENTS
    from .models.data_models import TransitionReason, TransitionContext
    from .utils.audio.tts_announcements import announce_termination, announce_conversation_start, precache_announcements
except ImportError:
    from assistant_framework.interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        WakeWordInterface,
        ContextInterface,
        TerminationInterface
    )
    from assistant_framework.utils.state_machine import AudioStateMachine, AudioState
    from assistant_framework.utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity
    from assistant_framework.utils.audio.barge_in import BargeInDetector, BargeInConfig, BargeInMode
    from assistant_framework.utils.audio.shared_audio_bus import SharedAudioBus, SharedAudioBusConfig
    from assistant_framework.utils.logging.conversation_recorder import ConversationRecorder
    from assistant_framework.utils.audio.tones import (
        beep_send_detected, beep_shutdown
    )
    # Note: beep_wake_detected, beep_listening_start, beep_ready_to_listen
    # are now handled automatically by state machine transition beeps
    from assistant_framework.utils.logging.console_logger import (
        log_boot, log_shutdown, log_conversation_end, log_wake_word,
        log_user_message, log_assistant_response, log_tool_call,
        log_termination_detected
    )
    from assistant_framework.providers.wakeword import IsolatedOpenWakeWordProvider
    from assistant_framework.providers.transcription import AssemblyAIAsyncProvider
    from assistant_framework.providers.context import UnifiedContextProvider
    from assistant_framework.providers.termination import IsolatedTerminationProvider
    from assistant_framework.utils.briefing.briefing_manager import BriefingManager
    from assistant_framework.config import get_active_preset, TERMINATION_DETECTION_CONFIG, ENABLE_TTS_ANNOUNCEMENTS
    from assistant_framework.models.data_models import TransitionReason, TransitionContext
    from assistant_framework.utils.audio.tts_announcements import announce_termination, announce_conversation_start, precache_announcements


class RefactoredOrchestrator:
    """
    Refactored orchestrator with:
    - State machine for lifecycle management
    - Structured error handling
    - Simplified pipeline execution
    - Providers (process-isolated, fully async)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Turnaround/latency configuration (loaded early for state machine)
        turnaround = config.get('turnaround', {})
        self._barge_in_resume_delay = turnaround.get('barge_in_resume_delay', 0.1)
        self._transcription_stop_delay = turnaround.get('transcription_stop_delay', 0.15)
        self._state_transition_delay = turnaround.get('state_transition_delay', 0.25)
        self._streaming_tts_enabled = turnaround.get('streaming_tts_enabled', False)
        # Fast reboot optimizations
        self._wake_word_warm_mode = turnaround.get('wake_word_warm_mode', True)
        self._post_conversation_delay = turnaround.get('post_conversation_delay', 0.0)
        self._wake_word_stop_delay = turnaround.get('wake_word_stop_delay', 0.0)
        
        # Core infrastructure (respect current preset: dev/prod/test)
        try:
            preset = get_active_preset()
        except Exception:
            preset = "prod"
        self.state_machine = AudioStateMachine(mode=preset, transition_delay=self._state_transition_delay)
        self.error_handler = ErrorHandler()
        
        # Provider instances (will be lazily created)
        self._transcription: Optional[TranscriptionInterface] = None
        self._response: Optional[ResponseInterface] = None
        self._tts: Optional[TextToSpeechInterface] = None
        self._wakeword: Optional[WakeWordInterface] = None
        self._termination: Optional[TerminationInterface] = None
        
        # Termination detection (parallel "over out" detection)
        self._termination_enabled = TERMINATION_DETECTION_CONFIG.get('enabled', True)
        self._termination_task: Optional[asyncio.Task] = None
        self._termination_detected = False  # Flag to signal conversation termination
        self._termination_poll_interval = TERMINATION_DETECTION_CONFIG.get('interrupt_poll_interval', 0.05)
        
        # Barge-in support (interrupt TTS with speech)
        self._barge_in_enabled = config.get('barge_in_enabled', True)
        self._barge_in_detector: Optional[BargeInDetector] = None
        self._barge_in_triggered = False  # Flag to signal TTS interruption
        self._barge_in_audio: Optional[bytes] = None  # Captured audio from barge-in
        
        # Shared audio bus (zero-latency transitions between transcription/barge-in)
        shared_bus_config = config.get('shared_audio_bus', {})
        self._shared_audio_bus_enabled = shared_bus_config.get('enabled', True)
        self._shared_audio_bus: Optional[SharedAudioBus] = None
        if self._shared_audio_bus_enabled:
            self._shared_audio_bus = SharedAudioBus(SharedAudioBusConfig(
                enabled=True,
                sample_rate=shared_bus_config.get('sample_rate', 16000),
                channels=shared_bus_config.get('channels', 1),
                chunk_size=shared_bus_config.get('chunk_size', 1024),
                buffer_seconds=shared_bus_config.get('buffer_seconds', 3.0),
                device_index=shared_bus_config.get('device_index'),
                latency=shared_bus_config.get('latency', 'high'),
                is_bluetooth=shared_bus_config.get('is_bluetooth', False),
            ))

        # Briefing announcements
        try:
            self._briefing_manager: Optional[BriefingManager] = BriefingManager()
        except Exception:
            self._briefing_manager = None
        
        # Early barge-in tracking (append to previous message if interrupted early)
        barge_in_config = config.get('barge_in', {})
        self._early_barge_in_threshold = float(barge_in_config.get('early_barge_in_threshold', 3.0))
        # Runtime barge-in tuning (these should come from config.py BARGE_IN_CONFIG)
        self._barge_in_energy_threshold = float(barge_in_config.get('energy_threshold', 0.02))
        self._barge_in_min_speech_duration = float(barge_in_config.get('min_speech_duration', 0.15))
        self._barge_in_cooldown_after_tts_start = float(barge_in_config.get('cooldown_after_tts_start', 0.5))
        self._barge_in_sample_rate = int(barge_in_config.get('sample_rate', 16000))
        self._barge_in_chunk_size = int(barge_in_config.get('chunk_size', 1600))
        # Map config naming to BargeInConfig naming
        self._barge_in_buffer_seconds = float(barge_in_config.get('pre_barge_in_buffer_duration', barge_in_config.get('buffer_seconds', 2.0)))
        self._barge_in_capture_after_trigger = float(barge_in_config.get('post_barge_in_capture_duration', barge_in_config.get('capture_after_trigger', 0.3)))
        # Device settings for barge-in (critical for Bluetooth/Meta glasses)
        self._barge_in_device_index = barge_in_config.get('device_index', None)
        self._barge_in_latency = barge_in_config.get('latency', 'high')
        self._barge_in_is_bluetooth = barge_in_config.get('is_bluetooth', False)
        # Speech consistency settings (require continuous speech, not intermittent)
        self._barge_in_require_consecutive = barge_in_config.get('require_consecutive_speech', True)
        self._barge_in_silence_reset_threshold = int(barge_in_config.get('silence_reset_threshold', 3))
        # Processing-phase barge-in settings
        self._barge_in_during_processing = barge_in_config.get('enable_during_processing', True)
        self._barge_in_processing_cooldown = float(barge_in_config.get('processing_cooldown', 0.1))
        # False barge-in recovery: resume TTS if user doesn't speak after interrupting
        self._barge_in_recovery_enabled = barge_in_config.get('recovery_enabled', True)
        self._barge_in_recovery_timeout = float(barge_in_config.get('recovery_timeout', 4.0))
        self._barge_in_recovery_grace_period = float(barge_in_config.get('recovery_grace_period', 1.0))
        self._interrupted_tts_response: Optional[str] = None  # Store interrupted response for recovery
        self._barge_in_recovery_mode = False  # Flag: currently in recovery mode (waiting to see if user speaks)
        # Conservative speech rate estimate for resume position
        # Using LOWER rate than typical TTS (which is ~150-180 wpm = 2.5-3.0 wps)
        # This ensures we UNDERESTIMATE words spoken, so we repeat a few words
        # rather than skip words (better to have slight redundancy than miss content)
        self._tts_words_per_second = 1.2  # Very conservative: 72 wpm (TTS is usually 2.5-3x faster)
        self._early_barge_in = False  # Flag: next message should append to previous
        self._previous_user_message = ""  # Store last user message for appending
        self._vector_query_override = None  # Override for vector query (raw single message)
        self._playback_start_time: Optional[float] = None  # Track when TTS playback started
        self._transcription_preconnect_task: Optional[asyncio.Task] = None  # Track pre-connect for reliable handoff
        
        # TTS chunked synthesis config (for reduced latency on long messages)
        tts_config = config.get('tts', {}).get('config', {})
        self._chunked_synthesis_threshold = int(tts_config.get('chunked_synthesis_threshold', 150))
        
        # Conversation recording (Supabase)
        recording_config = config.get('recording', {})
        self._recording_enabled = recording_config.get('enabled', False)
        self._recorder: Optional[ConversationRecorder] = None
        self._supabase_url = recording_config.get('supabase_url')
        self._supabase_key = recording_config.get('supabase_key')
        
        # Track tool calls from last response (for recording)
        self._last_tool_calls = []
        
        # Context provider for conversation memory
        context_config = config.get('context', {}).get('config', {})
        self._context: Optional[ContextInterface] = UnifiedContextProvider(context_config)
        
        # Diagnostics: timeline ring buffer for debugging state transitions and events
        # Stores last 100 events for postmortem analysis on errors
        self._timeline: deque = deque(maxlen=100)
        
        self.is_initialized = False
    
    async def initialize(self, start_mcp: bool = True) -> bool:
        """
        Initialize orchestrator and ALL providers upfront.
        
        This creates and initializes all providers at startup for:
        - Consistent memory usage
        - No segfaults from rapid create/destroy
        - Faster operation (no lazy loading delays)
        
        Args:
            start_mcp: Whether to start MCP server (default: True)
        """
        try:
            print("🚀 Initializing RefactoredOrchestrator...")
            print("📦 Initializing all providers upfront for stable memory usage...")
            
            # Register recovery strategies
            self._register_recovery_strategies()
            
            # Initialize ALL providers upfront (not lazy)
            # This ensures consistent memory allocation
            print("\n🔧 Creating wake word provider...")
            self._wakeword = await self._create_wakeword_provider()
            
            print("🔧 Creating transcription provider...")
            self._transcription = await self._create_transcription_provider()
            
            if start_mcp:
                print("🔧 Creating response provider (starts MCP server)...")
                self._response = await self._create_response_provider()
            
            print("🔧 Creating TTS provider...")
            self._tts = await self._create_tts_provider()
            
            # Pass TTS provider to response provider for tool call announcements
            if self._response and hasattr(self._response, 'set_tts_provider'):
                self._response.set_tts_provider(self._tts)
            
            # Pre-cache TTS announcements for instant playback
            if ENABLE_TTS_ANNOUNCEMENTS and self._tts:
                # Get tool names from response provider for tool announcement caching
                tool_names = None
                if self._response and hasattr(self._response, 'available_tools'):
                    tool_names = list(self._response.available_tools.keys())
                await precache_announcements(self._tts, tool_names)
            
            # Run remaining initialization in PARALLEL for faster boot
            # - Termination detection (~1-2s, loads ONNX model)
            # - Conversation recorder (~0.1s)
            # - Vector memory (~0.3s)
            async def init_termination():
                if self._termination_enabled:
                    print("🔧 Creating termination detection provider...")
                    return await self._create_termination_provider()
                return None
            
            async def init_recorder():
                if self._recording_enabled and self._supabase_url and self._supabase_key:
                    print("🔧 Initializing conversation recorder...")
                    recorder = ConversationRecorder(
                        supabase_url=self._supabase_url,
                        supabase_key=self._supabase_key
                    )
                    await recorder.initialize()
                    return recorder
                return None
            
            async def init_vector_memory():
                if self._context and hasattr(self._context, 'initialize_vector_memory'):
                    print("🔧 Initializing vector memory...")
                    success = await self._context.initialize_vector_memory()
                    if success:
                        print("✅ Vector memory initialized")
                    else:
                        print("⚠️  Vector memory initialization failed (continuing without)")
                    return success
                return False
            
            # Run all three in parallel
            termination_result, recorder_result, _ = await asyncio.gather(
                init_termination(),
                init_recorder(),
                init_vector_memory(),
                return_exceptions=True
            )
            
            # Handle results
            if isinstance(termination_result, Exception):
                print(f"⚠️  Termination detection failed: {termination_result}")
            else:
                self._termination = termination_result
            
            if isinstance(recorder_result, Exception):
                print(f"⚠️  Conversation recorder failed: {recorder_result}")
            else:
                self._recorder = recorder_result
            
            # Register cleanup handlers with state machine (CRITICAL for preventing segfaults)
            self._register_cleanup_handlers()
            
            self.is_initialized = True
            print("\n✅ All providers initialized and ready")
            print("💡 Providers will be reused across conversations (cleanup on transitions)")
            # Note: beep_wake_model_ready() plays when wake word detection actually starts
            log_boot()  # 📡 Remote console log
            return True
            
        except Exception as e:
            print(f"❌ Orchestrator initialization failed: {e}")
            error = ComponentError(
                component="orchestrator",
                severity=ErrorSeverity.FATAL,
                message="Initialization failed",
                exception=e
            )
            await self.error_handler.handle_error(error)
            return False
    
    def _register_recovery_strategies(self):
        """Register component recovery strategies."""
        
        async def recover_wakeword(error):
            """Recover wake word detection."""
            print("🔧 Attempting wake word recovery...")
            if self._wakeword:
                await self._wakeword.cleanup()
                self._wakeword = None
            await asyncio.sleep(1.0)
            # Will be recreated on next use
        
        async def recover_transcription(error):
            """Recover transcription."""
            print("🔧 Attempting transcription recovery...")
            if self._transcription:
                await self._transcription.cleanup()
                self._transcription = None
            await asyncio.sleep(1.0)
        
        async def recover_response(error):
            """Recover response generation."""
            print("🔧 Attempting response recovery...")
            if self._response:
                await self._response.cleanup()
                self._response = None
            await asyncio.sleep(1.0)
        
        async def recover_tts(error):
            """Recover TTS."""
            print("🔧 Attempting TTS recovery...")
            if self._tts:
                await self._tts.cleanup()
                self._tts = None
            await asyncio.sleep(1.0)
        
        self.error_handler.register_recovery("wakeword", recover_wakeword)
        self.error_handler.register_recovery("transcription", recover_transcription)
        self.error_handler.register_recovery("response", recover_response)
        self.error_handler.register_recovery("tts", recover_tts)
    
    def _register_cleanup_handlers(self):
        """
        Register cleanup handlers with state machine for proper resource release.
        
        IMPORTANT: All handlers must be IDEMPOTENT - safe to call multiple times.
        The state machine calls these during transitions, but providers may also
        be cleaned up directly by orchestrator methods.
        """
        
        async def cleanup_wakeword():
            """Stop wake word detection and release audio resources."""
            if self._wakeword:
                try:
                    # Check if actually listening before stopping
                    if getattr(self._wakeword, 'is_listening', False) or getattr(self._wakeword, '_is_listening', False):
                        await self._wakeword.stop_detection()
                        print("✅ Wake word resources released")
                    else:
                        print("ℹ️  Wake word already stopped")
                except Exception as e:
                    print(f"⚠️  Wake word cleanup error: {e}")
        
        async def cleanup_transcription():
            """Stop transcription and release audio/network resources."""
            if self._transcription:
                try:
                    # Check if actually active before stopping (idempotent check)
                    if getattr(self._transcription, 'is_active', False) or getattr(self._transcription, '_is_active', False):
                        await self._transcription.stop_streaming()
                        print("✅ Transcription resources released")
                    else:
                        print("ℹ️  Transcription already stopped")
                except Exception as e:
                    print(f"⚠️  Transcription cleanup error: {e}")
        
        async def cleanup_response():
            """Release response generation resources (websockets, etc.)."""
            if self._response:
                try:
                    # Response provider manages its own lifecycle
                    # No active cleanup needed between requests
                    print("✅ Response provider ready for next request")
                except Exception as e:
                    print(f"⚠️  Response cleanup error: {e}")
        
        async def cleanup_tts():
            """Stop TTS playback and release audio resources."""
            if self._tts:
                try:
                    # Only stop if actually playing
                    if hasattr(self._tts, 'is_playing') and self._tts.is_playing:
                        if hasattr(self._tts, 'stop_audio'):
                            self._tts.stop_audio()
                        print("✅ TTS playback stopped")
                    else:
                        print("✅ TTS resources released")
                except Exception as e:
                    print(f"⚠️  TTS cleanup error: {e}")
        
        # Register handlers with state machine
        self.state_machine.register_cleanup_handler("wakeword", cleanup_wakeword)
        self.state_machine.register_cleanup_handler("transcription", cleanup_transcription)
        self.state_machine.register_cleanup_handler("response", cleanup_response)
        self.state_machine.register_cleanup_handler("tts", cleanup_tts)
        
        print("✅ Cleanup handlers registered with state machine")
    
    # =========================================================================
    # TRANSITION WRAPPER & DIAGNOSTICS
    # =========================================================================
    
    async def _transition(
        self,
        to_state: AudioState,
        component: Optional[str],
        ctx: TransitionContext
    ) -> bool:
        """
        THE ONLY WAY to change audio state in this orchestrator.
        
        All state transitions flow through this method to ensure:
        - Every transition is logged with full context (reason, initiator, etc.)
        - Timeline events are recorded for debugging
        - Consistent error handling
        - Easy to add metrics/alerts later
        
        Args:
            to_state: Target AudioState
            component: Component name for cleanup handler ("wakeword", "transcription", etc.)
            ctx: TransitionContext with reason, initiator, and optional metadata
            
        Returns:
            True if transition succeeded, False otherwise
        """
        from_state = self.state_machine.current_state
        start = time.perf_counter()
        
        # Build metadata from context
        metadata = ctx.to_metadata()
        
        # Attempt the transition
        try:
            ok = await self.state_machine.transition_to(
                to_state,
                component=component,
                metadata=metadata
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            
            # Record to timeline
            self._record_timeline_event("state_transition", {
                "from": from_state.name,
                "to": to_state.name,
                "component": component,
                "reason": ctx.reason.value,
                "initiated_by": ctx.initiated_by,
                "ok": ok,
                "ms": elapsed_ms,
            })
            
            if not ok:
                print(f"⚠️  Transition {from_state.name} → {to_state.name} failed "
                      f"(reason: {ctx.reason.value}, by: {ctx.initiated_by})")
            
            return ok
            
        except ValueError as e:
            # Invalid transition attempted
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            
            # Record failed transition
            self._record_timeline_event("state_transition_error", {
                "from": from_state.name,
                "to": to_state.name,
                "component": component,
                "reason": ctx.reason.value,
                "initiated_by": ctx.initiated_by,
                "error": str(e),
                "ms": elapsed_ms,
            })
            
            print(f"❌ Invalid transition {from_state.name} → {to_state.name}: {e}")
            print(f"   Reason: {ctx.reason.value}, Initiated by: {ctx.initiated_by}")
            
            # Auto-dump recent history for debugging
            self._dump_recent_history()
            
            raise
    
    def _record_timeline_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record event to timeline ring buffer.
        
        Events are stored with timestamp and can be dumped for debugging
        when errors occur.
        
        Args:
            event_type: Type of event ("state_transition", "barge_in", "error", etc.)
            data: Event-specific data to record
        """
        self._timeline.append({
            "ts": datetime.now().isoformat(),
            "type": event_type,
            **data
        })
    
    def _estimate_remaining_text(self, full_text: str) -> Optional[str]:
        """
        Estimate remaining text after barge-in based on elapsed playback time.
        
        Uses playback start time and average speech rate to determine approximately
        how much text was spoken, then returns the remaining unspoken text.
        
        Args:
            full_text: The full TTS text that was being spoken
            
        Returns:
            The remaining text that wasn't spoken, or None if nearly complete
        """
        if not self._playback_start_time:
            return full_text  # No timing info, return full text
        
        elapsed = asyncio.get_event_loop().time() - self._playback_start_time
        
        # Estimate words spoken using CONSERVATIVE rate (intentionally low)
        # This ensures we underestimate → repeat a few words rather than skip any
        words_spoken = int(elapsed * self._tts_words_per_second)
        
        # Split into words
        words = full_text.split()
        total_words = len(words)
        
        # If we've spoken most of the text (>90%), don't bother recovering
        if words_spoken >= total_words * 0.9:
            return None
        
        # Get remaining words
        remaining_words = words[words_spoken:]
        if not remaining_words:
            return None
        
        remaining_text = " ".join(remaining_words)
        
        print(f"📍 Playback estimate: {elapsed:.1f}s elapsed, ~{words_spoken}/{total_words} words spoken")
        
        return remaining_text
    
    def _dump_recent_history(self, last_n: int = 30) -> None:
        """
        Print recent timeline for debugging.
        
        Called automatically on transition errors. Can also be called manually
        for diagnostics.
        
        Args:
            last_n: Number of recent events to show (default: 30)
        """
        print("\n" + "=" * 70)
        print("🔍 RECENT TIMELINE (for debugging)")
        print("=" * 70)
        
        events = list(self._timeline)[-last_n:]
        
        if not events:
            print("  (no events recorded)")
            print("=" * 70 + "\n")
            return
        
        for e in events:
            ts = e.get("ts", "?")
            # Extract just the time portion for readability
            if "T" in ts:
                ts = ts.split("T")[1][:12]  # HH:MM:SS.mmm
            
            etype = e.get("type", "?")
            
            if etype == "state_transition":
                status = "✓" if e.get("ok", False) else "✗"
                print(f"  {ts} | {status} {e.get('from','?'):22} → {e.get('to','?'):22} "
                      f"| {e.get('reason','?')} ({e.get('ms',0)}ms)")
            elif etype == "state_transition_error":
                print(f"  {ts} | ❌ {e.get('from','?'):22} → {e.get('to','?'):22} "
                      f"| {e.get('reason','?')} ERROR: {e.get('error','?')}")
            elif etype == "barge_in":
                print(f"  {ts} | 🎤 BARGE-IN | early={e.get('early', False)}, "
                      f"elapsed={e.get('elapsed_s', '?')}s")
            elif etype == "termination":
                print(f"  {ts} | 🛑 TERMINATION | phrase={e.get('phrase', '?')}, "
                      f"state={e.get('interrupted_state', '?')}")
            elif etype == "error":
                print(f"  {ts} | ❌ ERROR | component={e.get('component', '?')}, "
                      f"msg={e.get('message', '?')[:50]}")
            else:
                # Generic event
                detail = ", ".join(f"{k}={v}" for k, v in e.items() 
                                   if k not in ("ts", "type"))
                print(f"  {ts} | {etype}: {detail[:60]}")
        
        print("=" * 70)
        
        # Also print current state
        print(f"  Current state: {self.state_machine.current_state.name}")
        print("=" * 70 + "\n")
    
    def _get_conversation_id(self) -> Optional[str]:
        """Helper to get current conversation ID for transition context."""
        if self._recorder and self._recorder.current_session_id:
            return self._recorder.current_session_id
        return None
    
    async def _create_wakeword_provider(self) -> WakeWordInterface:
        """Create wake word provider (called once at startup)."""
        config = self.config['wakeword']['config']
        provider = IsolatedOpenWakeWordProvider(config)
        await provider.initialize()
        return provider
    
    async def _create_transcription_provider(self) -> TranscriptionInterface:
        """Create transcription provider (called once at startup)."""
        config = self.config['transcription']['config']
        # Pass shared audio bus for zero-latency transitions
        provider = AssemblyAIAsyncProvider(config, shared_bus=self._shared_audio_bus)
        await provider.initialize()
        return provider
    
    async def _create_response_provider(self) -> ResponseInterface:
        """Create response provider (called once at startup)."""
        from .factory import ProviderFactory
        provider_config = {
            'response': self.config['response']
        }
        providers = ProviderFactory.create_all_providers(provider_config)
        provider = providers['response']
        await provider.initialize()
        return provider
    
    async def _create_tts_provider(self) -> TextToSpeechInterface:
        """Create TTS provider (called once at startup)."""
        from .factory import ProviderFactory
        provider_config = {
            'tts': self.config['tts']
        }
        providers = ProviderFactory.create_all_providers(provider_config)
        provider = providers['tts']
        await provider.initialize()
        return provider
    
    async def _create_termination_provider(self) -> Optional[TerminationInterface]:
        """
        Create termination detection provider (called once at startup).
        
        Returns None if model not available (graceful degradation).
        """
        try:
            provider = IsolatedTerminationProvider(TERMINATION_DETECTION_CONFIG)
            if not provider.is_available:
                print("⚠️  Termination detection unavailable (model not found)")
                return None
            success = await provider.initialize()
            if success:
                print("✅ Termination detection ready")
                return provider
            else:
                print("⚠️  Termination detection initialization failed")
                return None
        except Exception as e:
            print(f"⚠️  Termination detection setup failed: {e}")
            return None
    
    async def _start_termination_detection(self) -> None:
        """
        Start parallel termination detection.
        
        Creates a background task that monitors for termination phrases
        and sets the _termination_detected flag when triggered.
        """
        if not self._termination or not self._termination_enabled:
            return
        
        async def _monitor_termination():
            """Background task to monitor for termination phrase."""
            try:
                # Resume detection (provider starts paused)
                await self._termination.resume_detection()
                
                async for event in self._termination.start_detection():
                    if event:
                        print(f"🛑 Termination phrase detected: {event.phrase_name}")
                        print(f"   Interrupted state: {event.interrupted_state}")
                        log_termination_detected(event.phrase_name, event.interrupted_state or "UNKNOWN")
                        beep_shutdown()
                        # Note: "Conversation ended" TTS announcement is played at session end
                        # (handles all conversation end paths, not just termination phrase)
                        self._termination_detected = True
                        break  # Stop after first detection
            except asyncio.CancelledError:
                pass  # Normal cancellation
            except Exception as e:
                print(f"⚠️  Termination detection error: {e}")
        
        # Reset flag and start monitoring task
        self._termination_detected = False
        self._termination_task = asyncio.create_task(_monitor_termination())
    
    async def _stop_termination_detection(self, fire_and_forget: bool = False) -> None:
        """
        Stop parallel termination detection.
        
        Args:
            fire_and_forget: If True, don't wait for pause confirmation (faster).
                           Use when termination was triggered (already stopped).
        """
        if self._termination_task:
            self._termination_task.cancel()
            try:
                await self._termination_task
            except asyncio.CancelledError:
                pass
            self._termination_task = None
        
        if self._termination:
            if fire_and_forget:
                # Don't wait - detection already stopped when termination triggered
                asyncio.create_task(self._termination.pause_detection())
            else:
                await self._termination.pause_detection()
    
    def _update_termination_state(self, state: str) -> None:
        """Update termination detector with current state (for metadata)."""
        if self._termination:
            self._termination.set_current_state(state)
    
    async def run_wake_word_detection(self):
        """Run wake word detection loop."""
        try:
            # Transition to wake word state
            await self._transition(
                AudioState.WAKE_WORD_LISTENING,
                component="wakeword",
                ctx=TransitionContext(
                    reason=TransitionReason.WAKE_DETECTED,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                )
            )
            
            # Use pre-initialized provider
            wakeword = self._wakeword
            
            # Start detection
            print("👂 Listening for wake word...")
            async for event in wakeword.start_detection():
                print(f"🔔 Wake word detected: {event.model_name} (score: {event.score:.3f})")
                # Note: Wake word beep now handled by state machine transition
                log_wake_word(event.model_name, event.score)  # 📡 Remote console log
                yield event
            
        except Exception as e:
            print(f"❌ Wake word detection error: {e}")
            error = ComponentError(
                component="wakeword",
                severity=ErrorSeverity.RECOVERABLE,
                message="Detection error",
                exception=e
            )
            await self.error_handler.handle_error(error)
            raise
        
        # Note: No finally block needed - state machine handles cleanup on transition
        # The finally block was causing double cleanup (here + state machine)
        # which led to segmentation faults
    
    async def run_transcription(self, use_extended_timeout: bool = False) -> Optional[str]:
        """
        Run transcription and return final text when send phrase is detected.
        
        Args:
            use_extended_timeout: If True, use a longer silence timeout (for post-tool-call follow-ups)
        """
        try:
            # Transition to transcription state
            await self._transition(
                AudioState.TRANSCRIBING,
                component="transcription",
                ctx=TransitionContext(
                    reason=TransitionReason.TRANSCRIPTION_START,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                )
            )
            
            # Wait for pre-connect task if it was started during TTS
            if self._transcription_preconnect_task:
                try:
                    await asyncio.wait_for(self._transcription_preconnect_task, timeout=0.3)
                except asyncio.TimeoutError:
                    pass  # Will connect on-demand
                self._transcription_preconnect_task = None
            
            # Use pre-initialized provider
            transcription = self._transcription
            if not transcription:
                print("❌ Transcription provider not available")
                return None
            
            # Check for barge-in prefill audio
            # IMPORTANT: Prefill is ONLY used when barge-in was detected.
            # _barge_in_audio is ONLY set by:
            #   1. run_response() when barge_in_during_response is True
            #   2. run_tts() when barge_in_occurred is True
            # Otherwise, it remains None and no prefill is used.
            if self._barge_in_audio:
                duration = len(self._barge_in_audio) / 2 / 16000
                print(f"📼 BARGE-IN PREFILL: {duration:.2f}s ({len(self._barge_in_audio)} bytes) → transcription")
                if hasattr(transcription, 'set_prefill_audio'):
                    transcription.set_prefill_audio(self._barge_in_audio)
                    print("✅ Prefill audio set on transcription provider")
                else:
                    print("⚠️  Transcription provider doesn't support prefill audio")
                self._barge_in_audio = None  # Clear after use
            else:
                print("🎤 Starting fresh transcription (no barge-in prefill)")
            
            # Get send phrases from config
            from .config import SEND_PHRASES, TERMINATION_PHRASES, PREFIX_TRIM_PHRASES, AUTO_SEND_SILENCE_TIMEOUT, AUTO_SEND_SILENCE_TIMEOUT_DURING_TOOLS
            
            def apply_trim_phrases(text: str) -> str:
                """Apply trim phrases - drop all text before and including the phrase (case insensitive)."""
                if not text:
                    return text
                result = text
                text_lower = text.lower()
                for trim_phrase in PREFIX_TRIM_PHRASES:
                    trim_lower = trim_phrase.lower()
                    if trim_lower in text_lower:
                        idx = text_lower.rfind(trim_lower)  # Use last occurrence
                        result = text[idx + len(trim_phrase):].strip()
                        text_lower = result.lower()
                        print(f"✂️  Trim phrase '{trim_phrase}' applied - dropped preceding text")
                return result
            
            def apply_custom_spelling(text: str) -> str:
                """Apply custom spelling corrections (case-insensitive find-and-replace)."""
                if not text:
                    return text
                from .config import ASSEMBLYAI_CONFIG
                custom_spelling = ASSEMBLYAI_CONFIG.get("custom_spelling", {})
                if not custom_spelling:
                    return text
                result = text
                for correct, wrong_forms in custom_spelling.items():
                    for wrong in wrong_forms:
                        # Case-insensitive replacement using re
                        import re
                        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                        if pattern.search(result):
                            result = pattern.sub(correct, result)
                return result
            
            # Use extended timeout if requested (e.g., after tool calls)
            effective_timeout = AUTO_SEND_SILENCE_TIMEOUT_DURING_TOOLS if use_extended_timeout else AUTO_SEND_SILENCE_TIMEOUT
            
            # Start streaming
            print("🎙️  Transcribing...")
            print(f"💡 Say one of these to send: {', '.join(SEND_PHRASES)}")
            if effective_timeout > 0:
                timeout_note = " (extended after tool calls)" if use_extended_timeout else ""
                print(f"⏱️  Auto-send after {effective_timeout:.0f}s of silence{timeout_note}")
            # Note: Listening start beep now handled by state machine transition
            
            accumulated_text = ""
            real_speech_after_grace = False  # Track if user spoke AFTER grace period (not just prefill)
            grace_period_ended_logged = False  # Track if we've logged grace period end
            
            # Create an async iterator we can poll with timeout
            stream_iter = transcription.start_streaming().__aiter__()
            
            # NOW start the timers - after stream is created (closer to actual session start)
            last_activity_time = asyncio.get_event_loop().time()
            transcription_start_time = last_activity_time  # Track when transcription started for recovery
            
            # Barge-in recovery mode logging
            if self._barge_in_recovery_mode:
                print(f"🔄 Barge-in recovery mode: will resume TTS if no NEW speech within {self._barge_in_recovery_timeout}s")
                print(f"⏳ GRACE PERIOD STARTED at t=0.00s: ignoring transcription results for {self._barge_in_recovery_grace_period}s")
            
            while True:
                try:
                    # Calculate remaining time until auto-send or recovery
                    elapsed_since_start = asyncio.get_event_loop().time() - transcription_start_time
                    
                    # Log when grace period ends
                    if self._barge_in_recovery_mode and not grace_period_ended_logged:
                        if elapsed_since_start >= self._barge_in_recovery_grace_period:
                            print(f"⏳ GRACE PERIOD ENDED at t={elapsed_since_start:.2f}s - now listening for real speech")
                            grace_period_ended_logged = True
                    
                    # Check recovery timeout FIRST if in recovery mode and no real speech yet
                    # (accumulated_text may exist from prefill, but we only care about real speech after grace)
                    if self._barge_in_recovery_mode and not real_speech_after_grace:
                        recovery_remaining = self._barge_in_recovery_timeout - elapsed_since_start
                        if recovery_remaining <= 0:
                            # Recovery timeout expired - no real speech detected, resume TTS
                            if accumulated_text:
                                print(f"⏱️  Only prefill speech detected, no new speech after {self._barge_in_recovery_timeout}s - resuming interrupted TTS")
                            else:
                                print(f"⏱️  No speech detected after {self._barge_in_recovery_timeout}s - resuming interrupted TTS")
                            self._barge_in_recovery_mode = False
                            # Return special marker to signal TTS resume
                            return "__RESUME_TTS__"
                        timeout = min(recovery_remaining, 1.0)  # Check every second max
                    elif effective_timeout > 0 and accumulated_text and real_speech_after_grace:
                        # Normal auto-send: only if real speech occurred after grace period
                        elapsed = asyncio.get_event_loop().time() - last_activity_time
                        remaining = effective_timeout - elapsed
                        if remaining <= 0:
                            # Auto-send timeout reached
                            print(f"⏱️  Auto-sending after {effective_timeout:.0f}s of silence...")
                            beep_send_detected()  # 🔔 Send phrase sound
                            # Clear recovery mode since user spoke
                            self._barge_in_recovery_mode = False
                            self._interrupted_tts_response = None
                            return accumulated_text
                        timeout = remaining
                    elif effective_timeout > 0 and accumulated_text:
                        # Has accumulated text but not confirmed as real speech yet
                        # Use both timeouts - whichever comes first
                        elapsed = asyncio.get_event_loop().time() - last_activity_time
                        auto_send_remaining = effective_timeout - elapsed
                        if self._barge_in_recovery_mode:
                            recovery_remaining = self._barge_in_recovery_timeout - elapsed_since_start
                            timeout = min(auto_send_remaining, recovery_remaining, 1.0)
                        else:
                            timeout = auto_send_remaining
                    else:
                        # No auto-send or no text yet - wait indefinitely (long timeout)
                        timeout = 60.0
                    
                    # Wait for next transcription result with timeout
                    result = await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout)
                    
                    # Reset activity timer on any transcription activity
                    last_activity_time = asyncio.get_event_loop().time()
                    
                    # Check ANY speech activity (final OR partial) after grace period to cancel recovery
                    # This prevents recovery timeout from firing while user is actively speaking
                    if self._barge_in_recovery_mode and result.text.strip():
                        elapsed_since_start = asyncio.get_event_loop().time() - transcription_start_time
                        if elapsed_since_start >= self._barge_in_recovery_grace_period:
                            if not real_speech_after_grace:
                                print(f"✅ User speech detected after grace period ({elapsed_since_start:.1f}s) - cancelling TTS recovery")
                                self._barge_in_recovery_mode = False
                                self._interrupted_tts_response = None
                                real_speech_after_grace = True  # Mark that real speech occurred
                        else:
                            print(f"⏳ Speech during grace period ({elapsed_since_start:.1f}s < {self._barge_in_recovery_grace_period}s) - may be prefill, waiting for more...")
                    
                    if result.is_final:
                        
                        # Accumulate final transcriptions
                        if accumulated_text:
                            accumulated_text += " " + result.text
                        else:
                            accumulated_text = result.text
                        
                        print(f"📝 Final: {result.text}")
                        
                        # Check for send phrases (case-insensitive)
                        accumulated_lower = accumulated_text.lower()
                        for send_phrase in SEND_PHRASES:
                            if send_phrase.lower() in accumulated_lower:
                                print(f"✅ Send phrase detected in final: '{send_phrase}'")
                                beep_send_detected()  # 🔔 Send phrase sound
                                # Remove the send phrase from the accumulated text (case-insensitive)
                                import re
                                pattern = re.compile(re.escape(send_phrase), re.IGNORECASE)
                                cleaned_text = pattern.sub("", accumulated_text).strip()
                                # Clean up extra spaces
                                cleaned_text = " ".join(cleaned_text.split())
                                # Apply trim phrases and custom spelling before returning
                                cleaned_text = apply_custom_spelling(apply_trim_phrases(cleaned_text))
                                return cleaned_text if cleaned_text else None
                        
                        # Check for termination phrases
                        for term_phrase in TERMINATION_PHRASES:
                            if term_phrase.lower() in accumulated_lower:
                                print(f"🛑 Termination phrase detected: '{term_phrase}'")
                                beep_shutdown()  # 🔔 Shutdown/goodbye sound
                                log_conversation_end()  # 📡 Remote console log - graceful end
                                return None
                    else:
                        # Check partials too for faster response
                        print(f"📝 Partial: {result.text}")
                        
                        # Build full text including partial
                        full_text = accumulated_text + " " + result.text if accumulated_text else result.text
                        full_text_lower = full_text.lower()
                        
                        # Check for send phrases in partial (for instant response)
                        for send_phrase in SEND_PHRASES:
                            if send_phrase.lower() in full_text_lower:
                                print(f"⚡ Send phrase detected in partial: '{send_phrase}' (instant send!)")
                                beep_send_detected()  # 🔔 Send phrase sound
                                # Remove the send phrase
                                import re
                                pattern = re.compile(re.escape(send_phrase), re.IGNORECASE)
                                cleaned_text = pattern.sub("", full_text).strip()
                                cleaned_text = " ".join(cleaned_text.split())
                                # Apply trim phrases and custom spelling before returning
                                cleaned_text = apply_custom_spelling(apply_trim_phrases(cleaned_text))
                                return cleaned_text if cleaned_text else None
                        
                        # Check for termination phrases in partial
                        for term_phrase in TERMINATION_PHRASES:
                            if term_phrase.lower() in full_text_lower:
                                print(f"🛑 Termination phrase detected in partial: '{term_phrase}'")
                                beep_shutdown()  # 🔔 Shutdown/goodbye sound
                                log_conversation_end()  # 📡 Remote console log - graceful end
                                return None
                                
                except asyncio.TimeoutError:
                    if effective_timeout > 0 and accumulated_text:
                        print(f"⏱️  Auto-sending after {effective_timeout:.0f}s of silence...")
                        beep_send_detected()  # 🔔 Send phrase sound
                        # Apply trim phrases and custom spelling before returning
                        return apply_custom_spelling(apply_trim_phrases(accumulated_text))
                    continue
                    
                except StopAsyncIteration:
                    # Stream ended naturally
                    break
            
            # Stream ended - check if we should resume TTS (recovery mode)
            if self._barge_in_recovery_mode and not real_speech_after_grace:
                # No real speech was detected - should resume TTS
                if accumulated_text:
                    print(f"🔄 Stream ended with only prefill speech - resuming interrupted TTS")
                else:
                    print(f"🔄 Stream ended with no speech - resuming interrupted TTS")
                self._barge_in_recovery_mode = False
                return "__RESUME_TTS__"
            
            # If stream ends naturally without send phrase, return accumulated text
            # Apply trim phrases and custom spelling before returning
            final_text = apply_custom_spelling(apply_trim_phrases(accumulated_text)) if accumulated_text else None
            return final_text if final_text else None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            
            # Check if we should resume TTS instead of failing
            if self._barge_in_recovery_mode:
                print(f"🔄 Transcription error during recovery mode - resuming interrupted TTS")
                self._barge_in_recovery_mode = False
                return "__RESUME_TTS__"
            
            error = ComponentError(
                component="transcription",
                severity=ErrorSeverity.RECOVERABLE,
                message="Transcription error",
                exception=e
            )
            await self.error_handler.handle_error(error)
            # Transition to IDLE to allow recovery
            try:
                if self.state_machine.current_state != AudioState.IDLE:
                    await self._transition(
                        AudioState.IDLE,
                        component=None,
                        ctx=TransitionContext(
                            reason=TransitionReason.TRANSCRIPTION_ERROR,
                            initiated_by="error_handler",
                            conversation_id=self._get_conversation_id(),
                            error_message=str(e)[:200],
                        )
                    )
            except ValueError:
                await self.state_machine.emergency_reset()
            return None
        
        # Note: No finally block needed - state machine handles cleanup on transition
        # The finally block was causing double cleanup (here + state machine)
        # which led to segmentation faults
    
    async def run_response(self, user_message: str, enable_barge_in: bool = True, skip_vector_memory: bool = False, skip_conversation_context: bool = False) -> Optional[str]:
        """
        Generate response for user message with conversation context.
        
        Args:
            user_message: The user's message to respond to
            enable_barge_in: If True, allows user to interrupt during response generation
            skip_vector_memory: If True, skip vector memory context (for opener synthesis)
            skip_conversation_context: If True, skip conversation history (for opener synthesis)
            
        Returns:
            The assistant's response, or None if interrupted/failed
        """
        # Clear previous tool calls
        self._last_tool_calls = []
        barge_in_during_response = False
        
        try:
            # Ensure transcription is fully stopped before starting response
            # This prevents audio device conflicts and executor thread issues
            # NOTE: The state machine cleanup handler will also call stop_streaming,
            # but stop_streaming is idempotent so this is safe
            if self._transcription and self._transcription.is_active:
                print("🔄 Ensuring transcription fully stopped before response...")
                await self._transcription.stop_streaming()
                # Brief wait for audio device to fully release
                await asyncio.sleep(self._transcription_stop_delay)
            
            # Transition to response state
            # NOTE: State machine may trigger additional cleanup via registered handlers
            await self._transition(
                AudioState.PROCESSING_RESPONSE,
                component="response",
                ctx=TransitionContext(
                    reason=TransitionReason.RESPONSE_START,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                    user_message_snippet=user_message[:50] if user_message else None,
                )
            )
            
            # Add user message to context BEFORE generating response
            if self._context:
                self._context.add_message("user", user_message)
                # Auto-trim if context is getting too long
                self._context.auto_trim_if_needed()
            
            # Use pre-initialized provider
            response = self._response
            if not response:
                print("❌ Response provider not available (MCP not started?)")
                return None
            
            # Start vector memory query in parallel (before context preparation)
            # Skip for opener synthesis to avoid irrelevant past conversation context
            vector_task = None
            if not skip_vector_memory and self._context and hasattr(self._context, 'get_vector_memory_context'):
                query_for_vector = getattr(self, '_vector_query_override', None) or user_message
                vector_task = asyncio.create_task(
                    self._context.get_vector_memory_context(query_for_vector)
                )
            
            # Get unified context bundle (single pass over conversation history)
            # Skip for opener synthesis - briefings should be self-contained
            context = None
            tool_context = None
            if self._context and not skip_conversation_context:
                # Use unified method for efficiency (computes both in one pass)
                if hasattr(self._context, 'get_context_bundle'):
                    bundle = self._context.get_context_bundle()
                    context = bundle.response_context
                    tool_context = bundle.tool_context
                else:
                    # Fallback to separate calls for compatibility
                    context = self._context.get_recent_for_response()
                    tool_context = self._context.get_tool_context()
            
            # Wait for vector memory query (parallelized with context prep above)
            vector_context = ""
            if vector_task:
                vector_context = await vector_task or ""
                if vector_context and context:
                    # Inject vector memory as a system message in context
                    context.insert(1, {"role": "system", "content": vector_context})
            
            # Stream response with context
            print("💭 Generating response...")
            log_user_message(user_message)  # 📡 Remote console log
            
            # Start barge-in detection during response generation if enabled
            # This allows user to interrupt while LLM is thinking
            if enable_barge_in and self._barge_in_enabled and self._barge_in_during_processing:
                self._barge_in_triggered = False
                bt_mode = " [Bluetooth/Meta]" if self._barge_in_is_bluetooth else ""
                print(f"👂 Barge-in enabled during response generation{bt_mode}")
                
                self._barge_in_detector = BargeInDetector(
                    config=BargeInConfig(
                        mode=BargeInMode.ENERGY,
                        energy_threshold=self._barge_in_energy_threshold,
                        min_speech_duration=self._barge_in_min_speech_duration,
                        cooldown_after_tts_start=self._barge_in_processing_cooldown,
                        sample_rate=self._barge_in_sample_rate,
                        chunk_size=self._barge_in_chunk_size,
                        buffer_seconds=self._barge_in_buffer_seconds,
                        capture_after_trigger=self._barge_in_capture_after_trigger,
                        device_index=self._barge_in_device_index,
                        latency=self._barge_in_latency,
                        is_bluetooth=self._barge_in_is_bluetooth,
                        require_consecutive_speech=self._barge_in_require_consecutive,
                        silence_reset_threshold=self._barge_in_silence_reset_threshold,
                    ),
                    shared_bus=self._shared_audio_bus,
                )
                
                def on_processing_barge_in():
                    print("🎤 BARGE-IN: Interrupting response generation!")
                    self._barge_in_triggered = True
                
                await self._barge_in_detector.start(on_barge_in=on_processing_barge_in)
            
            # Record comprehensive API input tokens (system prompt + context + tools)
            if self._recorder and self._recorder.current_session_id:
                # Get system prompt (includes persistent memory)
                system_prompt = ""
                if context:
                    for msg in context:
                        if msg.get("role") == "system":
                            system_prompt += msg.get("content", "") + "\n"
                
                # Get tool definitions from response provider
                tool_defs = []
                if hasattr(response, 'openai_functions'):
                    tool_defs = response.openai_functions or []
                
                self._recorder.record_api_request(
                    system_prompt=system_prompt,
                    context_messages=context or [],
                    tool_definitions=tool_defs,
                    user_message=user_message,
                    vector_context=vector_context
                )
            
            full_response = ""
            streamed_deltas = False
            
            async for chunk in response.stream_response(user_message, context=context, tool_context=tool_context):
                # Check for barge-in during response streaming
                if self._barge_in_triggered:
                    print("\n⚡ Barge-in triggered - cancelling response generation")
                    barge_in_during_response = True
                    break
                
                if chunk.content:
                    if chunk.is_complete:
                        # Final complete chunk contains the FULL text
                        # If we were streaming deltas, we already printed them
                        # If not, print the complete response now
                        full_response = chunk.content
                        if not streamed_deltas:
                            print(chunk.content, end="", flush=True)
                        # Capture tool calls from final chunk
                        if chunk.tool_calls:
                            self._last_tool_calls = chunk.tool_calls
                            # Log tool calls to remote console
                            for tc in chunk.tool_calls:
                                tool_name = getattr(tc, 'name', None) or 'unknown'
                                log_tool_call(tool_name)  # 📡 Remote console log
                    else:
                        # Streaming delta - accumulate and print
                        full_response += chunk.content
                        print(chunk.content, end="", flush=True)
                        streamed_deltas = True
            
            print()  # Newline after response
            
            # Handle barge-in during response generation
            if barge_in_during_response:
                # Capture audio for transcription prefill
                if self._barge_in_detector:
                    self._barge_in_audio = self._barge_in_detector.get_captured_audio_bytes()
                    if self._barge_in_audio:
                        duration = len(self._barge_in_audio) / 2 / 16000
                        print(f"📼 Barge-in audio captured: {duration:.2f}s ({len(self._barge_in_audio)} bytes)")
                    else:
                        print("⚠️  No barge-in audio captured (detector returned None)")
                    await self._barge_in_detector.stop()
                    self._barge_in_detector = None
                else:
                    print("⚠️  Barge-in detector was None when trying to capture audio")
                
                # Add partial response to context if any
                if self._context and full_response:
                    self._context.add_message("assistant", full_response + " [interrupted]")
                
                return None  # Signal that response was interrupted
            
            # Capture composition API token usage (from tool decisions and final answer composition)
            if self._recorder and hasattr(response, 'get_composition_token_usage'):
                comp_usage = response.get_composition_token_usage()
                if comp_usage["total"] > 0:
                    # Add composition input tokens to session total
                    self._recorder._session_input_tokens += comp_usage["input_tokens"]
                    # Add composition output tokens to session total
                    self._recorder._session_output_tokens += comp_usage["output_tokens"]
                    print(f"📊 Composition totals: +{comp_usage['input_tokens']:,} in, +{comp_usage['output_tokens']:,} out")
            
            # Add assistant response to context AFTER generation
            if self._context and full_response:
                self._context.add_message("assistant", full_response)
            
            # Log response to remote console
            if full_response:
                log_assistant_response(full_response)  # 📡 Remote console log
            
            # Cleanup barge-in detector (if not already cleaned up)
            if self._barge_in_detector:
                await self._barge_in_detector.stop()
                self._barge_in_detector = None
            
            return full_response if full_response else None
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            
            # Cleanup barge-in detector on error
            if self._barge_in_detector:
                try:
                    await self._barge_in_detector.stop()
                except Exception:
                    pass
                self._barge_in_detector = None
            
            error = ComponentError(
                component="response",
                severity=ErrorSeverity.RECOVERABLE,
                message="Response error",
                exception=e
            )
            await self.error_handler.handle_error(error)
            # Transition to IDLE to allow recovery (PROCESSING_RESPONSE can't go directly to TRANSCRIBING)
            try:
                if self.state_machine.current_state != AudioState.IDLE:
                    await self._transition(
                        AudioState.IDLE,
                        component=None,
                        ctx=TransitionContext(
                            reason=TransitionReason.RESPONSE_ERROR,
                            initiated_by="error_handler",
                            conversation_id=self._get_conversation_id(),
                            error_message=str(e)[:200],
                        )
                    )
            except ValueError:
                # If normal transition fails, use emergency reset
                await self.state_machine.emergency_reset()
            return None
    
    async def _record_tool_calls(self) -> None:
        """Record any tool calls from the last response to Supabase."""
        if not self._recorder or not self._recorder.current_session_id:
            return
        
        if not self._last_tool_calls:
            return
        
        import json
        
        for tool_call in self._last_tool_calls:
            try:
                # Extract tool call details
                tool_name = getattr(tool_call, 'name', None) or 'unknown'
                arguments = getattr(tool_call, 'arguments', {}) or {}
                result = getattr(tool_call, 'result', None)
                
                # Serialize result if it's not already a string
                if result is not None and not isinstance(result, str):
                    try:
                        result = json.dumps(result)
                    except (TypeError, ValueError):
                        result = str(result)
                
                # Truncate result if too long (Supabase text limit)
                if result and len(result) > 10000:
                    result = result[:10000] + "... [truncated]"
                
                await self._recorder.record_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=None  # Could track this in the future
                )
                print(f"   📝 Recorded tool call: {tool_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to record tool call: {e}")
        
        # Clear tool calls after recording
        self._last_tool_calls = []
    
    async def run_tts(self, text: str, transition_to_idle: bool = True, enable_barge_in: bool = True) -> bool:
        """
        Synthesize and play text with optional barge-in support.
        
        Args:
            text: Text to synthesize
            transition_to_idle: If True, transitions to IDLE after TTS.
                               If False, stays in current state (for chaining to transcription)
            enable_barge_in: If True, allows user to interrupt speech by talking
            
        Returns:
            True if speech completed normally, False if interrupted by barge-in
        """
        self._barge_in_triggered = False
        barge_in_occurred = False
        
        try:
            # Transition to TTS state
            await self._transition(
                AudioState.SYNTHESIZING,
                component="tts",
                ctx=TransitionContext(
                    reason=TransitionReason.TTS_START,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                )
            )
            
            # Use pre-initialized provider
            tts = self._tts
            
            # Check for streaming support (preferred - lowest latency)
            # Streaming plays audio as it arrives from the API
            use_streaming = hasattr(tts, 'synthesize_and_play_streaming')
            
            # Check if text is long enough to benefit from chunked synthesis
            # Chunked synthesis reduces perceived latency by playing first chunk
            # while synthesizing the rest (fallback for non-streaming providers)
            use_chunked = (
                not use_streaming and
                self._chunked_synthesis_threshold > 0 and
                len(text) > self._chunked_synthesis_threshold and 
                hasattr(tts, 'synthesize_and_play_chunked')
            )
            
            if use_streaming:
                print(f"🔊 Using streaming synthesis...")
            elif use_chunked:
                print(f"🔊 Using chunked synthesis for {len(text)} chars...")
            else:
                # Synthesize audio first (for short messages)
                print("🔊 Synthesizing speech...")
                audio = await tts.synthesize(text)
            
            # Setup barge-in detection if enabled
            if enable_barge_in and self._barge_in_enabled:
                bt_mode = " [Bluetooth/Meta]" if self._barge_in_is_bluetooth else ""
                bus_status = "running" if (self._shared_audio_bus and self._shared_audio_bus.is_running) else "not running"
                print(f"👂 Barge-in enabled{bt_mode} - speak to interrupt (shared bus: {bus_status})")
                self._barge_in_detector = BargeInDetector(
                    config=BargeInConfig(
                        mode=BargeInMode.ENERGY,
                        energy_threshold=self._barge_in_energy_threshold,
                        min_speech_duration=self._barge_in_min_speech_duration,
                        cooldown_after_tts_start=self._barge_in_cooldown_after_tts_start,
                        sample_rate=self._barge_in_sample_rate,
                        chunk_size=self._barge_in_chunk_size,
                        buffer_seconds=self._barge_in_buffer_seconds,
                        capture_after_trigger=self._barge_in_capture_after_trigger,
                        device_index=self._barge_in_device_index,
                        latency=self._barge_in_latency,
                        is_bluetooth=self._barge_in_is_bluetooth,
                        require_consecutive_speech=self._barge_in_require_consecutive,
                        silence_reset_threshold=self._barge_in_silence_reset_threshold,
                    ),
                    shared_bus=self._shared_audio_bus,
                )
                print(
                    "🔧 Barge-in config: "
                    f"threshold={self._barge_in_energy_threshold}, "
                    f"min_speech={self._barge_in_min_speech_duration}s, "
                    f"cooldown={self._barge_in_cooldown_after_tts_start}s, "
                    f"device={self._barge_in_device_index}"
                )
                
                # Define callback to stop TTS when barge-in detected
                def on_barge_in():
                    print("🎤 BARGE-IN: Interrupting speech!")
                    self._barge_in_triggered = True
                    
                    # Check if this is an early barge-in (within threshold)
                    if self._playback_start_time:
                        elapsed = asyncio.get_event_loop().time() - self._playback_start_time
                        if elapsed < self._early_barge_in_threshold:
                            self._early_barge_in = True
                            print(f"⚡ Early barge-in ({elapsed:.1f}s < {self._early_barge_in_threshold}s) - will append next message")
                        else:
                            print(f"⏱️  Late barge-in ({elapsed:.1f}s >= {self._early_barge_in_threshold}s) - new message")
                    
                    if tts and hasattr(tts, 'stop_audio'):
                        tts.stop_audio()
                
                await self._barge_in_detector.start(on_barge_in=on_barge_in)
                
                # Start the early barge-in window now (when assistant starts responding)
                self._playback_start_time = asyncio.get_event_loop().time()
            
            # Pre-connect transcription WebSocket while speaking (for faster barge-in response)
            if self._transcription and hasattr(self._transcription, 'preconnect'):
                self._transcription_preconnect_task = asyncio.create_task(self._transcription.preconnect())
            
            # Play audio (will be interrupted if barge-in or termination triggers)
            print("🔊 Speaking...")
            
            # Run playback as task so we can interrupt for termination
            if use_streaming:
                # Streaming: synthesize and play in one step with lowest latency
                playback_task = asyncio.create_task(tts.synthesize_and_play_streaming(text))
            elif use_chunked:
                playback_task = asyncio.create_task(tts.synthesize_and_play_chunked(text))
            else:
                playback_task = asyncio.create_task(tts.play_audio_async(audio))
            
            # Poll for termination during playback
            termination_triggered = False
            while not playback_task.done():
                if self._termination_detected:
                    print("🛑 Termination detected during TTS - cutting audio")
                    termination_triggered = True
                    # Immediately stop audio playback
                    if tts and hasattr(tts, 'stop_audio'):
                        tts.stop_audio()
                    playback_task.cancel()
                    try:
                        await playback_task
                    except asyncio.CancelledError:
                        pass
                    break
                # Also check barge-in flag (set by callback)
                if self._barge_in_triggered:
                    break
                await asyncio.sleep(self._termination_poll_interval)
            
            # Wait for playback to finish if not cancelled
            if not termination_triggered and not self._barge_in_triggered:
                try:
                    await playback_task
                except asyncio.CancelledError:
                    pass
            
            # Check if barge-in or termination occurred
            if termination_triggered:
                print("🛑 Speech terminated by user command")
                barge_in_occurred = True  # Treat same as barge-in for flow control
            elif self._barge_in_triggered:
                print("⚡ Speech interrupted by barge-in")
                barge_in_occurred = True
                # Save the REMAINING response for potential recovery (resume from where we left off)
                if self._barge_in_recovery_enabled:
                    remaining_text = self._estimate_remaining_text(text)
                    if remaining_text:
                        self._interrupted_tts_response = remaining_text
                        self._barge_in_recovery_mode = True
                        print(f"💾 Saved remaining response for recovery: ~{len(remaining_text.split())} words")
                    else:
                        # Nearly complete, no need to recover
                        self._interrupted_tts_response = None
                        self._barge_in_recovery_mode = False
                        print("💾 Response was nearly complete, no recovery needed")
            else:
                print("✅ Speech complete")
                # Log actual speech rate for calibration
                if self._playback_start_time:
                    actual_duration = asyncio.get_event_loop().time() - self._playback_start_time
                    word_count = len(text.split())
                    if actual_duration > 0:
                        actual_wps = word_count / actual_duration
                        print(f"📊 TTS stats: {word_count} words in {actual_duration:.1f}s = {actual_wps:.2f} wps (estimate uses {self._tts_words_per_second} wps)")
                # Clear any saved response on normal completion
                self._interrupted_tts_response = None
                self._barge_in_recovery_mode = False
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            error = ComponentError(
                component="tts",
                severity=ErrorSeverity.RECOVERABLE,
                message="TTS error",
                exception=e
            )
            await self.error_handler.handle_error(error)
        
        finally:
            # Stop barge-in detector and capture audio if triggered
            if self._barge_in_detector:
                # Get captured audio before stopping
                if barge_in_occurred:
                    self._barge_in_audio = self._barge_in_detector.get_captured_audio_bytes()
                    if self._barge_in_audio:
                        duration = len(self._barge_in_audio) / 2 / 16000  # Assuming 16kHz
                        print(f"📼 Captured {duration:.2f}s of barge-in audio for transcription")
                
                await self._barge_in_detector.stop()
                self._barge_in_detector = None
            
            # Transition based on whether barge-in occurred
            if barge_in_occurred:
                # Don't transition - caller will handle transition to transcription
                # Record barge-in event for diagnostics
                self._record_timeline_event("barge_in", {
                    "early": self._early_barge_in,
                    "conversation_id": self._get_conversation_id(),
                })
            elif transition_to_idle:
                # Normal completion - go to IDLE
                await self._transition(
                    AudioState.IDLE,
                    component=None,
                    ctx=TransitionContext(
                        reason=TransitionReason.TTS_COMPLETE,
                        initiated_by="system",
                        conversation_id=self._get_conversation_id(),
                    )
                )
        
        return not barge_in_occurred
    
    async def run_response_with_streaming_tts(self, user_message: str, enable_barge_in: bool = True) -> tuple[str, bool]:
        """
        EXPERIMENTAL: Generate response and stream TTS simultaneously.
        
        Starts speaking as soon as the first sentence is ready, while
        continuing to generate the rest of the response.
        
        Args:
            user_message: User's message to respond to
            enable_barge_in: Allow user to interrupt speech
            
        Returns:
            Tuple of (full_response_text, was_interrupted)
        """
        self._barge_in_triggered = False
        self._last_tool_calls = []
        full_response = ""
        
        try:
            # Ensure transcription is stopped
            if self._transcription and self._transcription.is_active:
                print("🔄 Ensuring transcription fully stopped...")
                await self._transcription.stop_streaming()
                await asyncio.sleep(self._transcription_stop_delay)
            
            # Transition to response state
            await self._transition(
                AudioState.PROCESSING_RESPONSE,
                component="response",
                ctx=TransitionContext(
                    reason=TransitionReason.RESPONSE_START,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                    user_message_snippet=user_message[:50] if user_message else None,
                    extra={"streaming_tts": True},
                )
            )
            
            # Add user message to context
            if self._context:
                self._context.add_message("user", user_message)
                self._context.auto_trim_if_needed()
            
            response = self._response
            tts = self._tts
            
            if not response or not tts:
                print("❌ Response or TTS provider not available")
                return "", False
            
            # Start vector memory query in parallel
            vector_task = None
            if self._context and hasattr(self._context, 'get_vector_memory_context'):
                query_for_vector = getattr(self, '_vector_query_override', None) or user_message
                vector_task = asyncio.create_task(
                    self._context.get_vector_memory_context(query_for_vector)
                )
            
            # Get unified context bundle (single pass over conversation history)
            context = None
            tool_context = None
            if self._context:
                if hasattr(self._context, 'get_context_bundle'):
                    bundle = self._context.get_context_bundle()
                    context = bundle.response_context
                    tool_context = bundle.tool_context
                else:
                    context = self._context.get_recent_for_response()
                    tool_context = self._context.get_tool_context()
            
            # Wait for vector memory query (parallelized with context prep above)
            vector_context = ""
            if vector_task:
                vector_context = await vector_task or ""
                if vector_context and context:
                    # Inject vector memory as a system message in context
                    context.insert(1, {"role": "system", "content": vector_context})
            
            # Record comprehensive API input tokens (system prompt + context + tools)
            if self._recorder and self._recorder.current_session_id:
                system_prompt = ""
                if context:
                    for msg in context:
                        if msg.get("role") == "system":
                            system_prompt += msg.get("content", "") + "\n"
                
                tool_defs = []
                if hasattr(response, 'openai_functions'):
                    tool_defs = response.openai_functions or []
                
                self._recorder.record_api_request(
                    system_prompt=system_prompt,
                    context_messages=context or [],
                    tool_definitions=tool_defs,
                    user_message=user_message,
                    vector_context=vector_context
                )
            
            # Prepare barge-in detection
            barge_in_callback = None
            barge_in_started_during_processing = False
            if enable_barge_in and self._barge_in_enabled:
                bt_mode = " [Bluetooth/Meta]" if self._barge_in_is_bluetooth else ""
                
                # Determine cooldown: shorter during processing (no TTS feedback), longer during TTS
                initial_cooldown = (
                    self._barge_in_processing_cooldown 
                    if self._barge_in_during_processing 
                    else self._barge_in_cooldown_after_tts_start
                )
                
                print(f"👂 Barge-in will be enabled{bt_mode}")
                self._barge_in_detector = BargeInDetector(
                    config=BargeInConfig(
                        mode=BargeInMode.ENERGY,
                        energy_threshold=self._barge_in_energy_threshold,
                        min_speech_duration=self._barge_in_min_speech_duration,
                        cooldown_after_tts_start=initial_cooldown,
                        sample_rate=self._barge_in_sample_rate,
                        chunk_size=self._barge_in_chunk_size,
                        buffer_seconds=self._barge_in_buffer_seconds,
                        capture_after_trigger=self._barge_in_capture_after_trigger,
                        device_index=self._barge_in_device_index,
                        latency=self._barge_in_latency,
                        is_bluetooth=self._barge_in_is_bluetooth,
                        require_consecutive_speech=self._barge_in_require_consecutive,
                        silence_reset_threshold=self._barge_in_silence_reset_threshold,
                    ),
                    shared_bus=self._shared_audio_bus,
                )
                print(
                    "🔧 Barge-in config: "
                    f"threshold={self._barge_in_energy_threshold}, "
                    f"min_speech={self._barge_in_min_speech_duration}s, "
                    f"cooldown={initial_cooldown}s"
                )
                
                def on_barge_in():
                    print("🎤 BARGE-IN: Interrupting!")
                    self._barge_in_triggered = True
                    
                    # Check if this is an early barge-in (within threshold)
                    # Only applies if playback has started
                    if self._playback_start_time:
                        elapsed = asyncio.get_event_loop().time() - self._playback_start_time
                        if elapsed < self._early_barge_in_threshold:
                            self._early_barge_in = True
                            print(f"⚡ Early barge-in ({elapsed:.1f}s < {self._early_barge_in_threshold}s) - will append next message")
                        else:
                            print(f"⏱️  Late barge-in ({elapsed:.1f}s >= {self._early_barge_in_threshold}s) - new message")
                    else:
                        # Barge-in during processing (before TTS started)
                        self._early_barge_in = True
                        print("⚡ Barge-in during response generation - will append next message")
                    
                    if tts and hasattr(tts, 'stop_audio'):
                        tts.stop_audio()
                
                barge_in_callback = on_barge_in
                
                # START barge-in immediately if enabled during processing phase
                if self._barge_in_during_processing:
                    await self._barge_in_detector.start(on_barge_in=barge_in_callback)
                    barge_in_started_during_processing = True
                    print(f"👂 Barge-in detection started during processing (cooldown: {initial_cooldown}s)")
            
            # Pre-connect transcription WebSocket
            if self._transcription and hasattr(self._transcription, 'preconnect'):
                self._transcription_preconnect_task = asyncio.create_task(self._transcription.preconnect())
            
            # Create async generator that yields response chunks
            streamed_any = False
            
            async def response_text_generator():
                nonlocal full_response, streamed_any
                print("💭 Generating response (streaming to TTS)...")
                log_user_message(user_message)
                
                chunk_count = 0
                async for chunk in response.stream_response(user_message, context=context, tool_context=tool_context):
                    chunk_count += 1
                    print(f"[DEBUG] Chunk {chunk_count}: is_complete={chunk.is_complete}, content_len={len(chunk.content) if chunk.content else 0}, has_tools={bool(chunk.tool_calls)}")
                    
                    # Check for barge-in or termination
                    if self._barge_in_triggered or self._termination_detected:
                        if self._termination_detected:
                            print("\n🛑 Termination detected during response generation")
                            # Stop audio immediately
                            if tts and hasattr(tts, 'stop_audio'):
                                tts.stop_audio()
                        break
                    
                    if chunk.content:
                        if chunk.is_complete:
                            # Final chunk - may have tool calls
                            full_response = chunk.content
                            if chunk.tool_calls:
                                self._last_tool_calls = chunk.tool_calls
                                for tc in chunk.tool_calls:
                                    tool_name = getattr(tc, 'name', None) or 'unknown'
                                    log_tool_call(tool_name)
                            
                            # If we haven't streamed any deltas yet, yield the complete response
                            # This handles providers that return everything in one chunk
                            if not streamed_any:
                                print(f"📝 Complete response received ({len(chunk.content)} chars), streaming to TTS...")
                                yield chunk.content
                                streamed_any = True
                        else:
                            # Streaming delta
                            full_response += chunk.content
                            print(chunk.content, end="", flush=True)
                            yield chunk.content
                            streamed_any = True
                    else:
                        print(f"[DEBUG] Chunk {chunk_count} has no content!")
                
                print(f"\n[DEBUG] Generator complete. Total chunks: {chunk_count}, streamed_any: {streamed_any}")
            
            # Check if barge-in occurred during processing (before TTS starts)
            if self._barge_in_triggered:
                print("⚡ Barge-in triggered during response generation - skipping TTS")
                # Capture audio for transcription prefill
                if self._barge_in_detector:
                    self._barge_in_audio = self._barge_in_detector.get_captured_audio_bytes()
                    if self._barge_in_audio:
                        duration = len(self._barge_in_audio) / 2 / 16000
                        print(f"📼 Captured {duration:.2f}s of barge-in audio")
                    await self._barge_in_detector.stop()
                # Add partial response to context if any was generated
                if self._context and full_response:
                    self._context.add_message("assistant", full_response + " [interrupted]")
                return full_response, True  # was_interrupted=True
            
            # Transition to TTS state
            await self._transition(
                AudioState.SYNTHESIZING,
                component="tts",
                ctx=TransitionContext(
                    reason=TransitionReason.TTS_START,
                    initiated_by="system",
                    conversation_id=self._get_conversation_id(),
                    extra={"streaming_tts": True},
                )
            )
            
            # Handle barge-in detection for TTS phase
            if self._barge_in_detector and barge_in_callback:
                if barge_in_started_during_processing:
                    # Barge-in already running - just mark playback start time
                    self._playback_start_time = asyncio.get_event_loop().time()
                    print("👂 Barge-in detection continuing for TTS playback")
                else:
                    # Start barge-in now (fallback when processing barge-in disabled)
                    await self._barge_in_detector.start(on_barge_in=barge_in_callback)
                    self._playback_start_time = asyncio.get_event_loop().time()
                    print(f"👂 Barge-in detection started (cooldown: {self._barge_in_cooldown_after_tts_start}s)")
            
            # Stream TTS with sentence buffering
            print("🔊 Starting streaming TTS...")
            
            def on_sentence_start(num, text):
                print(f"🎤 Sentence {num}: {text[:30]}...")
            
            def on_complete(interrupted):
                if interrupted:
                    print("⚡ Streaming TTS interrupted")
                else:
                    print("✅ Streaming TTS complete")
            
            # Use streaming TTS if available
            if hasattr(tts, 'speak_streaming'):
                speech_completed = await tts.speak_streaming(
                    response_text_generator(),
                    on_sentence_start=on_sentence_start,
                    on_complete=on_complete
                )
            else:
                # Fallback: collect all text then speak normally
                print("⚠️  TTS doesn't support streaming, falling back to batch mode")
                async for _ in response_text_generator():
                    if self._termination_detected:
                        break
                if full_response and not self._termination_detected:
                    audio = await tts.synthesize(full_response)
                    # Run playback as task for termination interruption
                    playback_task = asyncio.create_task(tts.play_audio_async(audio))
                    while not playback_task.done():
                        if self._termination_detected:
                            print("🛑 Termination detected during fallback TTS")
                            if tts and hasattr(tts, 'stop_audio'):
                                tts.stop_audio()
                            playback_task.cancel()
                            try:
                                await playback_task
                            except asyncio.CancelledError:
                                pass
                            break
                        await asyncio.sleep(self._termination_poll_interval)
                    if not self._termination_detected:
                        try:
                            await playback_task
                        except asyncio.CancelledError:
                            pass
                speech_completed = not self._barge_in_triggered and not self._termination_detected
            
            # Capture composition API token usage (from tool decisions and final answer composition)
            if self._recorder and hasattr(response, 'get_composition_token_usage'):
                comp_usage = response.get_composition_token_usage()
                if comp_usage["total"] > 0:
                    self._recorder._session_input_tokens += comp_usage["input_tokens"]
                    self._recorder._session_output_tokens += comp_usage["output_tokens"]
                    print(f"📊 Composition totals: +{comp_usage['input_tokens']:,} in, +{comp_usage['output_tokens']:,} out")
            
            # Add response to context
            if self._context and full_response:
                self._context.add_message("assistant", full_response)
            
            if full_response:
                log_assistant_response(full_response)
            
            return full_response, not speech_completed
            
        except Exception as e:
            print(f"❌ Streaming response error: {e}")
            import traceback
            traceback.print_exc()
            return full_response, False
            
        finally:
            # Cleanup barge-in
            if self._barge_in_detector:
                if self._barge_in_triggered:
                    self._barge_in_audio = self._barge_in_detector.get_captured_audio_bytes()
                    if self._barge_in_audio:
                        duration = len(self._barge_in_audio) / 2 / 16000
                        print(f"📼 Captured {duration:.2f}s of barge-in audio")
                
                await self._barge_in_detector.stop()
                self._barge_in_detector = None
    
    async def run_full_conversation(self):
        """Run complete conversation pipeline with barge-in support."""
        try:
            # 1. Wait for wake word
            wake_model = None
            async for wake_event in self.run_wake_word_detection():
                print(f"\n🎯 Wake word detected: {wake_event.model_name}\n")
                wake_model = wake_event.model_name
                break  # Got wake word, proceed
            
            # Start pre-connecting transcription BEFORE stopping wake word (parallel operation)
            preconnect_task = None
            if self._transcription and hasattr(self._transcription, 'preconnect'):
                preconnect_task = asyncio.create_task(self._transcription.preconnect())
            
            # Explicitly stop wake word detection before proceeding
            # The async generator break doesn't stop the subprocess
            if self._wakeword:
                await self._wakeword.stop_detection()
                # Wait for subprocess to fully terminate (reduced from 0.5s)
                await asyncio.sleep(0.1)
            
            # Ensure preconnect completed (overlapped with wake word stop)
            if preconnect_task:
                try:
                    await asyncio.wait_for(preconnect_task, timeout=0.5)
                except asyncio.TimeoutError:
                    pass  # Will connect on-demand
            
            # Start recording session and reset conversation context
            if self._recorder and self._recorder.is_initialized:
                await self._recorder.start_session(wake_word_model=wake_model)
            if self._context:
                self._context.reset()
                print("🧠 Conversation context reset for new session")
            
            # 2. Transcribe user speech
            user_text = await self.run_transcription()
            
            # Handle barge-in recovery: resume TTS if no speech detected
            if user_text == "__RESUME_TTS__" and self._interrupted_tts_response:
                print("🔄 Resuming interrupted TTS response...")
                response_to_resume = self._interrupted_tts_response
                self._interrupted_tts_response = None
                
                # Transition out of TRANSCRIBING before TTS
                if self.state_machine.current_state == AudioState.TRANSCRIBING:
                    await self._transition(
                        AudioState.IDLE,
                        component=None,
                        ctx=TransitionContext(
                            reason=TransitionReason.TRANSCRIPTION_COMPLETE,
                            initiated_by="recovery",
                            conversation_id=self._get_conversation_id(),
                        )
                    )
                
                await self.run_tts(response_to_resume, transition_to_idle=True, enable_barge_in=True)
                if self._recorder and self._recorder.current_session_id:
                    await self._recorder.end_session()
                return
            
            if not user_text:
                print("⚠️  No transcription received")
                if self._recorder and self._recorder.current_session_id:
                    await self._recorder.end_session()
                return
            
            print(f"\n👤 User: {user_text}\n")
            
            # Record user message
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.record_message("user", user_text)
            
            # 3. Generate response
            assistant_text = await self.run_response(user_text)
            if not assistant_text:
                print("⚠️  No response generated")
                if self._recorder and self._recorder.current_session_id:
                    await self._recorder.end_session()
                return
            
            print(f"\n🤖 Assistant: {assistant_text}\n")
            
            # Record assistant message
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.record_message("assistant", assistant_text)
                # Record tool calls used to generate this response
                await self._record_tool_calls()
            
            # 4. Speak response with barge-in support
            speech_completed = await self.run_tts(assistant_text, transition_to_idle=True, enable_barge_in=True)
            
            # End recording session
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.end_session()
            
            # Update persistent memory with conversation learnings
            if self._context and hasattr(self._context, 'on_conversation_end'):
                self._context.on_conversation_end()
            
            if speech_completed:
                print("\n✅ Conversation complete\n")
            else:
                print("\n⚡ Speech interrupted - ready for follow-up\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.end_session(metadata={"ended_reason": "keyboard_interrupt"})
            await self.state_machine.emergency_reset()
        except Exception as e:
            print(f"\n❌ Conversation error: {e}")
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.end_session(metadata={"ended_reason": "error", "error": str(e)})
            await self.state_machine.emergency_reset()

    def _get_primary_user(self) -> str:
        """
        Best-effort resolve primary user for briefings retrieval.
        Falls back to 'Morgan' if not configured.
        """
        try:
            user_state = self.config.get("user_state", {})
            primary_user = user_state.get("primary_user")
            if primary_user:
                return primary_user
        except Exception:
            pass
        return os.getenv("EMAIL_NOTIFICATION_RECIPIENT", "Morgan")

    async def _brief_pending_announcements(self, user: str) -> None:
        """
        Fetch pending briefings and proactively announce to the user before conversation starts.
        
        Uses pre-generated openers (TTS only, no LLM latency) if available.
        Falls back to LLM generation if briefings don't have openers yet.
        Marks delivered after speaking.
        """
        if not self._briefing_manager:
            return

        # First, try to get briefings with pre-generated openers (fast path - TTS only)
        try:
            briefings_with_opener = await self._briefing_manager.get_pending_briefings_with_opener(user=user)
        except Exception:
            briefings_with_opener = []

        if briefings_with_opener:
            # Fast path: use pre-generated opener (no LLM call needed)
            opener = self._briefing_manager.get_combined_opener(briefings_with_opener)
            if opener:
                print(f"📢 Speaking pre-generated opener for {len(briefings_with_opener)} briefing(s)")
                await self.run_tts(opener, transition_to_idle=False, enable_barge_in=False)
                
                # Inject briefing info into conversation context so follow-ups have context
                if self._context:
                    # Extract raw messages from briefings for context
                    raw_messages = []
                    for briefing in briefings_with_opener:
                        content = briefing.get("content") or {}
                        if isinstance(content, str):
                            try:
                                import json
                                content = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                content = {"message": content}
                        message = content.get("message") or content.get("fact", "")
                        if message:
                            raw_messages.append(message)
                    
                    # Add system note about what was briefed
                    if raw_messages:
                        briefing_summary = "Briefings delivered: " + "; ".join(raw_messages)
                        self._context.add_message("system", f"[Context: {briefing_summary}]")
                    # Add the assistant's spoken response
                    self._context.add_message("assistant", opener)
                
                ids = [b.get("id") for b in briefings_with_opener if b.get("id")]
                try:
                    await self._briefing_manager.mark_delivered(ids)
                except Exception:
                    pass
                return

        # Fallback: check for briefings without openers (need LLM generation)
        try:
            pending = await self._briefing_manager.get_pending_briefings(user=user)
        except Exception:
            return

        pending = [p for p in pending if p]
        if not pending:
            return

        # Generate opener via LLM (slower path, for backwards compatibility)
        # Skip vector memory and conversation context - openers should be self-contained
        print("⚠️  Briefings without pre-generated openers, falling back to LLM generation")
        
        briefing_lines: List[str] = []
        raw_messages: List[str] = []  # Store raw briefing messages for context injection
        for idx, briefing in enumerate(pending, 1):
            content = briefing.get("content") or {}
            # Handle content as string (JSON) or dict
            if isinstance(content, str):
                try:
                    import json
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {"message": content}
            meta = content.get("meta") or {}
            instructions = content.get("llm_instructions")

            meta_bits = []
            ts = meta.get("timestamp")
            src = meta.get("source")
            if ts:
                meta_bits.append(f"time: {ts}")
            if src:
                meta_bits.append(f"source: {src}")
            meta_str = f" ({'; '.join(meta_bits)})" if meta_bits else ""

            # Support both 'message' and legacy 'fact' key
            message = content.get('message') or content.get('fact', '')
            raw_messages.append(message)
            line = f"{idx}. {message}{meta_str}"
            if instructions:
                line += f" [llm_instructions: {instructions}]"
            briefing_lines.append(line)

        # Embed guidance in the prompt itself
        briefing_prompt = (
            "[INSTRUCTION: You are proactively briefing the user on pending announcements before taking requests. "
            "Be concise, friendly, and avoid over-explaining. Offer to handle or dismiss items if appropriate.]\n\n"
            "Pending briefings to announce:\n" + "\n".join(briefing_lines)
        )

        response = await self.run_response(briefing_prompt, skip_vector_memory=True, skip_conversation_context=True)
        if response:
            await self.run_tts(response, transition_to_idle=False, enable_barge_in=False)
            
            # Inject briefing info into conversation context so follow-ups have context
            if self._context:
                # Add a system note about what was briefed
                briefing_summary = "Briefings delivered: " + "; ".join(raw_messages)
                self._context.add_message("system", f"[Context: {briefing_summary}]")
                # Add the assistant's spoken response
                self._context.add_message("assistant", response)
            
            ids = [b.get("id") for b in pending if b.get("id")]
            try:
                await self._briefing_manager.mark_delivered(ids)
            except Exception:
                pass

    async def run_continuous_loop(self):
        """Run continuous conversation loop with multi-turn support and barge-in."""
        print("🔁 Starting continuous conversation loop...")
        print("   Press Ctrl+C to stop")
        print("   💡 You can interrupt the assistant by speaking!\n")
        
        try:
            while True:
                # 1. Wait for wake word to start conversation
                # Start speculative pre-warming tasks during idle (before wake word)
                prewarm_tasks = []
                
                # Pre-warm OpenAI WebSocket connection
                if self._response and hasattr(self._response, 'ensure_ws_warm'):
                    prewarm_tasks.append(asyncio.create_task(self._response.ensure_ws_warm()))
                
                # Pre-warm vector memory cache
                if self._context and hasattr(self._context, 'preload_vector_cache'):
                    prewarm_tasks.append(asyncio.create_task(self._context.preload_vector_cache()))
                
                wake_model = None
                async for wake_event in self.run_wake_word_detection():
                    print(f"\n🎯 Wake word detected: {wake_event.model_name}\n")
                    wake_model = wake_event.model_name
                    break  # Got wake word, enter conversation mode
                
                # Cancel any remaining pre-warm tasks (they should be done by now)
                for task in prewarm_tasks:
                    if not task.done():
                        task.cancel()
                
                # OPTIMISTIC PARALLEL STARTUP: Start multiple operations simultaneously
                # 1. Pre-connect transcription WebSocket
                # 2. Stop/pause wake word (fire-and-forget style)
                # 3. Both happen in parallel to minimize transition time
                
                preconnect_task = None
                if self._transcription and hasattr(self._transcription, 'preconnect'):
                    preconnect_task = asyncio.create_task(self._transcription.preconnect())
                
                # Fire-and-forget wake word pause (don't await, let it run in background)
                wake_word_task = None
                if self._wakeword:
                    if self._wake_word_warm_mode and hasattr(self._wakeword, 'pause_detection'):
                        wake_word_task = asyncio.create_task(self._wakeword.pause_detection())
                    else:
                        wake_word_task = asyncio.create_task(self._wakeword.stop_detection())
                
                # Short overlap window - let both start, then ensure wake word is done
                await asyncio.sleep(0.03)  # 30ms head start for parallel operations
                
                # Now wait for wake word to finish (should be almost done)
                if wake_word_task:
                    try:
                        await asyncio.wait_for(wake_word_task, timeout=0.3)
                    except asyncio.TimeoutError:
                        print("⚠️  Wake word pause timed out (continuing)")
                
                # Configurable delay (default 0 in warm mode)
                if self._wake_word_stop_delay > 0:
                    await asyncio.sleep(self._wake_word_stop_delay)
                
                # Ensure preconnect completed (overlapped with wake word stop)
                if preconnect_task:
                    try:
                        await asyncio.wait_for(preconnect_task, timeout=0.3)
                    except asyncio.TimeoutError:
                        pass  # Will connect on-demand
                
                # Start shared audio bus for conversation (zero-latency transitions)
                if self._shared_audio_bus:
                    await self._shared_audio_bus.start()
                
                # Start recording session and reset conversation context
                if self._recorder and self._recorder.is_initialized:
                    await self._recorder.start_session(wake_word_model=wake_model)
                if self._context:
                    self._context.reset()
                    print("🧠 Conversation context reset for new session")
                
                # Clear any stale barge-in state from previous conversation
                self._barge_in_audio = None
                self._barge_in_triggered = False

                # Proactively announce pending briefings
                # If briefing_wake_words is configured, only announce for those wake words
                # If not configured (empty), always announce briefings
                briefing_wake_words = self.config.get('wakeword', {}).get('config', {}).get('briefing_wake_words', [])
                
                if briefing_wake_words:
                    # Selective mode: only announce for specific wake words
                    should_brief = wake_model in briefing_wake_words
                    if should_brief:
                        print(f"📢 Briefing wake word detected: {wake_model}")
                    else:
                        print(f"💬 Standard wake word detected: {wake_model} (skipping briefings)")
                else:
                    # Default mode: always announce briefings
                    should_brief = True
                
                if should_brief:
                    primary_user = self._get_primary_user()
                    await self._brief_pending_announcements(primary_user)

                # 2. Enter multi-turn conversation mode
                # Start parallel termination detection (listens for "over out" etc.)
                await self._start_termination_detection()
                
                # TTS announcement for conversation start (runs in separate thread)
                if ENABLE_TTS_ANNOUNCEMENTS and self._tts:
                    announce_conversation_start(self._tts)
                
                print("💬 Conversation mode active (say termination phrase to exit)")
                conversation_active = True
                
                while conversation_active:
                    # Check if termination phrase was detected in parallel
                    if self._termination_detected:
                        print("🛑 Termination phrase detected - ending conversation")
                        conversation_active = False
                        break
                    
                    # Update termination detector with current state
                    self._update_termination_state("TRANSCRIBING")
                    
                    # Transcribe user speech with termination interrupt support
                    # Run transcription as task so we can cancel it if termination detected
                    # Use extended timeout if last response used tool calls (gives user more time to think)
                    use_extended_timeout = len(self._last_tool_calls) > 0
                    transcription_task = asyncio.create_task(self.run_transcription(use_extended_timeout=use_extended_timeout))
                    
                    # Poll for termination while transcription is running
                    user_text = None
                    while not transcription_task.done():
                        if self._termination_detected:
                            print("🛑 Termination detected during transcription - cancelling")
                            transcription_task.cancel()
                            try:
                                await transcription_task
                            except asyncio.CancelledError:
                                pass
                            # Stop transcription provider immediately
                            if self._transcription and hasattr(self._transcription, 'stop_streaming'):
                                try:
                                    await self._transcription.stop_streaming()
                                except Exception:
                                    pass  # Don't let cleanup errors block termination
                            conversation_active = False
                            break
                        # Short sleep to avoid busy-waiting (configurable)
                        await asyncio.sleep(self._termination_poll_interval)
                    
                    # If terminated during transcription, break outer loop
                    if not conversation_active:
                        break
                    
                    # Get transcription result
                    try:
                        user_text = transcription_task.result()
                    except asyncio.CancelledError:
                        user_text = None
                    
                    # Handle barge-in recovery: resume TTS if no speech detected
                    if user_text == "__RESUME_TTS__" and self._interrupted_tts_response:
                        print("🔄 Resuming interrupted TTS response...")
                        response_to_resume = self._interrupted_tts_response
                        self._interrupted_tts_response = None  # Clear saved response
                        
                        # Transition out of TRANSCRIBING before TTS
                        if self.state_machine.current_state == AudioState.TRANSCRIBING:
                            await self._transition(
                                AudioState.IDLE,
                                component=None,
                                ctx=TransitionContext(
                                    reason=TransitionReason.TRANSCRIPTION_COMPLETE,
                                    initiated_by="recovery",
                                    conversation_id=self._get_conversation_id(),
                                )
                            )
                        
                        # Resume TTS with the interrupted response
                        speech_completed = await self.run_tts(
                            response_to_resume, 
                            transition_to_idle=False,  # Don't go to idle, stay in conversation
                            enable_barge_in=True  # Allow barge-in again
                        )
                        
                        if speech_completed:
                            print("✅ Resumed TTS completed")
                        # Continue conversation loop (go back to wake word or ready for next input)
                        continue
                    
                    # Check if user wants to end conversation
                    if not user_text or user_text == "__RESUME_TTS__":
                        if user_text == "__RESUME_TTS__":
                            print("⚠️  Barge-in recovery triggered but no saved response")
                        else:
                            print("⚠️  No transcription received, ending conversation")
                        conversation_active = False
                        break
                    
                    # Handle early barge-in: append to previous message
                    # Store the raw new message BEFORE combining (for next potential append)
                    raw_new_message = user_text
                    
                    if self._early_barge_in and self._previous_user_message:
                        combined_text = f"{self._previous_user_message} {user_text}"
                        print("🔗 Early barge-in: appending to previous message")
                        print(f"   Previous: \"{self._previous_user_message}\"")
                        print(f"   Added: \"{user_text}\"")
                        user_text = combined_text
                    
                    # Always reset early barge-in flag after processing
                    self._early_barge_in = False
                    
                    # Store only the RAW new message (not combined) for potential early barge-in append
                    # This prevents accumulation: A, then A+B, then A+B+C...
                    self._previous_user_message = raw_new_message
                    
                    # Set vector query to use ONLY the raw new message (not combined)
                    # This ensures vector search is always for the single most recent utterance
                    self._vector_query_override = raw_new_message
                    
                    print(f"\n👤 User: {user_text}\n")
                    
                    # Record user message
                    if self._recorder and self._recorder.current_session_id:
                        await self._recorder.record_message("user", user_text)
                    
                    # Generate response and speak it
                    # Update termination detector with state for better metadata
                    self._update_termination_state("PROCESSING_RESPONSE")
                    
                    if self._streaming_tts_enabled:
                        # EXPERIMENTAL: Stream TTS - start speaking before response is complete
                        print("⚡ Using streaming TTS mode")
                        assistant_text, barge_in_occurred = await self.run_response_with_streaming_tts(
                            user_text,
                            enable_barge_in=True
                        )
                        speech_completed = not barge_in_occurred
                        
                        # Check for termination during response/TTS
                        if self._termination_detected:
                            print("🛑 Termination detected during streaming response")
                            conversation_active = False
                            break
                        
                        if not assistant_text:
                            print("⚠️  No response generated")
                            # Transition to IDLE before continuing (state may be stuck in PROCESSING_RESPONSE/SYNTHESIZING)
                            if self.state_machine.current_state != AudioState.IDLE:
                                await self._transition(
                                    AudioState.IDLE,
                                    component=None,
                                    ctx=TransitionContext(
                                        reason=TransitionReason.RESPONSE_NO_OUTPUT,
                                        initiated_by="system",
                                        conversation_id=self._get_conversation_id(),
                                        extra={"streaming_tts": True},
                                    )
                                )
                            continue
                        
                        print(f"\n🤖 Assistant: {assistant_text}\n")
                        
                        # Record assistant message
                        if self._recorder and self._recorder.current_session_id:
                            await self._recorder.record_message("assistant", assistant_text)
                            # Record tool calls used to generate this response
                            await self._record_tool_calls()
                    else:
                        # Traditional mode: generate full response, then speak
                        assistant_text = await self.run_response(user_text, enable_barge_in=True)
                        
                        # Check for termination during response generation
                        if self._termination_detected:
                            print("🛑 Termination detected during response generation")
                            conversation_active = False
                            break
                        
                        # Check if response was interrupted by barge-in
                        if self._barge_in_triggered and not assistant_text:
                            print("⚡ Response interrupted by barge-in - skipping TTS")
                            # User spoke during response generation
                            # Mark as barge-in so we continue to transcription
                            speech_completed = False
                            self._barge_in_triggered = False  # Reset flag
                            # Transition to IDLE before transcription
                            if self.state_machine.current_state != AudioState.IDLE:
                                await self._transition(
                                    AudioState.IDLE,
                                    component=None,
                                    ctx=TransitionContext(
                                        reason=TransitionReason.TTS_BARGE_IN,
                                        initiated_by="user",
                                        conversation_id=self._get_conversation_id(),
                                        extra={"interrupted_during": "response_generation"},
                                    )
                                )
                            # Skip to barge-in handling (will start transcription with prefill)
                        elif not assistant_text:
                            print("⚠️  No response generated")
                            # Transition to IDLE before continuing (state may be stuck in PROCESSING_RESPONSE)
                            if self.state_machine.current_state != AudioState.IDLE:
                                await self._transition(
                                    AudioState.IDLE,
                                    component=None,
                                    ctx=TransitionContext(
                                        reason=TransitionReason.RESPONSE_NO_OUTPUT,
                                        initiated_by="system",
                                        conversation_id=self._get_conversation_id(),
                                    )
                                )
                            continue  # Try next question
                        else:
                            print(f"\n🤖 Assistant: {assistant_text}\n")
                            
                            # Record assistant message
                            if self._recorder and self._recorder.current_session_id:
                                await self._recorder.record_message("assistant", assistant_text)
                                # Record tool calls used to generate this response
                                await self._record_tool_calls()
                            
                            # Update state before TTS
                            self._update_termination_state("SYNTHESIZING")
                            
                            # Speak response with barge-in enabled
                            # If user interrupts, we'll immediately start transcribing
                            speech_completed = await self.run_tts(
                                assistant_text, 
                                transition_to_idle=False,  # Don't auto-transition, we handle it
                                enable_barge_in=True
                            )
                            
                            # Check for termination during TTS
                            if self._termination_detected:
                                print("🛑 Termination detected during TTS")
                                conversation_active = False
                                break
                    
                    if speech_completed:
                        # Normal completion - transition to IDLE then back to transcription
                        await self._transition(
                            AudioState.IDLE,
                            component=None,
                            ctx=TransitionContext(
                                reason=TransitionReason.TTS_COMPLETE,
                                initiated_by="system",
                                conversation_id=self._get_conversation_id(),
                            )
                        )
                        print("🎤 Ready for next question (or say termination phrase)...\n")
                        # Note: Ready beep now handled by state machine transition
                        # Clear previous message since response completed without barge-in
                        self._previous_user_message = ""
                        self._vector_query_override = None
                    else:
                        # Barge-in occurred! Skip IDLE and go directly to transcription
                        # The captured audio will be prefilled to transcription
                        if self._barge_in_audio:
                            duration = len(self._barge_in_audio) / 2 / 16000
                            print(f"⚡ Barge-in: {duration:.2f}s ({len(self._barge_in_audio)} bytes) will be fed to transcription")
                        else:
                            print("⚡ Barge-in: No prefill audio available, going directly to transcription")
                        # Small delay for audio device handoff
                        await asyncio.sleep(self._barge_in_resume_delay)
                        print("🎤 Listening (continuing barge-in conversation)...\n")
                        # Continue the loop - next iteration will run transcription
                
                # Log if terminated by phrase detection (vs other reasons)
                was_terminated = self._termination_detected
                if was_terminated:
                    log_conversation_end()
                    self._termination_detected = False  # Reset for next conversation
                
                # End recording session and update memory (run in parallel for speed)
                async def _end_conversation_tasks():
                    if self._recorder and self._recorder.current_session_id:
                        await self._recorder.end_session()
                    if self._context and hasattr(self._context, 'on_conversation_end'):
                        self._context.on_conversation_end()
                
                # Run cleanup tasks in parallel with state transition
                # Also stop termination detection (fire-and-forget - don't await)
                cleanup_task = asyncio.create_task(_end_conversation_tasks())
                asyncio.create_task(self._stop_termination_detection(fire_and_forget=was_terminated))
                
                # Stop shared audio bus (conversation ended, wake word will use own stream)
                if self._shared_audio_bus:
                    await self._shared_audio_bus.stop()
                
                # TTS announcement for conversation end (plays for ALL conversation endings)
                if ENABLE_TTS_ANNOUNCEMENTS and self._tts:
                    announce_termination(self._tts)
                
                # Conversation ended - ensure we're in IDLE before restarting wake word
                print("✅ Conversation session ended\n")
                
                # Critical: Transition to IDLE if not already there
                # This prevents TRANSCRIBING → WAKE_WORD_LISTENING (invalid transition)
                if self.state_machine.current_state != AudioState.IDLE:
                    print("🔄 Transitioning to IDLE before restarting wake word...")
                    await self._transition(
                        AudioState.IDLE,
                        component=None,
                        ctx=TransitionContext(
                            reason=TransitionReason.SESSION_END,
                            initiated_by="system",
                            conversation_id=self._get_conversation_id(),
                            extra={"was_terminated": was_terminated},
                        )
                    )
                
                # Wait for cleanup tasks to complete
                await cleanup_task
                
                # Configurable settling time (default 0 with warm mode)
                if self._post_conversation_delay > 0:
                    print(f"⏳ Waiting {self._post_conversation_delay}s for audio device to settle...")
                    await asyncio.sleep(self._post_conversation_delay)
                
        except KeyboardInterrupt:
            print("\n⚠️  Stopping continuous loop...")
            if self._recorder and self._recorder.current_session_id:
                await self._recorder.end_session(metadata={"ended_reason": "keyboard_interrupt"})
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all resources."""
        print("🧹 Cleaning up orchestrator...")
        log_shutdown()  # 📡 Remote console log
        
        # FIRST: Immediately kill any audio processes (most important for user experience)
        import subprocess
        for proc_name in ["ffplay", "afplay"]:
            try:
                subprocess.run(["pkill", "-9", "-x", proc_name], capture_output=True, timeout=1)
            except Exception:
                pass
        
        # Stop shared audio bus if running
        if self._shared_audio_bus:
            try:
                await self._shared_audio_bus.stop()
            except Exception as e:
                print(f"⚠️  Shared audio bus cleanup error: {e}")
        
        # Cleanup barge-in detector if active
        if self._barge_in_detector:
            try:
                await self._barge_in_detector.stop()
            except Exception as e:
                print(f"⚠️  Barge-in cleanup error: {e}")
            self._barge_in_detector = None
        
        # Cleanup termination detector
        if self._termination:
            try:
                await self._termination.cleanup()
            except Exception as e:
                print(f"⚠️  Termination detector cleanup error: {e}")
            self._termination = None
        
        # Cleanup conversation recorder
        if self._recorder:
            try:
                await self._recorder.cleanup()
            except Exception as e:
                print(f"⚠️  Recorder cleanup error: {e}")
            self._recorder = None
        
        # Only cleanup providers if they're still active
        # (State machine may have already cleaned them up)
        if self._wakeword and getattr(self._wakeword, 'is_listening', False):
            await self._wakeword.cleanup()
        if self._transcription and getattr(self._transcription, 'is_active', False):
            await self._transcription.cleanup()
        if self._response:
            await self._response.cleanup()
        if self._tts:
            await self._tts.cleanup()
        
        # Only reset state machine if not already in IDLE
        current_state = self.state_machine.current_state
        if current_state != AudioState.IDLE:
            await self.state_machine.emergency_reset()
        else:
            print("ℹ️  Already in IDLE state, skipping emergency reset")
        
        print("✅ Cleanup complete")
    
    async def interrupt_tts(self):
        """Interrupt TTS playback if active."""
        if self._tts and hasattr(self._tts, 'is_playing') and self._tts.is_playing:
            print("⏸️  Interrupting TTS playback...")
            if hasattr(self._tts, 'stop_audio'):
                self._tts.stop_audio()
            # Transition back to IDLE
            await self._transition(
                AudioState.IDLE,
                component=None,
                ctx=TransitionContext(
                    reason=TransitionReason.MANUAL_RESET,
                    initiated_by="user",
                    conversation_id=self._get_conversation_id(),
                )
            )
            print("✅ TTS interrupted")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'initialized': self.is_initialized,
            'state': self.state_machine.get_status(),
            'errors': self.error_handler.get_error_summary(),
            'providers': {
                'wakeword': self._wakeword is not None,
                'transcription': self._transcription is not None,
                'response': self._response is not None,
                'tts': self._tts is not None
            },
            'recording': {
                'enabled': self._recording_enabled,
                'initialized': self._recorder.is_initialized if self._recorder else False,
                'current_session': self._recorder.current_session_id if self._recorder else None
            }
        }
