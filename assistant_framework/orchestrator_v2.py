"""
Refactored orchestrator with state machine integration.
Simplified, reliable pipeline execution.
"""

import asyncio
from typing import Dict, Any, Optional

try:
    from .interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        WakeWordInterface
    )
    from .models.data_models import TranscriptionResult, ResponseChunk, WakeWordEvent
    from .utils.state_machine import AudioStateMachine, AudioState
    from .utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity
    from .utils.barge_in import BargeInDetector, BargeInConfig, BargeInMode
    from .providers.wakeword_v2 import IsolatedOpenWakeWordProvider
    from .providers.transcription_v2 import AssemblyAIAsyncProvider
    from .config import get_active_preset
except ImportError:
    from assistant_framework.interfaces import (
        TranscriptionInterface,
        ResponseInterface,
        TextToSpeechInterface,
        WakeWordInterface
    )
    from assistant_framework.models.data_models import TranscriptionResult, ResponseChunk, WakeWordEvent
    from assistant_framework.utils.state_machine import AudioStateMachine, AudioState
    from assistant_framework.utils.error_handling import ErrorHandler, ComponentError, ErrorSeverity
    from assistant_framework.utils.barge_in import BargeInDetector, BargeInConfig, BargeInMode
    from assistant_framework.providers.wakeword_v2 import IsolatedOpenWakeWordProvider
    from assistant_framework.providers.transcription_v2 import AssemblyAIAsyncProvider
    from assistant_framework.config import get_active_preset


class RefactoredOrchestrator:
    """
    Refactored orchestrator with:
    - State machine for lifecycle management
    - Structured error handling
    - Simplified pipeline execution
    - V2 providers (process-isolated, fully async)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core infrastructure (respect current preset: dev/prod/test)
        try:
            preset = get_active_preset()
        except Exception:
            preset = "prod"
        self.state_machine = AudioStateMachine(mode=preset)
        self.error_handler = ErrorHandler()
        
        # Provider instances (will be lazily created)
        self._transcription: Optional[TranscriptionInterface] = None
        self._response: Optional[ResponseInterface] = None
        self._tts: Optional[TextToSpeechInterface] = None
        self._wakeword: Optional[WakeWordInterface] = None
        
        # Barge-in support (interrupt TTS with speech)
        self._barge_in_enabled = config.get('barge_in_enabled', True)
        self._barge_in_detector: Optional[BargeInDetector] = None
        self._barge_in_triggered = False  # Flag to signal TTS interruption
        self._barge_in_audio: Optional[bytes] = None  # Captured audio from barge-in
        
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
            
            # Register cleanup handlers with state machine (CRITICAL for preventing segfaults)
            self._register_cleanup_handlers()
            
            self.is_initialized = True
            print("\n✅ All providers initialized and ready")
            print("💡 Providers will be reused across conversations (cleanup on transitions)")
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
    
    async def _create_wakeword_provider(self) -> WakeWordInterface:
        """Create wake word provider (called once at startup)."""
        config = self.config['wakeword']['config']
        provider = IsolatedOpenWakeWordProvider(config)
        await provider.initialize()
        return provider
    
    async def _create_transcription_provider(self) -> TranscriptionInterface:
        """Create transcription provider (called once at startup)."""
        config = self.config['transcription']['config']
        provider = AssemblyAIAsyncProvider(config)
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
    
    async def run_wake_word_detection(self):
        """Run wake word detection loop."""
        try:
            # Transition to wake word state
            await self.state_machine.transition_to(
                AudioState.WAKE_WORD_LISTENING,
                "wakeword"
            )
            
            # Use pre-initialized provider
            wakeword = self._wakeword
            
            # Start detection
            print("👂 Listening for wake word...")
            async for event in wakeword.start_detection():
                print(f"🔔 Wake word detected: {event.model_name} (score: {event.score:.3f})")
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
    
    async def run_transcription(self) -> Optional[str]:
        """Run transcription and return final text when send phrase is detected."""
        try:
            # Transition to transcription state
            await self.state_machine.transition_to(
                AudioState.TRANSCRIBING,
                "transcription"
            )
            
            # Use pre-initialized provider
            transcription = self._transcription
            if not transcription:
                print("❌ Transcription provider not available")
                return None
            
            # Check for barge-in prefill audio
            if self._barge_in_audio:
                print("📼 Using barge-in captured audio as prefill")
                if hasattr(transcription, 'set_prefill_audio'):
                    transcription.set_prefill_audio(self._barge_in_audio)
                self._barge_in_audio = None  # Clear after use
            
            # Get send phrases from config
            from .config import SEND_PHRASES, TERMINATION_PHRASES
            
            # Start streaming
            print("🎙️  Transcribing...")
            print(f"💡 Say one of these to send: {', '.join(SEND_PHRASES)}")
            
            accumulated_text = ""
            
            async for result in transcription.start_streaming():
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
                            # Remove the send phrase from the accumulated text (case-insensitive)
                            import re
                            pattern = re.compile(re.escape(send_phrase), re.IGNORECASE)
                            cleaned_text = pattern.sub("", accumulated_text).strip()
                            # Clean up extra spaces
                            cleaned_text = " ".join(cleaned_text.split())
                            return cleaned_text if cleaned_text else None
                    
                    # Check for termination phrases
                    for term_phrase in TERMINATION_PHRASES:
                        if term_phrase.lower() in accumulated_lower:
                            print(f"🛑 Termination phrase detected: '{term_phrase}'")
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
                            # Remove the send phrase
                            import re
                            pattern = re.compile(re.escape(send_phrase), re.IGNORECASE)
                            cleaned_text = pattern.sub("", full_text).strip()
                            cleaned_text = " ".join(cleaned_text.split())
                            return cleaned_text if cleaned_text else None
                    
                    # Check for termination phrases in partial
                    for term_phrase in TERMINATION_PHRASES:
                        if term_phrase.lower() in full_text_lower:
                            print(f"🛑 Termination phrase detected in partial: '{term_phrase}'")
                            return None
            
            # If stream ends naturally without send phrase, return accumulated text
            return accumulated_text if accumulated_text else None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            error = ComponentError(
                component="transcription",
                severity=ErrorSeverity.RECOVERABLE,
                message="Transcription error",
                exception=e
            )
            await self.error_handler.handle_error(error)
            return None
        
        # Note: No finally block needed - state machine handles cleanup on transition
        # The finally block was causing double cleanup (here + state machine)
        # which led to segmentation faults
    
    async def run_response(self, user_message: str) -> Optional[str]:
        """Generate response for user message."""
        try:
            # Ensure transcription is fully stopped before starting response
            # This prevents audio device conflicts and executor thread issues
            # NOTE: The state machine cleanup handler will also call stop_streaming,
            # but stop_streaming is idempotent so this is safe
            if self._transcription and self._transcription.is_active:
                print("🔄 Ensuring transcription fully stopped before response...")
                await self._transcription.stop_streaming()
                # Brief wait for audio device to fully release
                await asyncio.sleep(0.3)
            
            # Transition to response state
            # NOTE: State machine may trigger additional cleanup via registered handlers
            await self.state_machine.transition_to(
                AudioState.PROCESSING_RESPONSE,
                "response"
            )
            
            # Use pre-initialized provider
            response = self._response
            
            # Stream response
            print(f"💭 Generating response...")
            full_response = ""
            streamed_deltas = False
            
            async for chunk in response.stream_response(user_message):
                if chunk.content:
                    if chunk.is_complete:
                        # Final complete chunk contains the FULL text
                        # If we were streaming deltas, we already printed them
                        # If not, print the complete response now
                        full_response = chunk.content
                        if not streamed_deltas:
                            print(chunk.content, end="", flush=True)
                    else:
                        # Streaming delta - accumulate and print
                        full_response += chunk.content
                        print(chunk.content, end="", flush=True)
                        streamed_deltas = True
            
            print()  # Newline after response
            return full_response if full_response else None
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            error = ComponentError(
                component="response",
                severity=ErrorSeverity.RECOVERABLE,
                message="Response error",
                exception=e
            )
            await self.error_handler.handle_error(error)
            return None
    
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
            await self.state_machine.transition_to(
                AudioState.SYNTHESIZING,
                "tts"
            )
            
            # Use pre-initialized provider
            tts = self._tts
            
            # Synthesize audio first
            print("🔊 Synthesizing speech...")
            audio = await tts.synthesize(text)
            
            # Setup barge-in detection if enabled
            if enable_barge_in and self._barge_in_enabled:
                print("👂 Barge-in enabled - speak to interrupt")
                self._barge_in_detector = BargeInDetector(BargeInConfig(
                    mode=BargeInMode.ENERGY,
                    energy_threshold=0.025,  # Slightly higher to avoid TTS feedback
                    min_speech_duration=0.2,  # 200ms of speech required
                    cooldown_after_tts_start=0.8  # Wait 800ms before detecting (skip TTS start)
                ))
                
                # Define callback to stop TTS when barge-in detected
                def on_barge_in():
                    print("🎤 BARGE-IN: Interrupting speech!")
                    self._barge_in_triggered = True
                    if tts and hasattr(tts, 'stop_audio'):
                        tts.stop_audio()
                
                await self._barge_in_detector.start(on_barge_in=on_barge_in)
            
            # Play audio (will be interrupted if barge-in triggers)
            print("🔊 Speaking...")
            await tts.play_audio_async(audio)
            
            # Check if barge-in occurred
            if self._barge_in_triggered:
                print("⚡ Speech interrupted by user")
                barge_in_occurred = True
            else:
                print("✅ Speech complete")
            
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
                pass
            elif transition_to_idle:
                # Normal completion - go to IDLE
                await self.state_machine.transition_to(AudioState.IDLE)
        
        return not barge_in_occurred
    
    async def run_full_conversation(self):
        """Run complete conversation pipeline with barge-in support."""
        try:
            # 1. Wait for wake word
            async for wake_event in self.run_wake_word_detection():
                print(f"\n🎯 Wake word detected: {wake_event.model_name}\n")
                break  # Got wake word, proceed
            
            # Explicitly stop wake word detection before proceeding
            # The async generator break doesn't stop the subprocess
            if self._wakeword:
                await self._wakeword.stop_detection()
                # Wait for subprocess to fully terminate
                await asyncio.sleep(0.5)
            
            # 2. Transcribe user speech
            user_text = await self.run_transcription()
            if not user_text:
                print("⚠️  No transcription received")
                return
            
            print(f"\n👤 User: {user_text}\n")
            
            # 3. Generate response
            assistant_text = await self.run_response(user_text)
            if not assistant_text:
                print("⚠️  No response generated")
                return
            
            print(f"\n🤖 Assistant: {assistant_text}\n")
            
            # 4. Speak response with barge-in support
            speech_completed = await self.run_tts(assistant_text, transition_to_idle=True, enable_barge_in=True)
            
            if speech_completed:
                print("\n✅ Conversation complete\n")
            else:
                print("\n⚡ Speech interrupted - ready for follow-up\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            await self.state_machine.emergency_reset()
        except Exception as e:
            print(f"\n❌ Conversation error: {e}")
            await self.state_machine.emergency_reset()
    
    async def run_continuous_loop(self):
        """Run continuous conversation loop with multi-turn support and barge-in."""
        print("🔁 Starting continuous conversation loop...")
        print("   Press Ctrl+C to stop")
        print("   💡 You can interrupt the assistant by speaking!\n")
        
        try:
            while True:
                # 1. Wait for wake word to start conversation
                async for wake_event in self.run_wake_word_detection():
                    print(f"\n🎯 Wake word detected: {wake_event.model_name}\n")
                    break  # Got wake word, enter conversation mode
                
                # Explicitly stop wake word detection before entering conversation
                # The async generator break doesn't stop the subprocess
                if self._wakeword:
                    await self._wakeword.stop_detection()
                    # Wait for subprocess to fully terminate
                    await asyncio.sleep(0.5)
                
                # 2. Enter multi-turn conversation mode
                print("💬 Conversation mode active (say termination phrase to exit)")
                conversation_active = True
                
                while conversation_active:
                    # Transcribe user speech
                    user_text = await self.run_transcription()
                    
                    # Check if user wants to end conversation
                    if not user_text:
                        print("⚠️  No transcription received, ending conversation")
                        conversation_active = False
                        break
                    
                    print(f"\n👤 User: {user_text}\n")
                    
                    # Generate response
                    assistant_text = await self.run_response(user_text)
                    if not assistant_text:
                        print("⚠️  No response generated")
                        continue  # Try next question
                    
                    print(f"\n🤖 Assistant: {assistant_text}\n")
                    
                    # Speak response with barge-in enabled
                    # If user interrupts, we'll immediately start transcribing
                    speech_completed = await self.run_tts(
                        assistant_text, 
                        transition_to_idle=False,  # Don't auto-transition, we handle it
                        enable_barge_in=True
                    )
                    
                    if speech_completed:
                        # Normal completion - transition to IDLE then back to transcription
                        await self.state_machine.transition_to(AudioState.IDLE)
                        print("🎤 Ready for next question (or say termination phrase)...\n")
                    else:
                        # Barge-in occurred! Skip IDLE and go directly to transcription
                        # The captured audio will be prefilled to transcription
                        if self._barge_in_audio:
                            duration = len(self._barge_in_audio) / 2 / 16000
                            print(f"⚡ Barge-in: {duration:.2f}s of audio captured, feeding to transcription...")
                        else:
                            print("⚡ Barge-in: Going directly to transcription...")
                        # Small delay for audio device handoff
                        await asyncio.sleep(0.2)
                        print("🎤 Listening (prefill audio will be processed first)...\n")
                        # Continue the loop - next iteration will run transcription
                
                # Conversation ended - ensure we're in IDLE before restarting wake word
                print("✅ Conversation session ended\n")
                
                # Critical: Transition to IDLE if not already there
                # This prevents TRANSCRIBING → WAKE_WORD_LISTENING (invalid transition)
                if self.state_machine.current_state != AudioState.IDLE:
                    print("🔄 Transitioning to IDLE before restarting wake word...")
                    await self.state_machine.transition_to(AudioState.IDLE)
                
                # Extra settling time for audio device to fully release
                print("⏳ Waiting for audio device to settle...")
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n⚠️  Stopping continuous loop...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all resources."""
        print("🧹 Cleaning up orchestrator...")
        
        # Cleanup barge-in detector if active
        if self._barge_in_detector:
            try:
                await self._barge_in_detector.stop()
            except Exception as e:
                print(f"⚠️  Barge-in cleanup error: {e}")
            self._barge_in_detector = None
        
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
            await self.state_machine.transition_to(AudioState.IDLE)
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
            }
        }

