#!/usr/bin/env python3
"""
Main orchestration file for the home assistant system.
Manages the state machine between wake word detection and transcription.
"""

import asyncio
import re
import sys
import signal
import threading
import queue
from enum import Enum
from pathlib import Path
from typing import Optional  # noqa: F401 (kept for type hints below if needed)
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root and framework are on sys.path
project_root = str(Path(__file__).parent)
framework_path = str(Path(__file__).parent / "assistant_framework")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if framework_path not in sys.path:
    sys.path.insert(0, framework_path)

# Import VAD (optional). Fall back to a no-op if local module not present
try:
    from deprecated.vad import VoiceActivityDetector  # type: ignore
except Exception:
    class VoiceActivityDetector:  # minimal stub
        def __init__(self, **kwargs):
            self.on_speech_end = None

        def calculate_energy(self, audio_frame: bytes) -> float:
            try:
                import numpy as np  # lazy import
                audio_data = np.frombuffer(audio_frame, dtype=np.int16)
                if len(audio_data) > 0:
                    energy = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0)
                    return energy
            except Exception:
                pass
            return 0.0

        def detect_voice_activity(self, energy: float) -> None:
            return

from assistant_framework.config import (
    get_framework_config,
    validate_environment,
    TERMINATION_PHRASES,
    TERMINATION_CHECK_MODE,
    TERMINATION_TIMEOUT,
    AUDIO_HANDOFF_DELAY,
    SEND_PHRASES,
    PREFIX_TRIM_PHRASES
)
from assistant_framework.orchestrator import create_orchestrator
from assistant_framework.utils.audio_manager import safe_audio_transition, get_audio_manager
from assistant_framework.utils.tones import (
    beep_wake_detected,
    beep_agent_message,
    beep_transcription_end,
    beep_ready_to_listen,
    beep_send_detected,
)

# Apply macOS audio fixes early without importing the core package (avoids core.__init__ side-effects)
try:
    import importlib.util as _il  # type: ignore  # noqa: E402
    from pathlib import Path as _P  # noqa: E402
    _fix_path = _P(__file__).parent / "core" / "fixes" / "macos_audio_fix.py"
    if _fix_path.exists():
        _spec = _il.spec_from_file_location("macos_audio_fix", str(_fix_path))
        if _spec and _spec.loader:
            _mod = _il.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            try:
                _mod.configure_macos_audio()
                _mod.suppress_auhal_errors()
            except Exception:
                pass
except Exception:
    pass


class SystemState(Enum):
    """System state enumeration."""
    INITIALIZING = "initializing"
    WAITING_FOR_WAKE = "waiting_for_wake"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HomeAssistant:
    """Main home assistant orchestrator."""
    
    def __init__(self):
        self.state = SystemState.INITIALIZING
        self.orchestrator = None
        self.running = False
        self.transcription_start_time = None
        # Track length of last printed partial line to properly clear leftovers
        self._last_partial_len = 0
        # VAD related
        self.vad = None
        self.vad_thread = None
        self.audio_queue = queue.Queue()
        # Transcription buffer for send phrases
        self.transcription_buffer = []
        # TTS interruption tracking
        self.tts_playing = False
        self.interruption_word_count = 0
        self._tts_partial_ref = ""
        
    async def initialize(self):
        """Initialize the home assistant system."""
        print("🚀 Initializing Home Assistant System...")
        print("=" * 60)
        
        # Validate environment
        validation = validate_environment()
        if not validation['valid']:
            print("❌ Environment validation failed:")
            for error in validation['errors']:
                print(f"  - {error}")
            raise RuntimeError("Environment validation failed")
        
        print("✅ Environment validated")
        
        # Get configuration
        config = get_framework_config()
        
        # Create orchestrator
        print("📦 Creating orchestrator...")
        self.orchestrator = await create_orchestrator(config)
        
        if not self.orchestrator:
            raise RuntimeError("Failed to create orchestrator")
        
        print("✅ Orchestrator initialized")
        print(f"📝 Termination phrases: {', '.join(TERMINATION_PHRASES)}")
        print(f"⏱️  Termination timeout: {TERMINATION_TIMEOUT} seconds")
        print("=" * 60)
        
        self.state = SystemState.WAITING_FOR_WAKE
        self.running = True
        
    async def wait_for_wake_word(self):
        """Wait for wake word detection."""
        self.state = SystemState.WAITING_FOR_WAKE
        print("\n👂 Listening for wake word...")
        # Beep to indicate we're ready to listen again
        try:
            beep_ready_to_listen()
        except Exception:
            pass
        
        try:
            async for event in self.orchestrator.run_wakeword_only():
                print(f"✨ Wake word detected: {event['model_name']} (score: {event['score']:.2f})")
                # Beep on wake word activation (non-blocking, never raises)
                try:
                    beep_wake_detected()
                except Exception:
                    pass
                return True
        except Exception as e:
            print(f"❌ Wake word detection error: {e}")
            return False
            
    async def stop_wake_word(self):
        """Stop wake word detection and clean up audio resources."""
        try:
            if self.orchestrator.wakeword:
                await self.orchestrator.wakeword.stop_detection()
                print("🔄 Wake word detection stopped")
        except Exception as e:
            print(f"⚠️  Error stopping wake word: {e}")
            
    def clean_transcription_text(self, text: str) -> str:
        """Clean transcription text by removing artifacts and normalizing."""
        if not text:
            return text
            
        # Remove common transcription artifacts
        clean_text = text
        
        # Remove trailing single letter artifacts after periods (case insensitive)
        clean_text = re.sub(r'\.[a-zA-Z]\b', '', clean_text)
        
        # Also remove single letters at the very end of text after any punctuation
        clean_text = re.sub(r'[.!?][a-zA-Z]$', '', clean_text)
        
        # Remove multiple spaces and normalize punctuation
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'\s+([,.!?])', r'\1', clean_text)
        
        # Remove orphaned periods at word boundaries
        clean_text = re.sub(r'\b\.\s', ' ', clean_text)
        
        return clean_text.strip()
    
    def check_for_send_phrase(self, text: str) -> bool:
        """Check if the text contains a send phrase."""
        # Clean up the text by removing artifacts and normalize
        clean_text = self.clean_transcription_text(text)
        text_lower = clean_text.lower().strip()
        
        for phrase in SEND_PHRASES:
            phrase_lower = phrase.lower().strip()
            # Check for exact phrase match or phrase at word boundaries
            if phrase_lower in text_lower:
                # Verify it's a proper word boundary match, not just substring
                words = text_lower.split()
                phrase_words = phrase_lower.split()
                
                # Simple substring match for short phrases
                if len(phrase_words) <= 2:
                    return True
                    
                # For longer phrases, check word sequence
                for i in range(len(words) - len(phrase_words) + 1):
                    if words[i:i+len(phrase_words)] == phrase_words:
                        return True
        return False
    
    def check_for_termination(self, text: str) -> bool:
        """Check if the text contains a termination phrase."""
        # Clean up the text by removing artifacts and normalize
        clean_text = self.clean_transcription_text(text)
        text_lower = clean_text.lower().strip()
        
        for phrase in TERMINATION_PHRASES:
            phrase_lower = phrase.lower().strip()
            # Check for exact phrase match or phrase at word boundaries
            if phrase_lower in text_lower:
                # Verify it's a proper word boundary match, not just substring
                words = text_lower.split()
                phrase_words = phrase_lower.split()
                
                # Simple substring match for short phrases
                if len(phrase_words) <= 2:
                    return True
                    
                # For longer phrases, check word sequence
                for i in range(len(words) - len(phrase_words) + 1):
                    if words[i:i+len(phrase_words)] == phrase_words:
                        return True
        return False
    
    def run_vad_detection(self):
        """Run VAD detection in parallel with transcription."""
        # Create VAD with 3-second silence duration
        vad = VoiceActivityDetector(
            silence_duration_ms=3000,  # 3 seconds
            speech_duration_ms=200,
            adaptive_threshold=True,
            energy_threshold=0.01
        )
        
        def on_speech_end(timestamp, duration):
            # Only print if we're still transcribing
            if self.state == SystemState.TRANSCRIBING:
                # Clear the partial line before printing VAD message
                self._clear_partial_line()
                print("\n🔇 Finished speaking")
        
        vad.on_speech_end = on_speech_end
        
        # Process audio chunks from queue
        while self.state == SystemState.TRANSCRIBING:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if self.state != SystemState.TRANSCRIBING:
                    break
                energy = vad.calculate_energy(audio_data)
                vad.detect_voice_activity(energy)
            except queue.Empty:
                continue
            except Exception as e:
                if self.state == SystemState.TRANSCRIBING:
                    print(f"VAD error: {e}")
                break
    
    def remove_termination_phrase(self, text: str) -> str:
        """Remove termination phrase from text and clean artifacts."""
        # First clean the text
        clean_text = self.clean_transcription_text(text)
        clean_lower = clean_text.lower()
        result = clean_text
        
        for phrase in TERMINATION_PHRASES:
            phrase_lower = phrase.lower()
            if phrase_lower in clean_lower:
                # Find the position and remove it
                idx = clean_lower.find(phrase_lower)
                if idx != -1:
                    result = clean_text[:idx] + clean_text[idx + len(phrase):]
                    result = result.strip()
                    # Clean any remaining artifacts after removal
                    result = self.clean_transcription_text(result)
                    break
                    
        return result
    
    def remove_send_phrase(self, text: str) -> str:
        """Remove send phrase from text and clean artifacts."""
        # First clean the text
        clean_text = self.clean_transcription_text(text)
        clean_lower = clean_text.lower()
        result = clean_text
        
        for phrase in SEND_PHRASES:
            phrase_lower = phrase.lower()
            if phrase_lower in clean_lower:
                # Find the position and remove it
                idx = clean_lower.find(phrase_lower)
                if idx != -1:
                    result = clean_text[:idx] + clean_text[idx + len(phrase):]
                    result = result.strip()
                    # Clean any remaining artifacts after removal
                    result = self.clean_transcription_text(result)
                    break
                    
        return result

    def trim_prefix_on_phrases(self, text: str) -> str:
        """Remove everything up to and including the first matching trim phrase.

        - Case-insensitive
        - Matches at word boundaries to avoid mid-word hits
        - If multiple matches exist, trims to the LAST occurrence (most recent correction)
        - Consumes trailing punctuation/spaces after the phrase
        """
        try:
            if not PREFIX_TRIM_PHRASES:
                return text
            src = self.clean_transcription_text(text)
            import re as _re
            # Build matches and pick the last (rightmost) valid boundary match
            last_cut = None
            for phrase in PREFIX_TRIM_PHRASES:
                p = phrase.strip()
                if not p:
                    continue
                # Word-boundary, case-insensitive
                pattern = _re.compile(r"\b" + _re.escape(p) + r"\b", flags=_re.IGNORECASE)
                for m in pattern.finditer(src):
                    cut = m.end()
                    # Consume trailing punctuation/spaces
                    while cut < len(src) and src[cut] in " .,!?:;-\t\n\r":
                        cut += 1
                    last_cut = cut if (last_cut is None or cut > last_cut) else last_cut
            if last_cut is None:
                return text
            trimmed = src[last_cut:].lstrip()
            return trimmed
        except Exception:
            return text
    
    def get_transcription_buffer_text(self) -> str:
        """Get the complete transcription buffer as a single string."""
        return " ".join(self.transcription_buffer).strip()
    
    def clear_transcription_buffer(self):
        """Clear the transcription buffer."""
        self.transcription_buffer = []
    
    async def send_to_response_component(self, text: str):
        """Send text to the response component for processing."""
        if not text.strip():
            print("⚠️  No text to send to response component")
            return
        
        try:
            # Build the message to send: prefer the accumulated buffer; if empty, use cleaned current text
            buffer_text = self.get_transcription_buffer_text().strip()
            if not buffer_text:
                cleaned_once = self.remove_send_phrase(text).strip()
                buffer_text = cleaned_once
            
            if buffer_text:
                # Apply prefix trim before printing/sending so logs and context reflect final text
                final_text = self.trim_prefix_on_phrases(buffer_text)
                if not final_text.strip():
                    print("⚠️  No content to send after applying trim phrases")
                    # Clear the buffer after a reset phrase
                    self.clear_transcription_buffer()
                    return
                print(f"\n📤 Sending to response component: '{final_text}'")
                
                # Persist the user's message in context for memory
                try:
                    if self.orchestrator.context and final_text:
                        self.orchestrator.context.add_message("user", final_text)
                        self.orchestrator.context.auto_trim_if_needed()
                except Exception as e:
                    print(f"⚠️  Error updating context with user message: {e}")

                # Send to response component via orchestrator
                if self.orchestrator.response:
                    print("🤖 ", end="", flush=True)  # Print robot emoji once at start
                    full_response = ""
                    has_streamed = False
                    beeped_for_agent = False
                    
                    # Use orchestrator to include conversation history
                    async for chunk in self.orchestrator.run_response_only(
                        final_text,
                        use_context=True
                    ):
                        # Beep once when the first content arrives from the agent
                        if chunk.content:
                            if not beeped_for_agent:
                                try:
                                    beep_agent_message()
                                except Exception:
                                    pass
                                beeped_for_agent = True
                            if not chunk.is_complete:
                                # Stream partial responses (deltas)
                                print(chunk.content, end="", flush=True)
                                full_response += chunk.content
                                has_streamed = True
                            elif not has_streamed:
                                # Only show final response if we haven't streamed yet
                                print(chunk.content, end="", flush=True)
                                full_response = chunk.content
                    print()  # New line after response
                    
                    # Persist assistant reply to context for memory
                    try:
                        if self.orchestrator.context and full_response.strip():
                            self.orchestrator.context.add_message("assistant", full_response)
                            self.orchestrator.context.auto_trim_if_needed()
                    except Exception as e:
                        print(f"⚠️  Error updating context with assistant reply: {e}")
                    
                    # Play the response using TTS
                    if full_response.strip() and self.orchestrator.tts:
                        try:
                            print("🔊 Playing response...")
                            self.tts_playing = True
                            self.interruption_word_count = 0
                            self._tts_partial_ref = ""
                            
                            # Synthesize audio
                            audio_output = await self.orchestrator.tts.synthesize(full_response)
                            
                            # Start playing audio in background with error handling
                            asyncio.create_task(
                                self._play_tts_safely(audio_output)
                            )
                        except Exception as e:
                            print(f"❌ TTS synthesis error: {e}")
                            # Reset TTS state on synthesis error
                            self.tts_playing = False
                            self.interruption_word_count = 0
                            self._tts_partial_ref = ""
                
                # Clear the buffer after sending
                self.clear_transcription_buffer()
                
            else:
                print("⚠️  No content to send after cleaning")
                
        except Exception as e:
            print(f"❌ Error sending to response component: {e}")
    
    async def _play_tts_safely(self, audio_output):
        """Play TTS audio with error handling to prevent program termination."""
        try:
            await self.orchestrator.tts.play_audio_async(audio_output)
        except Exception as e:
            print(f"❌ TTS playback error: {e}")
            # Reset TTS state on error
            self.tts_playing = False
            self.interruption_word_count = 0
            self._tts_partial_ref = ""
    
    async def process_transcriptions(self):
        """Process transcriptions until termination phrase is detected."""
        self.state = SystemState.TRANSCRIBING
        
        # Clear transcription buffer at start of new session
        self.clear_transcription_buffer()
        
        print("\n🎤 Transcription started. Say a termination phrase to stop.")
        print(f"   Termination phrases: {', '.join(TERMINATION_PHRASES)}")
        print(f"   Send phrases: {', '.join(SEND_PHRASES)}")
        print("-" * 60)
        
        self.transcription_start_time = datetime.now()
        timeout_time = self.transcription_start_time + timedelta(seconds=TERMINATION_TIMEOUT)
        
        # Set audio callback to feed VAD
        if self.orchestrator.transcription:
            self.orchestrator.transcription.set_audio_callback(
                lambda data: self.audio_queue.put(data)
            )
        
        # Start VAD thread
        self.vad_thread = threading.Thread(target=self.run_vad_detection)
        self.vad_thread.daemon = True
        self.vad_thread.start()
        
        try:
            async for result in self.orchestrator.run_transcription_only():
                # Check overall timeout
                if datetime.now() > timeout_time:
                    print("\n⏰ Transcription timeout reached")
                    break
                
                # Check if TTS finished naturally
                if self.tts_playing:
                    try:
                        if not self.orchestrator.tts.is_playing:
                            print("✅ TTS playback completed naturally")
                            self.tts_playing = False
                            self.interruption_word_count = 0
                            self._tts_partial_ref = ""
                    except Exception as e:
                        print(f"⚠️  Error checking TTS status: {e}")
                        # Reset TTS state if we can't check status
                        self.tts_playing = False
                        self.interruption_word_count = 0
                        self._tts_partial_ref = ""
                
                # Check for TTS interruption first (if TTS is playing) using PARTIALS too
                if self.tts_playing and result.text.strip():
                    # Compare against last seen partial to count only newly added words
                    cleaned = self.clean_transcription_text(result.text.strip())
                    new_words = self._count_new_words(self._tts_partial_ref, cleaned)
                    if new_words > 0:
                        self.interruption_word_count += new_words
                        self._tts_partial_ref = cleaned
                        print(f"\n🎙️  Speech during TTS: '+{new_words}' (total: {self.interruption_word_count}) -> {cleaned}")
                    
                    if self.interruption_word_count >= 1:
                        print("🛑 TTS interrupted - stopping audio")
                        try:
                            self.orchestrator.tts.stop_audio()
                        except Exception as e:
                            print(f"⚠️  Error stopping TTS: {e}")
                        self.tts_playing = False
                        self.interruption_word_count = 0
                        self._tts_partial_ref = ""
                        # Continue transcribing for new content
                        self.clear_transcription_buffer()
                    continue  # Don't process as normal transcription while TTS is playing
                
                # Check for send phrases first (only on final results)
                if result.is_final and self.check_for_send_phrase(result.text):
                    # Found send phrase - send buffer to response component
                    clean_text = self.remove_send_phrase(result.text)
                    if clean_text:
                        # Add clean text to buffer before sending
                        self.transcription_buffer.append(clean_text)
                    
                    # Clear partial line and send to response
                    self._clear_partial_line()
                    print(f"📝 [FINAL] {self.clean_transcription_text(result.text)}")
                    print("\n📨 Send phrase detected")
                    try:
                        beep_send_detected()
                    except Exception:
                        pass
                    
                    # Send to response component
                    await self.send_to_response_component(result.text)
                    continue
                
                # Check for termination based on mode
                should_check = (
                    (TERMINATION_CHECK_MODE == "final" and result.is_final) or
                    (TERMINATION_CHECK_MODE == "partial")
                )
                
                if should_check and self.check_for_termination(result.text):
                    # Found termination phrase
                    clean_text = self.remove_termination_phrase(result.text)
                    if clean_text and result.is_final:
                        # Add to buffer before terminating
                        self.transcription_buffer.append(clean_text)
                        # Clear any partial line before printing a clean final
                        self._clear_partial_line()
                        print(f"📝 [FINAL] {clean_text}")
                    print("\n🛑 Termination phrase detected")
                    break
                
                # Display transcription with cleaning
                display_text = self.clean_transcription_text(result.text)
                if result.is_final:
                    # Add final results to buffer
                    self.transcription_buffer.append(display_text)
                    # Clear partial line then print final to avoid leftover characters
                    self._clear_partial_line()
                    print(f"📝 [FINAL] {display_text}")
                    
        except Exception as e:
            print(f"\n❌ Transcription error: {e}")
            
    async def stop_transcription(self):
        """Stop transcription and clean up resources."""
        try:
            # Change state first to signal threads to stop
            previous_state = self.state
            self.state = SystemState.PROCESSING
            
            # Stop any TTS first to avoid conflicts
            if self.tts_playing:
                try:
                    self.orchestrator.tts.stop_audio()
                    self.tts_playing = False
                    print("🔄 TTS stopped during transcription cleanup")
                except Exception as e:
                    print(f"⚠️  Error stopping TTS: {e}")
            
            # Clear the audio callback first to stop new data
            if self.orchestrator.transcription:
                self.orchestrator.transcription.set_audio_callback(None)
            
            # Stop VAD thread if running
            if self.vad_thread and self.vad_thread.is_alive():
                # VAD will stop due to state change
                self.vad_thread.join(timeout=2.0)  # Longer timeout
                self.vad_thread = None
                print("🔄 VAD thread stopped")
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Additional delay before stopping transcription
            await asyncio.sleep(0.5)
            
            # Stop transcription
            if self.orchestrator.transcription:
                await self.orchestrator.transcription.stop_streaming()
                print("🔄 Transcription stopped")
                # Beep to indicate transcription session has ended
                try:
                    beep_transcription_end()
                except Exception:
                    pass
            
            # Restore previous state if needed
            if previous_state == SystemState.TRANSCRIBING:
                self.state = SystemState.WAITING_FOR_WAKE
                
        except Exception as e:
            print(f"⚠️  Error stopping transcription: {e}")
        finally:
            # Ensure state is reset even on errors
            self.tts_playing = False
            self.interruption_word_count = 0
            self._tts_partial_ref = ""
            
    async def main_loop(self):
        """Main application loop."""
        await self.initialize()
        
        while self.running:
            try:
                # Phase 1: Wait for wake word
                wake_detected = await self.wait_for_wake_word()
                
                if not wake_detected:
                    print("⚠️  Wake word detection failed, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                # Phase 2: Safe transition from wake word to transcription
                await self.stop_wake_word()
                await safe_audio_transition("wakeword", "transcription", delay=AUDIO_HANDOFF_DELAY)

                # Phase 3: Process transcriptions
                await self.process_transcriptions()
                
                # Phase 4: Safe transition back to wake word
                await self.stop_transcription()
                
                # Add extra delay to let audio system settle and clear buffers
                print("⏳ Clearing audio buffers...")
                await asyncio.sleep(2.0)  # Longer delay to prevent segfault
                
                # Avoid forcing audio cleanup here; soft release happens in safe_audio_transition
                
                await safe_audio_transition("transcription", "wakeword", delay=AUDIO_HANDOFF_DELAY)
                
                # Additional delay before wake word starts listening
                await asyncio.sleep(1.0)  # Longer delay to prevent segfault
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                self.state = SystemState.ERROR
                await asyncio.sleep(5)
                self.state = SystemState.WAITING_FOR_WAKE
                
        await self.shutdown()
        
    async def shutdown(self):
        """Clean shutdown of the system."""
        self.state = SystemState.SHUTDOWN
        self.running = False
        
        print("\n🧹 Cleaning up...")
        
        # Create tasks with timeout for parallel cleanup
        cleanup_tasks = []
        
        try:
            # Stop transcription with timeout
            if self.orchestrator and self.orchestrator.transcription and self.orchestrator.transcription.is_active:
                cleanup_tasks.append(
                    asyncio.create_task(
                        asyncio.wait_for(self.stop_transcription(), timeout=1.0)
                    )
                )
            
            # Stop wake word with timeout
            if self.orchestrator and self.orchestrator.wakeword:
                cleanup_tasks.append(
                    asyncio.create_task(
                        asyncio.wait_for(self.stop_wake_word(), timeout=1.0)
                    )
                )
            
            # Wait for all cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Clean up orchestrator
            if self.orchestrator:
                await asyncio.wait_for(self.orchestrator.cleanup(), timeout=2.0)
            
            # Force cleanup audio resources
            get_audio_manager().force_cleanup()
                
        except asyncio.TimeoutError:
            print("⚠️  Cleanup timeout - forcing shutdown")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        finally:
            # Ensure audio cleanup even on errors
            try:
                get_audio_manager().force_cleanup()
            except Exception:
                pass
            
        print("✅ Shutdown complete")

    def _print_partial(self, text: str) -> None:
        """Render a partial line, padding to clear any previous longer content."""
        line = f"   [partial] {text}"
        pad = max(0, self._last_partial_len - len(line))
        # Carriage return to start of the line, then write line + enough spaces to clear leftovers
        sys.stdout.write('\r' + line + (' ' * pad))
        sys.stdout.flush()
        self._last_partial_len = len(line)

    def _clear_partial_line(self) -> None:
        """Erase the previously printed partial line entirely."""
        if self._last_partial_len > 0:
            sys.stdout.write('\r' + (' ' * self._last_partial_len) + '\r')
            sys.stdout.flush()
            self._last_partial_len = 0

    @staticmethod
    def _count_new_words(previous: str, current: str) -> int:
        """Count new words added from previous->current partial text.
        Assumes current is an append of previous; safe fallback if not.
        """
        if not current:
            return 0
        if not previous:
            return len(current.split())
        if current.startswith(previous):
            tail = current[len(previous):].strip()
            return len(tail.split()) if tail else 0
        # Fallback if replacements happen: diff by word count (non-negative)
        return max(0, len(current.split()) - len(previous.split()))


async def main():
    """Main entry point."""
    assistant = HomeAssistant()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n📍 Signal {sig} received, shutting down...")
        assistant.running = False
        # Force exit after second interrupt
        if hasattr(signal_handler, 'called'):
            print("🚨 Force shutdown!")
            sys.exit(1)
        signal_handler.called = True
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await assistant.main_loop()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║            🏠 Home Assistant Voice System 🎤              ║
║                                                          ║
║  Wake Word → Transcription → Termination Loop           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)