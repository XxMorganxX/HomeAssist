"""
Main streaming chatbot orchestrator.
Coordinates audio capture, speech detection, transcription, and chat responses.
"""

import os
import sys
import time
import queue
import platform
from pathlib import Path
from typing import List
import logging
import json
from datetime import datetime
import numpy as np

# Suppress common warnings from dependencies
from core.suppress_warnings import suppress_common_warnings
suppress_common_warnings()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db.db_connect import DBConnect

import sounddevice as sd
from dotenv import load_dotenv

# Make script act like it's run from project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from core.audio_processing import VADChunker, wav_bytes_from_frames
import config

# Conditionally import speech services based on config
if config.USE_REALTIME_API:
    try:
        from core.speech_services_realtime import SpeechServices, ConversationManager
        USING_REALTIME_API = True
        if getattr(config, 'REALTIME_STREAMING_MODE', False):
            print("🚀 Using OpenAI Real-time API with continuous streaming")
        else:
            print("🚀 Using OpenAI Real-time API with chunk-based mode")
    except ImportError as e:
        print(f"⚠️ Real-time API not available ({e}), falling back to traditional API")
        from core.speech_services import SpeechServices, ConversationManager
        USING_REALTIME_API = False
else:
    from core.speech_services import SpeechServices, ConversationManager
    USING_REALTIME_API = False
    print("📡 Using traditional API for speech services")

load_dotenv()

VERBOSE_LOGGING = False 
if not VERBOSE_LOGGING:
    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp_server").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)
else:
    # Keep all logs visible
    logging.basicConfig(level=logging.INFO)

class StreamingChatbot:
    """Streaming chatbot for real-time voice interactions."""
    
    def __init__(self):
        """Initialize streaming chatbot."""
        # Use realtime model if realtime API is enabled
        chat_model = config.OPENAI_CHAT_MODEL
        if USING_REALTIME_API and config.CHAT_PROVIDER == "openai":
            chat_model = getattr(config, 'REALTIME_MODEL', 'gpt-4o-realtime-preview-2024-12-17')
        elif config.CHAT_PROVIDER == "gemini":
            chat_model = config.GEMINI_CHAT_MODEL
            
        self.speech_services = SpeechServices(
            openai_api_key=os.getenv("OPENAI_KEY"),
            whisper_model=config.WHISPER_MODEL,
            chat_provider=config.CHAT_PROVIDER,
            chat_model=chat_model,
            gemini_api_key=os.getenv("GOOGLE_API_KEY"),
            tts_enabled=config.TTS_ENABLED,
            tts_model=config.TEXT_TO_SPEECH_MODEL,
            tts_voice=config.TTS_VOICE
        )
        
        # Check if we're using continuous streaming mode
        self.use_streaming_mode = (USING_REALTIME_API and 
                                  getattr(config, 'REALTIME_STREAMING_MODE', False) and
                                  hasattr(self.speech_services, 'use_realtime') and
                                  self.speech_services.use_realtime)
        
        self.conversation = ConversationManager(config.SYSTEM_PROMPT)
        self.chunker = VADChunker(
            sample_rate=config.SAMPLE_RATE,
            frame_ms=config.FRAME_MS,
            vad_mode=config.VAD_MODE,
            silence_end_sec=config.SILENCE_END_SEC,
            max_utterance_sec=config.MAX_UTTERANCE_SEC,
            aec_enabled=config.AEC_ENABLED
        )
        
        # Audio handling
        self.audio_queue = queue.Queue()
        self.audio_stream = None
        self.accumulated_chunks = []
        self.last_speech_activity = None
        
        # Session state
        self.session_ended = False
        
        # Latency tracking
        self.transcription_latencies = []  # Track individual chunk transcription times
        self.conversation_latencies = []   # Track conversation processing times
        
        # Streaming mode setup
        if self.use_streaming_mode:
            print("🌊 Continuous streaming mode enabled")
            # Set up real-time callbacks
            self.speech_services.set_callbacks(
                partial_transcript_callback=self._handle_partial_transcript,
                response_delta_callback=self._handle_response_delta,
                audio_response_callback=self._handle_audio_response
            )
            self.current_partial_transcript = ""
            self.streaming_active = False
        else:
            print("📦 Chunk-based mode enabled")
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"⚠️  Audio status: {status}")
        self.audio_queue.put(bytes(indata))
    
    def send_to_db(self, chat: dict):
        """Send chat to database with latency metrics."""
        with DBConnect() as db:
            # Prepare latency data
            accumulated_latencies = self.transcription_latencies.copy() if self.transcription_latencies else None
            final_latencies = self.conversation_latencies.copy() if self.conversation_latencies else None
            
            # Log latency summary if we have data
            if accumulated_latencies:
                total_transcription = sum(accumulated_latencies)
                avg_transcription = total_transcription / len(accumulated_latencies)
                print(f"⏱️  Transcription latencies: {len(accumulated_latencies)} chunks, total: {total_transcription:.3f}s, avg: {avg_transcription:.3f}s")
            
            if final_latencies:
                total_conversation = sum(final_latencies)
                avg_conversation = total_conversation / len(final_latencies)
                print(f"⏱️  Conversation latencies: {len(final_latencies)} responses, total: {total_conversation:.3f}s, avg: {avg_conversation:.3f}s")
            
            # Save to database with latency metrics
            db.insert_new_chat(
                session_time=datetime.now(), 
                chat_data=chat,
                accumulated_latencies=accumulated_latencies,
                final_latencies=final_latencies
            )
    
    def _handle_partial_transcript(self, delta: str):
        """Handle partial transcription updates in streaming mode."""
        self.current_partial_transcript += delta
        print(f"🎤  Partial: {self.current_partial_transcript}", end='\r')
    
    def _handle_response_delta(self, delta: str):
        """Handle partial response updates in streaming mode."""
        print(delta, end='', flush=True)
    
    def _handle_audio_response(self, audio_b64: str):
        """Handle audio response chunks in streaming mode."""
        if self.speech_services.tts_enabled:
            # Could decode and play audio chunks in real-time here
            pass
    
    def _start_streaming_if_needed(self):
        """Start streaming mode if voice activity is detected."""
        if self.use_streaming_mode and not self.streaming_active:
            if self.speech_services.start_streaming():
                self.streaming_active = True
                print("\n🌊 Started streaming...")
            else:
                print("⚠️ Failed to start streaming, falling back to chunk mode")
    
    def _stop_streaming_if_needed(self):
        """Stop streaming mode and process the accumulated audio."""
        if self.use_streaming_mode and self.streaming_active:
            self.speech_services.stop_streaming()
            self.streaming_active = False
            # The transcription will come through the message handler
            print(f"\n⏹️ Stopped streaming. Final: {self.current_partial_transcript}")
            self.current_partial_transcript = ""

    def process_chunk(self, chunk: bytes) -> None:
        """Process a speech chunk through Whisper."""
        transcription_start_time = time.time()
        
        wav_io = wav_bytes_from_frames([chunk])
        user_text = self.speech_services.transcribe(wav_io)
        
        transcription_end_time = time.time()
        transcription_latency = transcription_end_time - transcription_start_time
        
        # Track transcription latency
        self.transcription_latencies.append(transcription_latency)
        
        if user_text:
            if user_text != "Learn English for free www.engvid.com" and user_text != "Learn more at www.plastics-car.com":
                print(f"🎤  Chunk: {user_text} (transcription: {transcription_latency:.3f}s)")
                
                # Check for end conversation phrases
                user_text_lower = user_text.lower()
                for phrase in config.END_CONVERSATION_PHRASES:
                    if phrase in user_text_lower:
                        print(f"🚫 End phrase '{phrase}' detected, stopping conversation")
                        print("🤖 Roger that! Conversation ending...")
                        self.session_ended = True
                        self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                        return
                
                # Check for terminal phrases that return to wake word mode
                for phrase in config.TERMINAL_PHRASES:
                    if phrase in user_text_lower:
                        print(f"🛑 Terminal phrase '{phrase}' detected, returning to wake word mode")
                        self.session_ended = True
                        self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                        return
            
                # Check for force send phrases
                for phrase in config.FORCE_SEND_PHRASES:
                    if phrase in user_text_lower:
                        print(f"⚡ Force send phrase '{phrase}' detected, processing immediately")
                        self.accumulated_chunks.append(user_text)
                        self.process_complete_message()
                        return
                
                self.accumulated_chunks.append(user_text)
            
    def process_complete_message(self) -> None:
        """Process accumulated chunks as complete message."""
        if not self.accumulated_chunks:
            return
            
        print("⏹️  [MESSAGE END] Complete silence detected, processing full message")
        
        # Combine all chunks
        complete_message = " ".join(self.accumulated_chunks)
        
        print(f"\n📝  Complete message: {complete_message}")
        
        # Check for end conversation phrases in complete message
        complete_message_lower = complete_message.lower()
        for phrase in config.END_CONVERSATION_PHRASES:
            if phrase in complete_message_lower:
                print(f"🚫 End phrase '{phrase}' detected in complete message, stopping conversation")
                print("🤖 Roger that! Conversation ending...")
                self.session_ended = True
                self.accumulated_chunks.clear()
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        # Check for terminal phrases in complete message
        for phrase in config.TERMINAL_PHRASES:
            if phrase in complete_message_lower:
                print(f"🛑 Terminal phrase '{phrase}' detected in complete message, returning to wake word mode")
                self.session_ended = True
                self.accumulated_chunks.clear()
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        print("🤖  Sending to ChatGPT...")
        
        # Start conversation processing timer
        conversation_start_time = time.time()
        
        # Add timestamp to queries about current information to force fresh tool calls
        enhanced_message = self._enhance_message_for_freshness(complete_message)
        
        # Add to conversation and get response
        self.conversation.add_user_message(enhanced_message)
        
        # Debug: Show conversation history length and content
        all_messages = self.conversation.get_messages()
        #print(f"📊 Conversation has {len(all_messages)} messages")
        # print("📜 Recent messages:")
        # for i, msg in enumerate(all_messages[-3:]):  # Show last 3 messages
        #     role = msg.get('role', 'unknown')
        #     content = msg.get('content', 'no content')[:100]  # First 100 chars
        #     print(f"   {i}: {role}: {content}")
        
        response_context = self.conversation.get_response_context()
        print(f"🤖 Using regular chat (temperature: {config.RESPONSE_TEMPERATURE}, context: {len(response_context)} messages)")
        response = self.speech_services.chat_completion(
            response_context,  # Use full context for response generation
            temperature=config.RESPONSE_TEMPERATURE  # Use configured response temperature
        )
        
        # End conversation processing timer
        conversation_end_time = time.time()
        conversation_latency = conversation_end_time - conversation_start_time
        self.conversation_latencies.append(conversation_latency)
        
        if response and response.get("content"):
            self.conversation.add_assistant_message(response["content"])
            print(f"🤖  GPT: {response['content']} (processing: {conversation_latency:.3f}s)\n")
            
            # Convert response to speech if TTS is enabled
            if self.speech_services.tts_enabled:
                audio_file = self.speech_services.text_to_speech(response["content"])
                if audio_file and hasattr(self, 'chunker') and self.chunker.aec_enabled:
                    # Add TTS audio as reference for AEC
                    self.chunker.add_reference_audio_file(audio_file)
            
            print("─" * 50)
        
        # Reset for next message
        self.accumulated_chunks = []
        self.last_speech_activity = None
        self.chunk_transcription_times = []  # Reset chunk latency tracking
        self.last_transcription_end_time = None
    
    def _enhance_message_for_freshness(self, message: str) -> str:
        """Add context to force fresh tool calls for time-sensitive queries."""
        message_lower = message.lower()
        
        # Keywords that indicate requests for current/real-time information
        real_time_keywords = [
            'today', 'now', 'current', 'schedule', 'calendar', 'events', 
            'what time', 'when', 'next', 'upcoming', 'status', 'weather'
        ]
        
        # Check if this appears to be a request for current information
        is_real_time_query = any(keyword in message_lower for keyword in real_time_keywords)
        
        if is_real_time_query:
            # Add current timestamp and explicit instruction to make the query unique
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Clear conversation history for real-time queries to prevent caching
            self._clear_similar_queries(message_lower)
            
            return f"{message} [FRESH_QUERY_{current_time.replace(' ', '_').replace(':', '-')}] - MANDATORY: Use calendar_data tool to get current information. Do not answer from memory."
        
        return message
    
    def _clear_similar_queries(self, current_query: str):
        """Remove similar queries from conversation history to prevent caching."""
        try:
            if not hasattr(self, 'conversation') or not self.conversation or not hasattr(self.conversation, 'messages'):
                return
                
            # Safety check for current_query
            if current_query is None:
                return
                
            # Keywords that make queries similar
            query_keywords = ['calendar', 'events', 'schedule', 'today']
            
            # Check if current query contains calendar-related keywords
            is_calendar_query = any(keyword in current_query for keyword in query_keywords)
            
            if is_calendar_query:
                # Instead of complex filtering that can break tool call sequences,
                # just clear everything except the system message for calendar queries
                # This is safer and ensures no broken tool call chains
                if not self.conversation.messages:
                    return
                    
                system_msg = self.conversation.messages[0] if (len(self.conversation.messages) > 0 and 
                                                             self.conversation.messages[0].get('role') == 'system') else None
                
                if system_msg:
                    # Reset to just system message for fresh calendar queries
                    self.conversation.messages = [system_msg]
                    
                    if config.DEBUG_MODE:
                        print("🗑️ Cleared all conversation history for fresh calendar query")
                else:
                    # If no system message, clear everything
                    self.conversation.messages = []
                    
                    if config.DEBUG_MODE:
                        print("🗑️ Cleared entire conversation history for fresh calendar query")
                    
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"⚠️ Error clearing similar queries: {e}")
            # Continue without clearing history if there's an error
        
    def run(self):
        """Main run loop for streaming chatbot using shared audio stream."""
        print("\n🎤  Ready. Speak into the microphone (Ctrl-C to quit)…")
        print("💡  I'll show each chunk as I hear it, then send the complete message after a longer pause.")
        print("─" * 50)
        
        # Use shared audio stream to allow terminal word detection during conversation
        from core.components import SharedAudioManager
        audio_manager = SharedAudioManager()
        
        # Brief delay to ensure any previous audio streams are fully cleaned up
        time.sleep(0.1)
        
        # Try to create shared stream for dual detection
        shared_stream_created = audio_manager.create_shared_stream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.FRAME_SIZE,
            channels=1,
            dtype=np.int16
        )
        
        if not shared_stream_created:
            print("⚠️ Could not create shared audio stream, falling back to exclusive mode")
            print("ℹ️ Terminal detection will use transcription-based detection only")
            
            # Fall back to exclusive audio access
            self._use_exclusive_audio_fallback()
            return
        
        # Subscribe to shared stream
        if not audio_manager.subscribe_to_stream("StreamingChatbot", self._shared_audio_callback):
            print("❌ Could not subscribe to shared audio stream")
            return
        
        # Start terminal word detector if available (after shared stream is created)
        terminal_detector_started = self._start_terminal_detector()
        
        try:
            while not self.session_ended:
                # Get audio frame from queue (populated by shared stream callback)
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                current_time = time.time()
                
                # Check if frame contains speech
                frame_has_speech = self.chunker.is_speech(frame)
                
                # Update last speech activity
                if frame_has_speech:
                    self.last_speech_activity = current_time
                
                # Handle streaming mode vs chunk mode
                if self.use_streaming_mode:
                    # Continuous streaming mode
                    if frame_has_speech:
                        self._start_streaming_if_needed()
                        # Stream the frame directly
                        if self.streaming_active:
                            self.speech_services.send_audio_frame(frame)
                    else:
                        # No speech detected, check if we should stop streaming
                        if (self.streaming_active and self.last_speech_activity and 
                            current_time - self.last_speech_activity > (config.REALTIME_VAD_SILENCE_MS / 1000.0)):
                            self._stop_streaming_if_needed()
                else:
                    # Traditional chunk-based mode
                    chunk = self.chunker.process(frame)
                    if chunk:
                        self.process_chunk(chunk)
                        # Exit immediately if session ended during chunk processing
                        if self.session_ended:
                            break
                    
                    # Check for end of complete message
                    if (self.accumulated_chunks and 
                        self.last_speech_activity and 
                        current_time - self.last_speech_activity > config.COMPLETE_SILENCE_SEC):
                        self.process_complete_message()
                    
        except KeyboardInterrupt:
            print("\n\n✋ Finished. Bye!")
        finally:
            # Stop terminal detector if we started it
            if terminal_detector_started:
                self._stop_terminal_detector()
            
            # Unsubscribe from shared stream
            audio_manager.unsubscribe_from_stream("StreamingChatbot")
            # Do NOT stop the shared stream here – other components (wake-word, terminal detector)
            # keep using it.  The stream is closed only once during full program shutdown.
    
    def _shared_audio_callback(self, audio_data):
        """Callback for shared audio stream."""
        # Convert numpy array back to bytes for compatibility with existing code
        audio_bytes = audio_data.astype(np.int16).tobytes()
        self.audio_queue.put(audio_bytes)
    
    def _start_terminal_detector(self) -> bool:
        """Start terminal word detector during conversation."""
        try:
            # Only start if terminal word detection is enabled
            if not config.TERMINAL_WORD_ENABLED:
                return False
                
            # Import here to avoid circular imports
            from core.components import get_global_orchestrator, ComponentState
            
            # Get the orchestrator instance
            orchestrator = get_global_orchestrator()
            if not orchestrator or not orchestrator.terminal_word_detector:
                if config.DEBUG_MODE:
                    print("⚠️ No terminal word detector available")
                return False
            
            # Ensure the shared stream is available
            from core.components import SharedAudioManager
            audio_manager = SharedAudioManager()
            stream_info = audio_manager.get_stream_info()
            
            if not stream_info.get("active"):
                if config.DEBUG_MODE:
                    print("⚠️ Shared stream not active, cannot start terminal detector")
                return False
            
            # Start terminal detector with shared stream
            terminal_detector = orchestrator.terminal_word_detector
            if terminal_detector.state == ComponentState.STOPPED:
                if config.DEBUG_MODE:
                    print("🛑 Starting terminal word detector with shared stream...")
                
                if terminal_detector.start_shared_stream():
                    if config.DEBUG_MODE:
                        print("✅ Terminal word detector started with shared stream")
                    return True
                else:
                    if config.DEBUG_MODE:
                        print("❌ Failed to start terminal word detector with shared stream")
                    return False
            else:
                if config.DEBUG_MODE:
                    print(f"ℹ️ Terminal word detector already in state: {terminal_detector.state.value}")
                return terminal_detector.state == ComponentState.RUNNING
            
        except Exception as e:
            print(f"⚠️ Error starting terminal detector: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return False
    
    def _stop_terminal_detector(self):
        """Stop terminal word detector after conversation."""
        try:
            if not config.TERMINAL_WORD_ENABLED:
                return
                
            # Import here to avoid circular imports
            from core.components import get_global_orchestrator, ComponentState
            
            # Get the orchestrator instance
            orchestrator = get_global_orchestrator()
            if not orchestrator or not orchestrator.terminal_word_detector:
                return
            
            # Stop terminal detector
            terminal_detector = orchestrator.terminal_word_detector
            if terminal_detector.state == ComponentState.RUNNING:
                if config.DEBUG_MODE:
                    print("🛑 Stopping terminal word detector...")
                
                terminal_detector.stop()
                if config.DEBUG_MODE:
                    print("✅ Terminal word detector stopped")
                    
        except Exception as e:
            print(f"⚠️ Error stopping terminal detector: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    def _use_exclusive_audio_fallback(self):
        """Fallback to exclusive audio access when shared stream fails."""
        print("🔄 Using exclusive audio access (fallback mode)")
        
        # Request exclusive audio access
        from core.components import SharedAudioManager
        audio_manager = SharedAudioManager()
        
        if not audio_manager.request_audio_access("StreamingChatbot", timeout=5.0):
            print("❌ Could not obtain exclusive audio access for conversation")
            return
        
        try:
            # Use traditional exclusive audio stream
            with sd.RawInputStream(
                samplerate=config.SAMPLE_RATE,
                blocksize=config.FRAME_SIZE,
                dtype="int16",
            channels=1,
            callback=self.audio_callback,
        ):
                print("✅ Exclusive audio stream created successfully")
                print("ℹ️ Note: Terminal word detection limited to transcription only in this mode")
                
            try:
                while not self.session_ended:
                    # Get audio frame
                    frame = self.audio_queue.get()
                    current_time = time.time()
                    
                    # Check if frame contains speech
                    frame_has_speech = self.chunker.is_speech(frame)
                    
                    # Update last speech activity
                    if frame_has_speech:
                        self.last_speech_activity = current_time
                    
                    # Process frame for chunks
                    chunk = self.chunker.process(frame)
                    if chunk:
                        self.process_chunk(chunk)
                        # Exit immediately if session ended during chunk processing
                        if self.session_ended:
                            break
                    
                    # Check for end of complete message
                    if (self.accumulated_chunks and 
                        self.last_speech_activity and 
                        current_time - self.last_speech_activity > config.COMPLETE_SILENCE_SEC):
                        self.process_complete_message()
                        
            except KeyboardInterrupt:
                print("\n\n✋ Finished. Bye!")
            
            # Handle graceful session termination
            if self.session_ended:
                print("\n🔚 Session ended gracefully. Goodbye!")
                print("🔄 Returning to wake word detection...")
                    
        except Exception as e:
            print(f"❌ Failed to create exclusive audio stream: {e}")
            # Try one more time with macOS recovery
            if platform.system() == 'Darwin':
                try:
                    print("🔄 Attempting macOS audio recovery...")
                    sd._terminate()
                    sd._initialize()
                    time.sleep(1.0)
                    
                    # Try again with different parameters
                    with sd.RawInputStream(
                        samplerate=config.SAMPLE_RATE,
                        blocksize=1024,  # Larger block size
                        dtype="int16",
                        channels=1,
                        callback=self.audio_callback,
                        latency='high'
                    ):
                        print("✅ Audio stream created with recovery parameters")
                        
                        while not self.session_ended:
                            frame = self.audio_queue.get()
                            current_time = time.time()
                            
                            frame_has_speech = self.chunker.is_speech(frame)
                            if frame_has_speech:
                                self.last_speech_activity = current_time
                            
                            chunk = self.chunker.process(frame)
                            if chunk:
                                self.process_chunk(chunk)
                                if self.session_ended:
                                    break
                            
                            if (self.accumulated_chunks and 
                                self.last_speech_activity and 
                                current_time - self.last_speech_activity > config.COMPLETE_SILENCE_SEC):
                                self.process_complete_message()
                                
                except Exception as recovery_error:
                    print(f"❌ Recovery also failed: {recovery_error}")
                    print("🚨 Audio system may require manual intervention")
                    
        finally:
            # Always release audio access when conversation ends
            audio_manager.release_audio_access("StreamingChatbot")


class ToolEnabledStreamingChatbot(StreamingChatbot):
    """Streaming chatbot with MCP tool support."""
    
    def __init__(self):
        """Initialize tool-enabled chatbot."""
        super().__init__()
        
        # Import MCP server here to avoid circular imports
        try:
            from mcp_server.server import MCPServer
            self.mcp_server = MCPServer()
            
            # Convert MCP tools to OpenAI functions
            self.functions = []
            for tool in self.mcp_server.list_tools():
                self.functions.append({
                    "name": tool['name'],
                    "description": self.mcp_server.get_tool_info(tool['name'])['description'],
                    "parameters": tool['schema']
                })
            
            print(f"🔧 Loaded {len(self.functions)} tools: {', '.join([f['name'] for f in self.functions])}")
        except Exception as e:
            print(f"⚠️  Could not load MCP tools: {e}")
            self.mcp_server = None
            self.functions = []
    
    def process_complete_message(self) -> None:
        """Process accumulated chunks with tool support."""
        if not self.accumulated_chunks:
            return
            
        print("⏹️  [MESSAGE END] Complete silence detected, processing full message")
        
        # Combine all chunks
        complete_message = " ".join(self.accumulated_chunks)
        print(f"\n📝  Complete message: {complete_message}")
        
        # Check for end conversation phrases in complete message
        complete_message_lower = complete_message.lower()
        for phrase in config.END_CONVERSATION_PHRASES:
            if phrase in complete_message_lower:
                print(f"🚫 End phrase '{phrase}' detected in complete message, stopping conversation")
                print("🤖 Roger that! Conversation ending...")
                self.session_ended = True
                self.accumulated_chunks.clear()
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        # Check for terminal phrases in complete message
        for phrase in config.TERMINAL_PHRASES:
            if phrase in complete_message_lower:
                print(f"🛑 Terminal phrase '{phrase}' detected in complete message, returning to wake word mode")
                self.session_ended = True
                self.accumulated_chunks.clear()
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        print("🤖  Sending to ChatGPT...")
        
        # Add timestamp to queries about current information to force fresh tool calls
        enhanced_message = self._enhance_message_for_freshness(complete_message)
        
        # Add to conversation and get response with tools
        self.conversation.add_user_message(enhanced_message)
        
        # Debug: Show conversation history length and content
        all_messages = self.conversation.get_messages()
        print(f"📊 Conversation has {len(all_messages)} messages")
        print("📜 Recent messages:")
        for i, msg in enumerate(all_messages[-3:]):  # Show last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', 'no content')[:100]  # First 100 chars
            print(f"   {i}: {role}: {content}")
        
        # Start conversation processing timer
        conversation_start_time = time.time()
        
        # First, check if we need to use tools (use focused context and lower temperature)
        if self.functions:
            tool_context = self.conversation.get_tool_context(max_messages=config.TOOL_CONTEXT_SIZE)
            print(f"🔧 Checking for tool usage (temperature: {config.TOOL_TEMPERATURE}, context: {len(tool_context)} messages)")
            response = self.speech_services.chat_completion(
                tool_context,  # Use focused context for tool selection
                temperature=config.TOOL_TEMPERATURE,  # Lower temperature for tool selection
                functions=self.functions
            )
        else:
            response_context = self.conversation.get_response_context()
            print(f"🤖 No tools available, using regular chat (temperature: {config.RESPONSE_TEMPERATURE}, context: {len(response_context)} messages)")
            response = self.speech_services.chat_completion(
                response_context,  # Use full context for response
                temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for responses
            )
        

        
        if response:
            # Handle Realtime API function calls
            if response.get("function_call") and response.get("call_id") and self.mcp_server and USING_REALTIME_API:
                # Realtime API function call
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])
                call_id = response["call_id"]
                
                print(f"🔧 [Realtime Tool: {func_name}]")
                
                # Execute tool
                tool_result = self.mcp_server.execute_tool(func_name, func_args)
                
                # Send result back to Realtime session
                if hasattr(self.speech_services, 'execute_function_call_realtime'):
                    self.speech_services.execute_function_call_realtime({
                        "name": func_name,
                        "arguments": response["function_call"]["arguments"],
                        "call_id": call_id
                    }, self.mcp_server)
                
                # Trigger response generation to get final answer
                if hasattr(self.speech_services, 'trigger_response'):
                    self.speech_services.trigger_response()
                
                # Wait for final response
                try:
                    final_response = self.speech_services.response_queue.get(timeout=15.0)
                    if final_response and final_response.get("content"):
                        self.conversation.add_assistant_message(final_response["content"])
                        print(f"🤖  GPT: {final_response['content']}\n")
                        
                        # Convert response to speech if TTS is enabled
                        if self.speech_services.tts_enabled:
                            audio_file = self.speech_services.text_to_speech(final_response["content"])
                            if audio_file and hasattr(self, 'chunker') and self.chunker.aec_enabled:
                                # Add TTS audio as reference for AEC
                                self.chunker.add_reference_audio_file(audio_file)
                        
                        print("─" * 50)
                except queue.Empty:
                    print("⚠️ No response after function execution")
            
            # Handle new tool_calls format (multiple tools) - Traditional API
            elif response.get("tool_calls") and self.mcp_server:
                # Add assistant message with tool calls
                self.conversation.messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response["tool_calls"]
                })
                
                # Execute each tool call in sequence
                print(f"🔧 [Executing {len(response['tool_calls'])} tool(s)]")
                for tool_call in response["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]
                    
                    print(f"  → Tool: {func_name}")
                    
                    # Execute tool
                    tool_result = self.mcp_server.execute_tool(func_name, func_args)
                    
                    # Add tool result to conversation
                    self.conversation.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result)
                    })
                
                # Get final response after all tool executions (use higher temperature for creative response)
                print(f"🎨 Generating final response (temperature: {config.RESPONSE_TEMPERATURE})")
                response_context = self.conversation.get_response_context()
                print(f"📝 Using full conversation context ({len(response_context)} messages) for response generation")
                final_response = self.speech_services.chat_completion(
                    response_context,  # Use full context for response generation
                    temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for creative responses
                )
                
                # End conversation processing timer
                conversation_end_time = time.time()
                conversation_latency = conversation_end_time - conversation_start_time
                self.conversation_latencies.append(conversation_latency)
                
                if final_response and final_response.get("content"):
                    self.conversation.add_assistant_message(final_response["content"])
                    print(f"🤖  GPT: {final_response['content']} (processing: {conversation_latency:.3f}s)\n")
                    
                    # Convert response to speech if TTS is enabled
                    if self.speech_services.tts_enabled:
                        audio_file = self.speech_services.text_to_speech(final_response["content"])
                        if audio_file and hasattr(self, 'chunker') and self.chunker.aec_enabled:
                            # Add TTS audio as reference for AEC
                            self.chunker.add_reference_audio_file(audio_file)
                    
                    print("─" * 50)
            
            # Handle old function_call format (backward compatibility)
            elif response.get("function_call") and self.mcp_server:
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])
                
                print(f"🔧 [Using tool: {func_name}]")
                
                # Execute tool
                tool_result = self.mcp_server.execute_tool(func_name, func_args)
                
                # Add function call and result to conversation
                self.conversation.messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": func_name, 
                        "arguments": response["function_call"]["arguments"]
                    }
                })
                self.conversation.messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(tool_result)
                })
                
                # Get final response after tool execution (use higher temperature for creative response)
                print(f"🎨 Generating final response (temperature: {config.RESPONSE_TEMPERATURE})")
                final_response = self.speech_services.chat_completion(
                    self.conversation.get_messages(),
                    temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for final response
                )
                
                # End conversation processing timer
                conversation_end_time = time.time()
                conversation_latency = conversation_end_time - conversation_start_time
                self.conversation_latencies.append(conversation_latency)
                
                if final_response and final_response.get("content"):
                    self.conversation.add_assistant_message(final_response["content"])
                    print(f"🤖  GPT: {final_response['content']} (processing: {conversation_latency:.3f}s)\n")
                    
                    # Convert response to speech if TTS is enabled
                    if self.speech_services.tts_enabled:
                        audio_file = self.speech_services.text_to_speech(final_response["content"])
                        if audio_file and hasattr(self, 'chunker') and self.chunker.aec_enabled:
                            # Add TTS audio as reference for AEC
                            self.chunker.add_reference_audio_file(audio_file)
                    
                    print("─" * 50)
            
            elif response.get("content"):
                # Regular response without tools (already using higher temperature)
                # End conversation processing timer
                conversation_end_time = time.time()
                conversation_latency = conversation_end_time - conversation_start_time
                self.conversation_latencies.append(conversation_latency)
                
                self.conversation.add_assistant_message(response["content"])
                print(f"🤖  GPT: {response['content']} (processing: {conversation_latency:.3f}s)\n")
                
                # Convert response to speech if TTS is enabled
                if self.speech_services.tts_enabled:
                    audio_file = self.speech_services.text_to_speech(response["content"])
                    if audio_file and hasattr(self, 'chunker') and self.chunker.aec_enabled:
                        # Add TTS audio as reference for AEC
                        self.chunker.add_reference_audio_file(audio_file)
                
                print("─" * 50)
        
        # Reset for next message
        self.accumulated_chunks = []
        self.last_speech_activity = None
        self.chunk_transcription_times = []  # Reset chunk latency tracking
        self.last_transcription_end_time = None

    def process_text_input(self, user_input: str):
        """Process direct text input for testing purposes."""
        print(f"\n📝 Processing: {user_input}")
        
        # Add timestamp to queries about current information to force fresh tool calls
        enhanced_message = self._enhance_message_for_freshness(user_input)
        
        # Add to conversation and get response with tools
        self.conversation.add_user_message(enhanced_message)
        
        # First, check if we need to use tools (use focused context and lower temperature)
        if self.functions:
            tool_context = self.conversation.get_tool_context(max_messages=config.TOOL_CONTEXT_SIZE)
            print(f"🔧 Checking for tool usage (temperature: {config.TOOL_TEMPERATURE}, context: {len(tool_context)} messages)")
            response = self.speech_services.chat_completion(
                tool_context,  # Use focused context for tool selection
                temperature=config.TOOL_TEMPERATURE,  # Lower temperature for tool selection
                functions=self.functions
            )
        else:
            response_context = self.conversation.get_response_context()
            print(f"🤖 No tools available, using regular chat (temperature: {config.RESPONSE_TEMPERATURE}, context: {len(response_context)} messages)")
            response = self.speech_services.chat_completion(
                response_context,  # Use full context for response
                temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for responses
            )
        
        final_response_content = None
        
        if response:
            # Handle new tool_calls format (multiple tools)
            if response.get("tool_calls") and self.mcp_server:
                # Add assistant message with tool calls
                self.conversation.messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response["tool_calls"]
                })
                
                # Execute each tool call in sequence
                print(f"🔧 [Executing {len(response['tool_calls'])} tool(s)]")
                for tool_call in response["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]
                    
                    print(f"  → Tool: {func_name} with args: {func_args}")
                    
                    # Execute tool
                    tool_result = self.mcp_server.execute_tool(func_name, func_args)
                    print(f"  → Result: {tool_result}")
                    
                    # Add tool result to conversation
                    self.conversation.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result)
                    })
                
                # Get final response after all tool executions
                print(f"🎨 Generating final response (temperature: {config.RESPONSE_TEMPERATURE})")
                response_context = self.conversation.get_response_context()
                print(f"📝 Using full conversation context ({len(response_context)} messages) for response generation")
                final_response = self.speech_services.chat_completion(
                    response_context,  # Use full context for response generation
                    temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for creative responses
                )
                
                if final_response and final_response.get("content"):
                    self.conversation.add_assistant_message(final_response["content"])
                    final_response_content = final_response["content"]
                else:
                    final_response_content = "Tools executed successfully but no response generated."
            
            # Handle old function_call format (backward compatibility)
            elif response.get("function_call") and self.mcp_server:
                func_name = response["function_call"]["name"]
                func_args = json.loads(response["function_call"]["arguments"])
                
                print(f"🔧 [Using tool: {func_name} with args: {func_args}]")
                
                # Execute tool
                tool_result = self.mcp_server.execute_tool(func_name, func_args)
                print(f"🔧 [Tool result: {tool_result}]")
                
                # Add function call and result to conversation
                self.conversation.messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": func_name, 
                        "arguments": response["function_call"]["arguments"]
                    }
                })
                self.conversation.messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(tool_result)
                })
                
                # Get final response after tool execution (use higher temperature for creative response)
                print(f"🎨 Generating final response (temperature: {config.RESPONSE_TEMPERATURE})")
                final_response = self.speech_services.chat_completion(
                    self.conversation.get_messages(),
                    temperature=config.RESPONSE_TEMPERATURE  # Higher temperature for final response
                )
                
                if final_response and final_response.get("content"):
                    self.conversation.add_assistant_message(final_response["content"])
                    final_response_content = final_response["content"]
                else:
                    final_response_content = "Tool executed successfully but no response generated."
            
            elif response.get("content"):
                # Regular response without tools
                self.conversation.add_assistant_message(response["content"])
                final_response_content = response["content"]
        
        if final_response_content is None:
            final_response_content = "No response generated."
        
        # Save test conversation to database
        try:
            print("💾 Saving test conversation to database...")
            self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
            print("✅ Test conversation saved successfully")
        except Exception as db_error:
            print(f"⚠️ Failed to save test conversation to database: {db_error}")
        
        return final_response_content


def main():
    """Entry point with tool selection."""
    print("🎤 Streaming Chatbot")
    print("Choose mode:")
    print("1. Basic chatbot (no tools)")
    print("2. Tool-enabled chatbot (with MCP tools)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    try:
        if choice == '1':
            print("\n🤖 Starting basic streaming chatbot...")
            chatbot = StreamingChatbot()
        else:
            print("\n🔧 Starting tool-enabled streaming chatbot...")
            chatbot = ToolEnabledStreamingChatbot()
        
        chatbot.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. OPENAI_KEY is set in your environment or .env file")
        print("2. You have installed all dependencies:")
        print("   pip install openai sounddevice webrtcvad numpy scipy python-dotenv")
        print("3. Your microphone is properly configured")

def main_wakeword():
    print("🎤 Streaming Chatbot")
    print("Choose mode:")
    chatbot = ToolEnabledStreamingChatbot()
    chatbot.run()

