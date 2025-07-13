"""
Real-time streaming chatbot using OpenAI's Realtime API.
Provides continuous audio streaming and real-time transcription for lower latency conversations.
"""

import os
import sys
import time
import queue
import platform
import json
import base64
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Optional
import logging
import numpy as np
from datetime import datetime

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


class RealtimeStreamingChatbot:
    """Real-time streaming chatbot using OpenAI's Realtime API."""
    
    def __init__(self):
        """Initialize real-time streaming chatbot."""
        openai_api_key = os.getenv("OPENAI_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
            
        # Get Gemini API key if using Gemini (fallback only)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        self.speech_services = SpeechServices(
            openai_api_key=openai_api_key,
            whisper_model=config.WHISPER_MODEL,
            chat_provider=config.CHAT_PROVIDER,
            chat_model="gpt-4o-realtime-preview",  # Force realtime model
            gemini_api_key=gemini_api_key,
            tts_enabled=config.TTS_ENABLED,
            tts_model=config.TEXT_TO_SPEECH_MODEL,
            tts_voice=config.TTS_VOICE
        )
        
        self.conversation = ConversationManager(config.SYSTEM_PROMPT)
        self.audio_queue = queue.Queue()
        
        # VAD for speech detection
        self.vad_chunker = VADChunker(
            sample_rate=config.SAMPLE_RATE,
            frame_ms=config.FRAME_MS,
            vad_mode=config.VAD_MODE,
            silence_end_sec=config.SILENCE_END_SEC,
            max_utterance_sec=config.MAX_UTTERANCE_SEC,
            aec_enabled=config.AEC_ENABLED
        )
        
        # State tracking
        self.session_ended = False
        self.is_streaming = False
        self.current_transcript = ""
        self.transcript_complete = False
        self.current_item_id = None  # Track current conversation item ID
        self.awaiting_response = False  # Flag to track when we're waiting for a response
        
        # Tool support
        self.mcp_server = None
        self.functions = []
        self._init_tools()
        
        # Speech activity tracking
        self.last_speech_activity = None
        self.speech_detected = False
        
        # Real-time latency tracking
        self.frame_timestamps = []  # Track when frames are sent
        self.speech_end_time = None  # Track when speech ends
        self.transcription_latencies = []  # Speech end to transcription latencies
        self.conversation_latencies = []  # Complete conversation turn latencies
        self.session_processing_latencies = []  # Processing time after speech ends
        self.session_final_latencies = []  # Turn completion latencies
        
        # Streaming control
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
    
    def _init_tools(self):
        """Initialize MCP tools for realtime chatbot."""
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
                
            if self.functions:
                print(f"🔧 Loaded {len(self.functions)} MCP tools for realtime chatbot")
            
        except ImportError as e:
            print(f"⚠️ MCP server not available: {e}")
            self.mcp_server = None
            self.functions = []
    
    def _trigger_response_with_tools(self):
        """Trigger response generation with tools configured."""
        if not self.speech_services.use_realtime or not self.speech_services.is_connected:
            return
            
        try:
            # Ensure temperature is at least 0.6 for API compliance
            safe_temperature = max(0.6, getattr(config, 'RESPONSE_TEMPERATURE', 0.7))
            
            message = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "temperature": safe_temperature,
                    "max_output_tokens": getattr(config, 'REALTIME_MAX_RESPONSE_TOKENS', 150),
                    "tools": self.speech_services._convert_functions_to_tools(self.functions)
                }
            }
            
            if self.speech_services.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.speech_services.websocket.send(json.dumps(message)),
                    self.speech_services.loop
                )
            
            print(f"🔧 Triggered response with {len(self.functions)} tools available")
            
        except Exception as e:
            print(f"⚠️ Error triggering response with tools: {e}")
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Queue audio with timestamp for latency tracking
        audio_bytes = bytes(indata)
        timestamp = time.time()
        self.audio_queue.put((audio_bytes, timestamp))
    
    def send_to_db(self, chat: dict):
        """Send chat to database with latency metrics."""
        with DBConnect() as db:
            # Log latency summary if we have data
            if self.session_processing_latencies:
                total_processing = sum(self.session_processing_latencies)
                avg_processing = total_processing / len(self.session_processing_latencies)
                print(f"⏱️  Processing latencies (speech end → transcription): {len(self.session_processing_latencies)} turns, total: {total_processing:.3f}s, avg: {avg_processing:.3f}s")
            
            if self.session_final_latencies:
                total_conversation = sum(self.session_final_latencies)
                avg_conversation = total_conversation / len(self.session_final_latencies)
                print(f"⏱️  Response generation latencies: {len(self.session_final_latencies)} turns, total: {total_conversation:.3f}s, avg: {avg_conversation:.3f}s")
            
            # Save to database with latency metrics
            db.insert_new_chat(
                session_time=datetime.now(), 
                chat_data=chat,
                accumulated_latencies=self.session_processing_latencies if self.session_processing_latencies else None,
                final_latencies=self.session_final_latencies if self.session_final_latencies else None
            )

    def process_audio_frame(self, audio_bytes: bytes, timestamp: float) -> None:
        """Stream audio frame to realtime API with VAD filtering."""
        if not self.is_streaming:
            return
            
        # For Realtime API with server VAD, we should stream all audio continuously
        # The server will handle voice activity detection and turn management
        
        # Store timestamp for latency calculation
        self.frame_timestamps.append(timestamp)
        
        # Stream frame to realtime API
        try:
            self.speech_services.send_audio_frame(audio_bytes)
        except Exception as e:
            print(f"⚠️ Error streaming audio frame: {e}")
    
    def handle_partial_transcript(self, transcript: str) -> None:
        """Handle partial transcription from realtime API."""
        if transcript and transcript != self.current_transcript:
            self.current_transcript = transcript
            # Use carriage return to overwrite the same line
            print(f"\r🎤  {transcript}", end="", flush=True)
    
    def handle_final_transcript(self, transcript: str, item_id: str = None) -> None:
        """Handle complete transcription from realtime API."""
        if not transcript:
            return
        
        # Store the item_id for later use in response
        self.current_item_id = item_id
            
        # Clear the partial transcript line and show final
        if getattr(config, 'REALTIME_STREAM_TRANSCRIPTION', False):
            print(f"\r🎤  Final: {transcript}")
        else:
            print(f"🎤  Final: {transcript}")
        
        # Calculate processing latency from when speech ended
        current_time = time.time()
        if self.speech_end_time:
            processing_latency = current_time - self.speech_end_time
            self.session_processing_latencies.append(processing_latency)
            print(f"⏱️  Processing latency (speech end → transcription): {processing_latency:.3f}s")
        elif self.frame_timestamps:
            # Fallback to old method if we don't have speech end time
            total_duration = current_time - self.frame_timestamps[0]
            print(f"⏱️  Total duration (first frame → transcription): {total_duration:.3f}s")
        
        # Check for end conversation phrases
        transcript_lower = transcript.lower()
        for phrase in config.END_CONVERSATION_PHRASES:
            if phrase in transcript_lower:
                print(f"🚫 End phrase '{phrase}' detected, stopping conversation")
                print("🤖 Roger that! Conversation ending...")
                self.session_ended = True
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        # Check for terminal phrases
        for phrase in config.TERMINAL_PHRASES:
            if phrase in transcript_lower:
                print(f"🛑 Terminal phrase '{phrase}' detected, returning to wake word mode")
                self.session_ended = True
                self.send_to_db(self.conversation.get_chat_minus_sys_prompt())
                return
        
        # Add user message to conversation
        enhanced_message = self._enhance_message_for_freshness(transcript)
        self.conversation.add_user_message(enhanced_message)
        
        # With server VAD enabled, the response should be generated automatically
        # We just need to wait for it - no manual triggering required
        print("🤖  Waiting for response...")
        self.awaiting_response = True
        
        # Reset for next utterance
        self.current_transcript = ""
        self.frame_timestamps = []
        self.speech_end_time = None
        self.speech_detected = False
    
    def _on_speech_stopped(self) -> None:
        """Callback when speech stops."""
        self.speech_end_time = time.time()
        self.speech_detected = False
        if config.DEBUG_MODE:
            print("🔇 Speech stopped (VAD detected silence)")
    
    def process_complete_transcript(self, transcript: str) -> None:
        """Process complete transcript and get response."""
        print("🤖  Processing with ChatGPT...")
        
        # Start timing for final latency
        final_latency_start = time.time()
        
        # Add timestamp to queries about current information to force fresh tool calls
        enhanced_message = self._enhance_message_for_freshness(transcript)
        
        # Add to conversation
        self.conversation.add_user_message(enhanced_message)
        
        # Get response using realtime API
        response_context = self.conversation.get_response_context()
        print(f"🤖 Using realtime chat (context: {len(response_context)} messages)")
        
        response = self.speech_services.chat_completion(
            response_context,
            temperature=config.RESPONSE_TEMPERATURE,
            item_id=self.current_item_id
        )
        
        # Calculate final latency
        final_latency = time.time() - final_latency_start
        self.session_final_latencies.append(final_latency)
        print(f"⏱️  Final processing latency: {final_latency:.3f}s")
        
        if response and response.get("content"):
            self.conversation.add_assistant_message(response["content"])
            print(f"🤖  GPT: {response['content']}\n")
            
            # Convert response to speech if TTS is enabled
            if self.speech_services.tts_enabled:
                audio_file = self.speech_services.text_to_speech(response["content"])
            
            print("─" * 50)
    
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
                # Clear everything except the system message for calendar queries
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
    
    def start_realtime_session(self) -> bool:
        """Start realtime API session."""
        try:
            # Ensure connection to realtime API
            if not self.speech_services._ensure_connected():
                print("❌ Failed to connect to realtime API")
                return False
            
            print("✅ Connected to OpenAI Realtime API")
            
            # Configure tools in the session if available
            if self.functions:
                print(f"🔧 Configuring {len(self.functions)} tools in session...")
                self.speech_services.update_session_tools(self.functions)
            
            self.is_streaming = True
            
            # Set up callbacks for realtime events
            self.speech_services.set_callbacks(
                partial_transcript_callback=self.handle_partial_transcript if getattr(config, 'REALTIME_STREAM_TRANSCRIPTION', False) else None,
                speech_stopped_callback=self._on_speech_stopped
            )
            
            # Start message handling thread
            self._streaming_thread = threading.Thread(target=self._handle_realtime_messages, daemon=True)
            self._streaming_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting realtime session: {e}")
            return False
    
    def _handle_realtime_messages(self):
        """Handle realtime API messages in background thread."""
        try:
            while self.is_streaming and not self.session_ended:
                try:
                    # Check for transcriptions
                    try:
                        result = self.speech_services.transcription_queue.get(timeout=0.1)
                        if result:
                            # Handle both old format (string) and new format (dict)
                            if isinstance(result, dict):
                                transcript = result.get("text", "")
                                item_id = result.get("item_id")
                                self.handle_final_transcript(transcript, item_id)
                            else:
                                self.handle_final_transcript(result)
                    except queue.Empty:
                        pass
                    
                    # Check for function calls first if we have tools
                    if self.awaiting_response and self.functions:
                        try:
                            function_call = self.speech_services.check_for_function_calls(timeout=0.1)
                            if function_call and self.mcp_server:
                                print(f"🔧 Executing tool: {function_call.get('name')}")
                                result = self.speech_services.execute_function_call_realtime(function_call, self.mcp_server)
                                if result:
                                    print(f"✅ Tool executed successfully")
                                # Don't reset awaiting_response yet - wait for the final text response
                        except queue.Empty:
                            pass
                    
                    # Check for responses - only process if we're actually waiting for one
                    if self.awaiting_response:
                        try:
                            response = self.speech_services.response_queue.get(timeout=0.1)
                            if response and response.get("content"):
                                # Process the response
                                self.conversation.add_assistant_message(response["content"])
                                print(f"🤖  GPT: {response['content']}\n")
                                
                                # Convert response to speech if TTS is enabled
                                if self.speech_services.tts_enabled:
                                    audio_file = self.speech_services.text_to_speech(response["content"])
                                
                                print("─" * 50)
                                self.awaiting_response = False  # Reset the flag
                        except queue.Empty:
                            pass
                        
                except Exception as inner_e:
                    print(f"⚠️ Error processing realtime messages: {inner_e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"⚠️ Error in realtime message handler: {e}")
    
    def stop_realtime_session(self):
        """Stop realtime API session."""
        if self.is_streaming:
            self.is_streaming = False
            print("🛑 Stopped realtime session")
    
    def run(self):
        """Main run loop for realtime streaming chatbot."""
        print("\n🎤  Ready for realtime conversation. Speak into the microphone (Ctrl-C to quit)…")
        print("💡  Using OpenAI Realtime API for continuous streaming transcription.")
        print("─" * 50)
        
        # Start realtime session
        if not self.start_realtime_session():
            print("❌ Failed to start realtime session, falling back to chunk-based mode")
            # Could implement fallback to original streaming_chatbot here
            return
        
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
            print("⚠️ Could not create shared audio stream, using exclusive mode")
            self._use_exclusive_audio_fallback()
            return
        
        # Subscribe to shared stream
        if not audio_manager.subscribe_to_stream("RealtimeStreamingChatbot", self._shared_audio_callback):
            print("❌ Could not subscribe to shared audio stream")
            return
        
        # Start terminal word detector if available
        terminal_detector_started = self._start_terminal_detector()
        
        try:
            while not self.session_ended:
                # Get audio frame from queue (populated by shared stream callback)
                try:
                    audio_bytes, timestamp = self.audio_queue.get(timeout=0.1)
                    self.process_audio_frame(audio_bytes, timestamp)
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n✋ Finished. Bye!")
        finally:
            # Stop realtime session
            self.stop_realtime_session()
            
            # Stop terminal detector if we started it
            if terminal_detector_started:
                self._stop_terminal_detector()
            
            # Unsubscribe from shared stream
            audio_manager.unsubscribe_from_stream("RealtimeStreamingChatbot")
    
    def _shared_audio_callback(self, audio_data):
        """Callback for shared audio stream."""
        # Convert numpy array back to bytes for compatibility with existing code
        audio_bytes = audio_data.astype(np.int16).tobytes()
        timestamp = time.time()
        self.audio_queue.put((audio_bytes, timestamp))
    
    def _start_terminal_detector(self) -> bool:
        """Start terminal word detector during conversation."""
        try:
            # Only start if terminal word detection is enabled
            if not hasattr(config, 'TERMINAL_WORD_ENABLED') or not config.TERMINAL_WORD_ENABLED:
                return False
                
            # Import here to avoid circular imports
            from core.components import get_global_orchestrator, ComponentState
            
            # Get the orchestrator instance
            orchestrator = get_global_orchestrator()
            if not orchestrator or not orchestrator.terminal_word_detector:
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print("⚠️ No terminal word detector available")
                return False
            
            # Ensure the shared stream is available
            from core.components import SharedAudioManager
            audio_manager = SharedAudioManager()
            stream_info = audio_manager.get_stream_info()
            
            if not stream_info.get("active"):
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print("⚠️ Shared stream not active, cannot start terminal detector")
                return False
            
            # Start terminal detector with shared stream
            terminal_detector = orchestrator.terminal_word_detector
            if terminal_detector.state == ComponentState.STOPPED:
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print("🛑 Starting terminal word detector with shared stream...")
                
                if terminal_detector.start_shared_stream():
                    if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                        print("✅ Terminal word detector started with shared stream")
                    return True
                else:
                    if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                        print("❌ Failed to start terminal word detector with shared stream")
                    return False
            else:
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print(f"ℹ️ Terminal word detector already in state: {terminal_detector.state.value}")
                return terminal_detector.state == ComponentState.RUNNING
            
        except Exception as e:
            print(f"⚠️ Error starting terminal detector: {e}")
            if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return False
    
    def _stop_terminal_detector(self):
        """Stop terminal word detector after conversation."""
        try:
            if not hasattr(config, 'TERMINAL_WORD_ENABLED') or not config.TERMINAL_WORD_ENABLED:
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
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print("🛑 Stopping terminal word detector...")
                
                terminal_detector.stop()
                if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print("✅ Terminal word detector stopped")
                    
        except Exception as e:
            print(f"⚠️ Error stopping terminal detector: {e}")
            if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    def _use_exclusive_audio_fallback(self):
        """Fallback to exclusive audio access when shared stream fails."""
        print("🔄 Using exclusive audio access (fallback mode)")
        
        # Request exclusive audio access
        from core.components import SharedAudioManager
        audio_manager = SharedAudioManager()
        
        if not audio_manager.request_audio_access("RealtimeStreamingChatbot", timeout=5.0):
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
                
                while not self.session_ended:
                    # Get audio frame
                    try:
                        audio_bytes, timestamp = self.audio_queue.get(timeout=0.1)
                        self.process_audio_frame(audio_bytes, timestamp)
                    except queue.Empty:
                        continue
                        
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
                            try:
                                audio_bytes, timestamp = self.audio_queue.get(timeout=0.1)
                                self.process_audio_frame(audio_bytes, timestamp)
                            except queue.Empty:
                                continue
                                
                except Exception as recovery_error:
                    print(f"❌ Recovery also failed: {recovery_error}")
                    print("🚨 Audio system may require manual intervention")
                    
        finally:
            # Always release audio access when conversation ends
            audio_manager.release_audio_access("RealtimeStreamingChatbot")




def main():
    """Main function for realtime streaming chatbot."""
    try:
        chatbot = RealtimeStreamingChatbot()
        chatbot.run()
    except KeyboardInterrupt:
        print("\n\n✋ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main_wakeword():
    """Main function for wake word mode."""
    try:
        # Import here to avoid circular imports
        from core.components import Orchestrator
        
        # Create and run orchestrator
        orchestrator = Orchestrator()
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n\n✋ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()