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
from typing import List, Dict, Optional, Any
import logging
import numpy as np
from datetime import datetime

# Suppress common warnings from dependencies
from core.fixes.suppress_warnings import suppress_common_warnings
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
        
        # Interruption handling
        self.response_interrupted = False  # Track if current response was interrupted
        self.partial_response_content = ""  # Store partial response before interruption
        self.interruption_timestamp = None  # Track when interruption occurred
        
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
        
        # Pre-warm components to reduce startup latency
        self._audio_manager = None
        self._prewarmed = False
        
        # Cost tracking for audio filtering
        self.audio_frames_sent = 0
        self.audio_frames_filtered = 0
        self.cost_savings_start_time = time.time()
        
        # Audio duration tracking
        self.audio_session_start_time = None
        self.audio_duration_per_turn = []
        
    
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
                    "max_output_tokens": getattr(config, 'REALTIME_MAX_RESPONSE_TOKENS', 150)
                }
            }
            # Note: Tools are configured at session level, not per response for Realtime API
            
            if self.speech_services.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.speech_services.websocket.send(json.dumps(message)),
                    self.speech_services.loop
                )
            
            print(f"🔧 Triggered response with {len(self.functions)} tools available")
            print(f"   Tools: {[f['name'] for f in self.functions]}")
            
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
        # Clear session summary when conversation ends
        if getattr(config, 'DISPLAY_CONTEXT', False):
            try:
                import os
                if hasattr(config, 'session_summary_file') and os.path.exists(config.session_summary_file):
                    os.remove(config.session_summary_file)
                    if config.DEBUG_MODE:
                        print("📋 Session summary cleared")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"⚠️ Error clearing session summary: {e}")
        
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
            
            # Log cost savings from audio filtering
            total_frames = self.audio_frames_sent + self.audio_frames_filtered
            if total_frames > 0:
                filtering_percentage = (self.audio_frames_filtered / total_frames) * 100
                session_duration = time.time() - self.cost_savings_start_time
                print(f"💰 Audio cost savings: {filtering_percentage:.1f}% of frames filtered ({self.audio_frames_filtered}/{total_frames}) over {session_duration:.1f}s")
            
            # Log audio duration summary
            if self.audio_duration_per_turn:
                total_audio_duration = sum(self.audio_duration_per_turn)
                avg_audio_duration = total_audio_duration / len(self.audio_duration_per_turn)
                print(f"🎵 Audio duration summary: {len(self.audio_duration_per_turn)} turns, total: {total_audio_duration:.2f}s, avg: {avg_audio_duration:.2f}s per turn")
            
            
            # Save to database with latency metrics
            db.insert_new_chat(
                session_time=datetime.now(), 
                chat_data=chat,
                accumulated_latencies=self.session_processing_latencies if self.session_processing_latencies else None,
                final_latencies=self.session_final_latencies if self.session_final_latencies else None
            )

    def process_audio_frame(self, audio_bytes: bytes, timestamp: float) -> None:
        """Stream audio frame to realtime API with VAD filtering and cost optimization."""
        if not self.is_streaming:
            return
            
        # Check if we should mask audio (during greeting playback)
        if hasattr(self, '_audio_mask_until') and self._audio_mask_until > 0:
            if time.time() < self._audio_mask_until:
                # Skip processing during mask period
                return
            else:
                # Mask period just ended, log it once
                # Audio masking ended
                self._audio_mask_until = 0  # Reset to avoid repeated checks
        
        # Cost optimization: Track audio duration to filter out very short sounds
        if getattr(config, 'REALTIME_COST_OPTIMIZATION', True):
            if not hasattr(self, '_audio_start_time'):
                self._audio_start_time = timestamp
            
            # Calculate audio duration in milliseconds
            audio_duration_ms = (timestamp - self._audio_start_time) * 1000
            min_duration = getattr(config, 'REALTIME_MIN_AUDIO_LENGTH_MS', 500)
            
            # Don't process very short audio (likely noise/clicks)
            if audio_duration_ms < min_duration and not self.speech_detected:
                return
            
        # Client-side VAD filtering to reduce costs by not sending silence
        if getattr(config, 'REALTIME_COST_OPTIMIZATION', True) and getattr(config, 'REALTIME_CLIENT_VAD_ENABLED', True):
            # Use direct WebRTC VAD for frame-by-frame speech detection
            try:
                # Initialize VAD and debug tracking if not already done
                if not hasattr(self, '_simple_vad'):
                    import webrtcvad
                    vad_level = getattr(config, 'REALTIME_VAD_AGGRESSIVENESS', 0)
                    self._simple_vad = webrtcvad.Vad(vad_level)
                    self._vad_stats = {
                        'total_frames': 0,
                        'speech_frames': 0,
                        'silence_frames': 0,
                        'last_speech_time': 0,
                        'consecutive_silence': 0,
                        'vad_level': vad_level,
                        'fallback_active': False,
                        'fallback_start_time': 0,
                        'fallback_triggered_count': 0,
                        'currently_speaking': False,  # Track speech state changes
                        'last_state_change': 0
                    }
                
                # Calculate frame specifications
                frame_duration_ms = config.FRAME_MS
                expected_frame_size = config.SAMPLE_RATE * frame_duration_ms // 1000 * 2  # 16-bit = 2 bytes per sample
                # Ensure audio frame is the right size (WebRTC VAD needs exact frame sizes)
                if len(audio_bytes) != expected_frame_size:
                    # Pad or truncate to expected size
                    if len(audio_bytes) < expected_frame_size:
                        audio_bytes = audio_bytes + b'\x00' * (expected_frame_size - len(audio_bytes))
                    else:
                        audio_bytes = audio_bytes[:expected_frame_size]
                
                # Check for VAD fallback mode (temporarily disable VAD if too aggressive)
                current_time = time.time()
                
                # Check if we should activate fallback mode
                if (not self._vad_stats['fallback_active'] and 
                    self._vad_stats['consecutive_silence'] > 250 and  # ~5 seconds of silence
                    self._vad_stats['total_frames'] > 250):  # Only after initial period
                    
                    speech_ratio = self._vad_stats['speech_frames'] / self._vad_stats['total_frames']
                    if speech_ratio < 0.02:  # Less than 2% speech detected
                        self._vad_stats['fallback_active'] = True
                        self._vad_stats['fallback_start_time'] = current_time
                        self._vad_stats['fallback_triggered_count'] += 1
                        print(f"🆘 VAD fallback activated (speech ratio: {speech_ratio:.1%})")
                
                # Check if we should deactivate fallback mode
                if (self._vad_stats['fallback_active'] and 
                    current_time - self._vad_stats['fallback_start_time'] > 10):  # After 10 seconds
                    self._vad_stats['fallback_active'] = False
                    print("✅ VAD fallback deactivated")
                
                # Check if speech is present in this frame (or skip VAD if in fallback mode)
                if self._vad_stats['fallback_active']:
                    is_speech = True  # Treat all audio as speech during fallback
                else:
                    is_speech = self._simple_vad.is_speech(audio_bytes, config.SAMPLE_RATE)
                
                # Update statistics
                self._vad_stats['total_frames'] += 1
                if is_speech and not self._vad_stats['fallback_active']:  # Only count real speech detections
                    self._vad_stats['speech_frames'] += 1
                    self._vad_stats['last_speech_time'] = current_time
                    self._vad_stats['consecutive_silence'] = 0
                elif not self._vad_stats['fallback_active']:  # Only count when VAD is active
                    self._vad_stats['silence_frames'] += 1
                    self._vad_stats['consecutive_silence'] += 1
                
                # Track speech state changes for minimal logging
                speech_state_changed = False
                if is_speech != self._vad_stats['currently_speaking']:
                    self._vad_stats['currently_speaking'] = is_speech
                    self._vad_stats['last_state_change'] = current_time
                    speech_state_changed = True
                
                # Speech state changed but no logging to reduce spam
                
                if not is_speech:
                    # No speech detected - don't send to API to save costs
                    self.audio_frames_filtered += 1
                    return
                
                # Speech detected (or fallback mode active) - send to API will proceed
                
            except Exception as e:
                # If VAD fails, fall back to sending all audio
                print(f"⚠️ VAD error: {e}, sending audio anyway")
                if hasattr(self, '_vad_stats'):
                    self._vad_stats['total_frames'] += 1
                pass
        
        # Store timestamp for latency calculation
        self.frame_timestamps.append(timestamp)
        
        # Stream frame to realtime API
        try:
            self.speech_services.send_audio_frame(audio_bytes)
            self.audio_frames_sent += 1
        except Exception as e:
            print(f"⚠️ Error streaming audio frame: {e}")
    
    def handle_partial_transcript(self, transcript: str) -> None:
        """Handle partial transcription from realtime API."""
        if transcript and transcript != self.current_transcript:
            self.current_transcript = transcript
            # Use carriage return to overwrite the same line
            print(f"\r🎤  {transcript}", end="", flush=True)
    
    def _should_use_realtime_for_query(self, transcript: str) -> bool:
        """Determine if we should use Realtime API or fall back to traditional for cost savings."""
        import config
        
        # Always use Realtime API if cost optimization is disabled
        if not getattr(config, 'REALTIME_COST_OPTIMIZATION', True):
            return True
        
        # If REALTIME_FOR_TOOLS_ONLY is False, use Realtime API for everything
        if not getattr(config, 'REALTIME_FOR_TOOLS_ONLY', False):
            return True
        
        # Only use Realtime for tool interactions when REALTIME_FOR_TOOLS_ONLY is True
        tool_keywords = [
            'calendar', 'schedule', 'events', 'today', 'tomorrow',
            'lights', 'light', 'turn on', 'turn off', 'dim',
            'music', 'play', 'pause', 'spotify', 'song',
            'scene', 'mood', 'party', 'movie'
        ]
        
        transcript_lower = transcript.lower()
        needs_tools = any(keyword in transcript_lower for keyword in tool_keywords)
        
        if not needs_tools:
            # Check word count threshold as fallback
            threshold = getattr(config, 'REALTIME_SIMPLE_QUERY_THRESHOLD', 10)
            word_count = len(transcript.split())
            
            if word_count < threshold:
                return False  # Use traditional API for short, non-tool queries
        
        return True  # Use Realtime API

    def _process_with_traditional_api(self, message: str) -> None:
        """Process query using traditional API with 4o-mini for cost savings."""
        try:
            # Initialize traditional speech services if needed
            if not hasattr(self, '_traditional_speech_services'):
                from core.speech_services import SpeechServices as TraditionalSpeechServices
                
                openai_api_key = os.getenv("OPENAI_KEY")
                self._traditional_speech_services = TraditionalSpeechServices(
                    openai_api_key=openai_api_key,
                    whisper_model=config.WHISPER_MODEL,
                    chat_provider="openai",
                    chat_model=config.OPENAI_CHAT_MODEL,  # Uses 4o-mini
                    tts_enabled=config.TTS_ENABLED,
                    tts_model=config.TEXT_TO_SPEECH_MODEL,
                    tts_voice=config.TTS_VOICE
                )
            
            # Get response using traditional API
            response_context = self.conversation.get_response_context()
            print(f"💰 Using 4o-mini (context: {len(response_context)} messages)")
            
            response = self._traditional_speech_services.chat_completion(
                response_context,
                temperature=config.RESPONSE_TEMPERATURE,
                functions=self.functions  # Include tools
            )
            
            if response and response.get("content"):
                self.conversation.add_assistant_message(response["content"])
                # Trigger context management after adding assistant message
                self.speech_services._manage_context_updates(self.conversation)
                
                
                print(f"🤖  GPT: {response['content']}\n")
                
                # Convert response to speech if TTS is enabled
                if self._traditional_speech_services.tts_enabled:
                    audio_file = self._traditional_speech_services.text_to_speech(response["content"])
                
                print("─" * 50)
            else:
                print("⚠️ No response from traditional API")
                
        except Exception as e:
            print(f"❌ Error with traditional API: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()

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
            # Removed per-message latency logging to reduce spam
        
        # Calculate audio duration for this turn (no logging)
        if self.audio_session_start_time:
            audio_duration = current_time - self.audio_session_start_time
            self.audio_duration_per_turn.append(audio_duration)
            # Reset for next turn
            self.audio_session_start_time = current_time
        
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
        
        # Check if this message comes after an interruption
        message_to_process = transcript
        if (self.partial_response_content and self.interruption_timestamp and 
            getattr(config, 'CONVERSATION_CONTEXT_PRESERVATION', True)):
            
            # Analyze the interruption intent
            intent_analysis = self._analyze_interruption_intent(transcript)
            
            if config.DEBUG_MODE:
                print(f"🔍 Interruption analysis: {intent_analysis['type']} (confidence: {intent_analysis['confidence']:.1f})")
                if intent_analysis['pattern_matched']:
                    print(f"   Pattern matched: '{intent_analysis['pattern_matched']}'")
            
            # Handle different interruption types
            if intent_analysis['type'] == 'continuation' and intent_analysis['confidence'] > 0.7:
                # Get the last user message to combine with new request
                last_user_msg = None
                for msg in reversed(self.conversation.messages):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break
                
                if last_user_msg:
                    message_to_process = f"{last_user_msg} and also {transcript}"
                    print("🔗 Combining with previous request")
                    # Remove the interrupted response from conversation
                    self.conversation.remove_incomplete_response()
                
            elif intent_analysis['type'] == 'refinement' and intent_analysis['confidence'] > 0.8:
                print("✏️ Clarifying previous request")
                # Remove the interrupted response and replace with clarified version
                self.conversation.remove_incomplete_response()
                
            elif intent_analysis['type'] == 'new_topic':
                print("🔄 Switching to new topic")
                # Keep conversation history but remove interrupted response
                self.conversation.remove_incomplete_response()
            
            # Clear the partial response since we've handled it
            self.partial_response_content = ""
            self.interruption_timestamp = None
        elif self.partial_response_content and self.interruption_timestamp:
            # Context preservation disabled - just clear interrupted response
            if config.DEBUG_MODE:
                print("🗑️ Clearing interrupted response (context preservation disabled)")
            self.conversation.remove_incomplete_response()
            self.partial_response_content = ""
            self.interruption_timestamp = None
        
        # Add user message to conversation
        enhanced_message = self._enhance_message_for_freshness(message_to_process)
        self.conversation.add_user_message(enhanced_message)
        
        
        # Smart API selection for cost optimization
        use_realtime = self._should_use_realtime_for_query(transcript)
        
        if use_realtime:
            # Use Realtime API
            self._using_realtime_api = True
            if config.DEBUG_MODE:
                print("🤖  Waiting for response... (Realtime API)")
            
            # Only trigger response if not already waiting for one
            if not self.awaiting_response:
                self.awaiting_response = True
                self._response_start_time = time.time()  # Track when we start waiting for response
                
                # Temporary debug flag (bypassing config)
                TEMP_DEBUG = True
                
                if TEMP_DEBUG:
                    print(f"🤖  Triggering response with {len(self.functions) if self.functions else 0} tools")
                
                # Trigger response generation with tools
                if self.functions:
                    self._trigger_response_with_tools()
                else:
                    # Trigger response without tools
                    self.speech_services.trigger_response()
            else:
                print("⚠️  Already waiting for response, skipping duplicate trigger")
        else:
            # Fall back to traditional API with 4o-mini
            self._using_realtime_api = False
            if config.DEBUG_MODE:
                print("🤖  Processing with traditional API (4o-mini)...")
            
            # Process using traditional API for cost savings
            self._process_with_traditional_api(enhanced_message)
            return
        
        # Reset for next utterance
        self.current_transcript = ""
        self.frame_timestamps = []
        self.speech_end_time = None
        self.speech_detected = False
        
        # Reset interruption state for next interaction
        self.response_interrupted = False
        self.partial_response_content = ""
        self.interruption_timestamp = None
    
    def _on_speech_started(self) -> None:
        """Callback when speech starts - handle interruptions."""
        current_time = time.time()
        
        # Check if interruption detection is enabled
        if not getattr(config, 'INTERRUPTION_DETECTION_ENABLED', True):
            # Track that speech has started but don't handle interruptions
            self.speech_detected = True
            if config.DEBUG_MODE:
                print("🎤 Speech started (interruption detection disabled)")
            return
        
        # Check if we're currently awaiting a response (AI is speaking/generating)
        if self.awaiting_response:
            # Apply grace period to avoid false positives
            grace_period = getattr(config, 'INTERRUPTION_GRACE_PERIOD_MS', 100) / 1000.0
            if hasattr(self, '_response_start_time') and (current_time - self._response_start_time) < grace_period:
                if config.DEBUG_MODE:
                    print(f"🔇 Ignoring interruption within grace period ({grace_period:.1f}s)")
                return
            
            if config.DEBUG_MODE:
                print("🛑 User interruption detected - cancelling response")
            
            # Mark as interrupted and store timestamp
            self.response_interrupted = True
            self.interruption_timestamp = current_time
            
            # Cancel the active response
            try:
                self.speech_services.cancel_active_response()
                if config.DEBUG_MODE:
                    print("✅ Response cancelled successfully")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"⚠️ Error cancelling response: {e}")
            
            # Provide acknowledgment if enabled
            if getattr(config, 'INTERRUPTION_ACKNOWLEDGMENT_ENABLED', True):
                print("🎤 Voice detected, listening...")
            
            # Reset awaiting response flag since we cancelled
            self.awaiting_response = False
        
        # Track that speech has started
        self.speech_detected = True
        if config.DEBUG_MODE:
            print("🎤 Speech started")
    
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
        
        # Calculate final latency (no logging)
        final_latency = time.time() - final_latency_start
        self.session_final_latencies.append(final_latency)
        
        if response and response.get("content"):
            self.conversation.add_assistant_message(response["content"])
            # Trigger context management after adding assistant message
            self.speech_services._manage_context_updates(self.conversation)
            print(f"🤖  GPT: {response['content']}\n")
            
            # Convert response to speech if TTS is enabled
            if self.speech_services.tts_enabled:
                audio_file = self.speech_services.text_to_speech(response["content"])
            
            print("─" * 50)
    
    def _analyze_interruption_intent(self, new_message: str) -> Dict[str, Any]:
        """Analyze the intent behind an interruption to understand user's goal."""
        message_lower = new_message.lower().strip()
        
        # Define patterns for different interruption types
        continuation_patterns = [
            "and also", "plus", "also", "and", "additionally", "furthermore", "moreover",
            "oh and", "and another thing", "one more thing", "by the way"
        ]
        
        refinement_patterns = [
            "i meant", "actually i meant", "to clarify", "what i meant was", "let me clarify",
            "i should have said", "correction", "sorry i meant", "no wait", "wait i meant"
        ]
        
        new_topic_patterns = [
            "actually", "never mind", "forget that", "instead", "on second thought",
            "change of plans", "different question", "new question", "something else",
            "hold on", "wait", "stop"
        ]
        
        # Check for continuation (adding to previous request)
        for pattern in continuation_patterns:
            if pattern in message_lower:
                return {
                    "type": "continuation",
                    "confidence": 0.8,
                    "pattern_matched": pattern,
                    "suggested_response": f"Got it, so you want your original request and also: {new_message}"
                }
        
        # Check for refinement (clarifying previous request)
        for pattern in refinement_patterns:
            if pattern in message_lower:
                return {
                    "type": "refinement", 
                    "confidence": 0.9,
                    "pattern_matched": pattern,
                    "suggested_response": f"Thanks for clarifying. Let me help with: {new_message}"
                }
        
        # Check for new topic (completely different request)
        for pattern in new_topic_patterns:
            if pattern in message_lower:
                return {
                    "type": "new_topic",
                    "confidence": 0.8,
                    "pattern_matched": pattern,
                    "suggested_response": f"Understood, switching to: {new_message}"
                }
        
        # Default: assume it's a new topic if no patterns match
        return {
            "type": "new_topic",
            "confidence": 0.5,
            "pattern_matched": None,
            "suggested_response": f"Let me help with: {new_message}"
        }
    
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
            
            if config.DEBUG_MODE:
                print("✅ Connected to OpenAI Realtime API")
            
            # Configure tools in the session if available
            if self.functions:
                if config.DEBUG_MODE:
                    print(f"🔧 Configuring {len(self.functions)} tools in session...")
                self.speech_services.update_session_tools(self.functions)
            
            self.is_streaming = True
            # Start tracking audio duration for this conversation
            self.audio_session_start_time = time.time()
            
            # Set up callbacks for realtime events
            self.speech_services.set_callbacks(
                partial_transcript_callback=self.handle_partial_transcript if getattr(config, 'REALTIME_STREAM_TRANSCRIPTION', False) else None,
                speech_stopped_callback=self._on_speech_stopped,
                speech_started_callback=self._on_speech_started
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
                            function_call = self.speech_services.check_for_function_calls(timeout=0.05)  # Faster polling
                            if function_call and self.mcp_server:
                                print(f"\n🔧 TOOL INVOKED: {function_call.get('name')}")
                                result = self.speech_services.execute_function_call_realtime(function_call, self.mcp_server)
                                if result:
                                    print(f"✅ Tool executed successfully")
                                # Don't reset awaiting_response yet - wait for the final text response
                        except queue.Empty:
                            pass
                    
                    # Check for responses - only process if we're actually waiting for one
                    if self.awaiting_response:
                        try:
                            response = self.speech_services.response_queue.get(timeout=0.05)  # Faster polling
                            if response and response.get("content"):
                                # Check if this response was interrupted
                                if self.response_interrupted:
                                    # Store partial content but don't process fully
                                    self.partial_response_content = response["content"]
                                    if config.DEBUG_MODE:
                                        print(f"🛑 Interrupted response stored: {response['content'][:50]}...")
                                    
                                    # Mark the response as interrupted in conversation history
                                    # First add it to conversation, then mark as interrupted
                                    self.conversation.add_assistant_message(response["content"])
                                    # Trigger context management after adding assistant message
                                    self.speech_services._manage_context_updates(self.conversation)
                                    self.conversation.mark_response_as_interrupted()
                                    
                                    # Reset interruption flag for next response
                                    self.response_interrupted = False
                                else:
                                    # Process the complete response normally
                                    self.conversation.add_assistant_message(response["content"])
                                    # Trigger context management after adding assistant message
                                    self.speech_services._manage_context_updates(self.conversation)
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
    
    def _run_audio_loop(self):
        """Optimized audio loop with minimal startup overhead."""
        try:
            # Use shared audio stream (skip creation overhead if already exists)
            from core.components import SharedAudioManager
            if not self._audio_manager:
                self._audio_manager = SharedAudioManager()
            
            # Quick audio setup
            if not self._prewarmed:
                shared_stream_created = self._audio_manager.create_shared_stream(
                    samplerate=config.SAMPLE_RATE,
                    blocksize=config.FRAME_SIZE,
                    channels=1,
                    dtype=np.int16
                )
                if shared_stream_created:
                    self._audio_manager.subscribe_to_stream("RealtimeStreamingChatbot", self._shared_audio_callback)
                    self._prewarmed = True
            
            # Start main audio processing loop immediately
            while not self.session_ended:
                try:
                    audio_bytes, timestamp = self.audio_queue.get(timeout=0.1)
                    self.process_audio_frame(audio_bytes, timestamp)
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n✋ Finished. Bye!")
        finally:
            self.stop_realtime_session()
            if self._audio_manager:
                self._audio_manager.unsubscribe_from_stream("RealtimeStreamingChatbot")
    
    def run(self, mask_duration=0):
        """Main run loop for realtime streaming chatbot."""
        print("─" * 50)
        
        # Store mask duration for audio processing
        self._audio_mask_until = time.time() + mask_duration if mask_duration > 0 else 0
        if mask_duration > 0 and config.DEBUG_MODE:
            print(f"🔇 Masking audio input for {mask_duration:.1f}s")
        
        # Start realtime session (skip if already connected from pre-warming)
        if not self.speech_services.is_connected:
            if not self.start_realtime_session():
                print("❌ Failed to start realtime session, falling back to chunk-based mode")
                return
        else:
            # Connection already pre-warmed, just start streaming
            if config.DEBUG_MODE:
                print("✅ Using pre-warmed realtime connection")
            self.is_streaming = True
            
            # Set up callbacks
            self.speech_services.set_callbacks(
                partial_transcript_callback=self.handle_partial_transcript if getattr(config, 'REALTIME_STREAM_TRANSCRIPTION', False) else None,
                speech_stopped_callback=self._on_speech_stopped,
                speech_started_callback=self._on_speech_started
            )
            
            # Configure tools if available
            if self.functions:
                if config.DEBUG_MODE:
                    print(f"🔧 Configuring {len(self.functions)} tools in session...")
                self.speech_services.update_session_tools(self.functions)
            
            # Start message handling thread
            self._streaming_thread = threading.Thread(target=self._handle_realtime_messages, daemon=True)
            self._streaming_thread.start()
        
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