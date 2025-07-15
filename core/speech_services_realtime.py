"""
Real-time speech services using OpenAI's 4o real-time API.
Drop-in replacement for speech_services.py - just change your import to use this file.

Usage:
    # Instead of: from core.speech_services import SpeechServices
    # Use: from core.speech_services_realtime import SpeechServices
"""

import io
import os
import json
import asyncio
import threading
import time
import base64
import subprocess
import tempfile
from typing import List, Dict, Optional, Callable, Any
import queue
import logging
import config
from config import SYSTEM_PROMPT

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

# Debug flag will be set later after imports are configured
DEBUG_REALTIME = False

def debug_log(message: str, data: dict = None):
    """Helper function for debug logging."""
    if DEBUG_REALTIME:
        if data:
            logger.warning(f"[REALTIME DEBUG] {message}: {json.dumps(data, indent=2)}")
        else:
            logger.warning(f"[REALTIME DEBUG] {message}")

try:
    import websockets
    import ssl
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not available - install with: pip install websockets")


class SpeechServices:
    """Real-time speech services using OpenAI's 4o real-time API with fallback to traditional API."""
    
    def __init__(self, 
                 openai_api_key: str, 
                 whisper_model: str = "whisper-1",
                 chat_provider: str = "openai",
                 chat_model: str = "gpt-4o-realtime-preview",
                 gemini_api_key: Optional[str] = None,
                 tts_enabled: bool = False,
                 tts_model: str = "tts-1",
                 tts_voice: str = "alloy"):
        """
        Initialize speech services with real-time API support.
        
        Args:
            openai_api_key: OpenAI API key
            whisper_model: Whisper model (for fallback)
            chat_provider: "openai" or "gemini" 
            chat_model: Chat model name
            gemini_api_key: Google API key (if using Gemini)
            tts_enabled: Enable text-to-speech
            tts_model: TTS model name
            tts_voice: TTS voice
        """
        self.openai_api_key = openai_api_key
        self.whisper_model = whisper_model
        self.chat_model = chat_model
        self.tts_enabled = tts_enabled
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        
        # Determine if we can use real-time API
        self.use_realtime = (
            WEBSOCKETS_AVAILABLE and 
            chat_provider.lower() == "openai" and 
            "realtime" in chat_model.lower()
        )
        
        if self.use_realtime:
            self._init_realtime()
        else:
            self._init_traditional(openai_api_key, chat_provider, chat_model, gemini_api_key)
    
    def _init_realtime(self):
        """Initialize real-time API components."""
        import config
        
        # Set debug flag from config
        global DEBUG_REALTIME
        DEBUG_REALTIME = getattr(config, 'REALTIME_API_DEBUG', False)
        
        model = getattr(config, 'REALTIME_MODEL', self.chat_model)
        self.ws_url = f"wss://api.openai.com/v1/realtime?model={model}"
        self.websocket = None
        self.is_connected = False
        self.session_id = None
        
        debug_log(f"Initializing real-time API with model: {model}")
        debug_log(f"WebSocket URL: {self.ws_url}")
        
        # Threading for real-time processing
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.function_call_queue = queue.Queue()
        self.loop = None
        self.loop_thread = None
        self._stop_event = threading.Event()
        
        # Connection management
        self._connection_lock = threading.Lock()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        
        # Callback functions for real-time events
        self.partial_transcript_callback = None
        self.response_delta_callback = None
        self.audio_response_callback = None
        self.speech_stopped_callback = None
        
        # Streaming state
        self.is_streaming = False
        self.current_input_item_id = None
        
        # Conversation item tracking
        self.conversation_items = []  # List of {id, type, role} for tracking items
        self.active_response_id = None  # Track if we have an active response
        self.last_user_item_id = None  # Track the last user input item

        # Audio configuration
        import config
        self.sample_rate = getattr(config, 'SAMPLE_RATE', 16000)

        logger.info("Real-time API mode enabled")
        debug_log("Real-time API components initialized")
    
    def _init_traditional(self, openai_api_key, chat_provider, chat_model, gemini_api_key):
        """Initialize traditional API components as fallback."""
        from openai import OpenAI
        from .model_providers import create_provider
        
        debug_log(f"Initializing traditional API with provider: {chat_provider}, model: {chat_model}")
        
        # OpenAI client for Whisper and TTS
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Set up chat provider
        self.chat_provider = chat_provider.lower()
        if self.chat_provider == "openai":
            self.model_provider = create_provider(
                "openai", 
                api_key=openai_api_key, 
                model=chat_model
            )
        elif self.chat_provider == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key required when using Gemini provider")
            self.model_provider = create_provider(
                "gemini", 
                api_key=gemini_api_key, 
                model=chat_model
            )
        else:
            raise ValueError(f"Unsupported chat provider: {chat_provider}")
        
        logger.info("Traditional API mode (fallback)")
        debug_log("Traditional API components initialized")
    
    def _ensure_connected(self) -> bool:
        """Ensure connection to real-time API."""
        if not self.use_realtime:
            debug_log("Not using real-time API, skipping connection check")
            return True
            
        with self._connection_lock:
            if self.is_connected:
                debug_log("Already connected to real-time API")
                return True
            
            debug_log("Attempting to connect to real-time API")
            return self._connect_realtime()
    
    def _connect_realtime(self) -> bool:
        """Connect to real-time API."""
        try:
            debug_log("Starting real-time connection process")
            
            if not self.loop_thread or not self.loop_thread.is_alive():
                debug_log("Creating new event loop thread")
                self.loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
                self.loop_thread.start()
                time.sleep(0.1)
            
            debug_log("Running async connection in event loop")
            future = asyncio.run_coroutine_threadsafe(self._async_connect(), self.loop)
            result = future.result(timeout=10.0)
            debug_log(f"Connection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Real-time connection failed: {e}")
            debug_log(f"Connection error details: {str(e)}")
            self._reconnect_attempts += 1
            
            if self._reconnect_attempts < self._max_reconnect_attempts:
                logger.info(f"Retrying connection ({self._reconnect_attempts}/{self._max_reconnect_attempts})")
                debug_log(f"Retry attempt {self._reconnect_attempts}")
                time.sleep(2.0)
                return self._connect_realtime()
            
            debug_log("Max reconnection attempts reached")
            return False
    
    def _start_event_loop(self):
        """Start asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def _async_connect(self) -> bool:
        """Async WebSocket connection."""
        try:
            debug_log("Starting async WebSocket connection")
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            debug_log("WebSocket headers prepared", headers)
            
            ssl_context = ssl.create_default_context()
            
            debug_log(f"Connecting to WebSocket URL: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ssl=ssl_context
            )
            
            self.is_connected = True
            logger.info("Connected to real-time API")
            debug_log("WebSocket connection established successfully")
            
            asyncio.create_task(self._listen_for_messages())
            await self._configure_session()
            
            debug_log("Async connection setup completed")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            debug_log(f"WebSocket connection error: {str(e)}")
            self.is_connected = False
            return False
    
    async def _configure_session(self, tools=None):
        """Configure real-time session."""
        import config
        
        debug_log("Configuring real-time session")
        
        # Debug: Print first 200 chars of system prompt to verify it's correct
        if DEBUG_REALTIME:
            logger.warning(f"[REALTIME DEBUG] System prompt preview: {SYSTEM_PROMPT[:200]}...")
        
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": SYSTEM_PROMPT,
                "voice": getattr(config, 'REALTIME_VOICE', self.tts_voice),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": getattr(config, 'REALTIME_VAD_THRESHOLD', 0.5),
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": getattr(config, 'REALTIME_VAD_SILENCE_MS', 500)
                },
                "temperature": max(0.6, getattr(config, 'RESPONSE_TEMPERATURE', 0.7)),
                "max_response_output_tokens": getattr(config, 'REALTIME_MAX_RESPONSE_TOKENS', 150)
            }
        }
        
        # Add tools to session if provided
        if tools:
            session_config["session"]["tools"] = self._convert_functions_to_tools(tools)
            debug_log(f"Adding {len(tools)} tools to session")
        
        debug_log("Session configuration payload", session_config)
        
        await self.websocket.send(json.dumps(session_config))
        logger.info("Real-time session configured")
        debug_log("Session configuration sent successfully")
    
    def update_session_tools(self, functions: list):
        """Update session with tools asynchronously."""
        if not self.use_realtime or not self.is_connected:
            return
            
        try:
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._configure_session(functions),
                    self.loop
                )
                debug_log(f"Updated session with {len(functions)} tools")
        except Exception as e:
            logger.error(f"Error updating session tools: {e}")
    
    async def _listen_for_messages(self):
        """Listen for WebSocket messages."""
        try:
            debug_log("Starting message listener")
            async for message in self.websocket:
                debug_log(f"Received WebSocket message: {message[:200]}...")
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            debug_log("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error listening: {e}")
            debug_log(f"Message listener error: {str(e)}")
            self.is_connected = False
    
    async def _handle_message(self, data: dict):
        """Handle WebSocket messages."""
        msg_type = data.get("type")
        debug_log(f"Handling message type: {msg_type}", data)
        
        if msg_type == "session.created":
            self.session_id = data.get("session", {}).get("id")
            logger.info(f"Session created: {self.session_id}")
            debug_log(f"Session created with ID: {self.session_id}")
            
        elif msg_type == "session.updated":
            logger.info("Session configuration updated")
            debug_log("Session configuration updated successfully")
            
        elif msg_type == "input_audio_buffer.speech_started":
            logger.debug("Speech started")
            cb = getattr(self, 'speech_started_callback', None)
            if callable(cb):
                cb()
            debug_log("Speech activity started")
            
        elif msg_type == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped")
            cb = getattr(self, 'speech_stopped_callback', None)
            if callable(cb):
                cb()
            debug_log("Speech activity stopped")
            
        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Partial transcription
            delta = data.get("delta", "")
            item_id = data.get("item_id")
            content_index = data.get("content_index", 0)
            cb = getattr(self, 'partial_transcript_callback', None)
            if delta and callable(cb):
                cb(delta)
            elif delta and cb is None:
                logger.debug("partial_transcript_callback is None")
            logger.debug(f"Transcription delta for item {item_id}, content_index {content_index}: {delta}")
            debug_log(f"Transcription delta: {delta}")
                
        elif msg_type == "conversation.item.created":
            # Track when conversation items are created
            item = data.get("item", {})
            item_id = item.get("id")
            if item_id:
                self.conversation_items.append({
                    "id": item_id,
                    "type": item.get("type"),
                    "role": item.get("role")
                })
                if item.get("role") == "user":
                    self.last_user_item_id = item_id
                logger.debug(f"Created conversation item: {item_id} (role: {item.get('role')})")
            debug_log(f"Conversation item created: {item_id}")
                
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Final transcription
            transcript = data.get("transcript", "")
            item_id = data.get("item_id")
            content_index = data.get("content_index", 0)
            if transcript:
                # Include the item_id and content_index with the transcript
                self.transcription_queue.put({
                    "text": transcript, 
                    "item_id": item_id,
                    "content_index": content_index
                })
                logger.debug(f"Transcription completed for item {item_id}, content_index {content_index}: {transcript}")
            debug_log(f"Transcription completed: {transcript}")
                
        elif msg_type == "response.created":
            # Track active response
            response = data.get("response", {})
            response_id = response.get("id")
            if response_id:
                self.active_response_id = response_id
                logger.debug(f"Response generation started: {response_id}")
            debug_log(f"Response created: {response_id}")
            
        elif msg_type == "response.text.delta":
            # Partial response text
            delta = data.get("delta", "")
            item_id = data.get("item_id")
            content_index = data.get("content_index", 0)
            cb = getattr(self, 'response_delta_callback', None)
            if delta and callable(cb):
                cb(delta)
            elif delta and cb is None:
                logger.debug("response_delta_callback is None")
            logger.debug(f"Response text delta for item {item_id}, content_index {content_index}: {delta}")
            debug_log(f"Response text delta: {delta}")
                
        elif msg_type == "response.text.done":
            text = data.get("text", "")
            item_id = data.get("item_id")
            content_index = data.get("content_index", 0)
            if text:
                self.response_queue.put({
                    "content": text, 
                    "role": "assistant",
                    "item_id": item_id,
                    "content_index": content_index
                })
                logger.debug(f"Response completed for item {item_id}, content_index {content_index}: {text}")
            debug_log(f"Response text completed: {text}")
                
        elif msg_type == "response.function_call_arguments.delta":
            # Function call arguments being built
            logger.debug("Function call arguments delta received")
            debug_log("Function call arguments delta received")
            
        elif msg_type == "response.function_call_arguments.done":
            # Function call ready to execute
            function_call = data.get("call", {})
            if function_call:
                self.function_call_queue.put(function_call)
                logger.debug(f"Function call ready: {function_call}")
            debug_log(f"Function call ready: {function_call}")
                
        elif msg_type == "response.audio.delta":
            # Audio response chunk
            audio_b64 = data.get("delta", "")
            cb = getattr(self, 'audio_response_callback', None)
            if audio_b64 and callable(cb):
                cb(audio_b64)
            elif audio_b64 and cb is None:
                logger.debug("audio_response_callback is None")
            debug_log("Audio response delta received")
                
        elif msg_type == "response.audio.done":
            logger.debug("Audio response completed")
            debug_log("Audio response completed")
            
        elif msg_type == "response.done":
            response_status = data.get("response", {}).get("status")
            response_id = data.get("response", {}).get("id")
            
            # Clear active response
            if response_id == self.active_response_id:
                self.active_response_id = None
                
            if response_status == "completed":
                logger.debug(f"Response {response_id} completed successfully")
            elif response_status == "incomplete":
                logger.warning(f"Response {response_id} incomplete - may have been interrupted")
            elif response_status == "failed":
                logger.error(f"Response {response_id} failed")
            debug_log(f"Response done: {response_id} - {response_status}")
                
        elif msg_type == "response.output_item.added":
            # New output item added to response
            item = data.get("item", {})
            logger.debug(f"Output item added: {item.get('type', 'unknown')}")
            debug_log(f"Output item added: {item.get('type', 'unknown')}")
            
        elif msg_type == "response.output_item.done":
            # Output item completed
            item = data.get("item", {})
            logger.debug(f"Output item done: {item.get('type', 'unknown')}")
            debug_log(f"Output item done: {item.get('type', 'unknown')}")
            
        elif msg_type == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error(f"API Error: {error_msg}")
            debug_log(f"API Error received: {error_msg}", data)
            
        else:
            logger.debug(f"Unhandled message type: {msg_type}")
            debug_log(f"Unhandled message type: {msg_type}", data)
    
    def transcribe(self, wav_io: io.BytesIO) -> str:
        """
        Transcribe audio - uses real-time API if available, otherwise traditional Whisper.
        
        Args:
            wav_io: BytesIO buffer containing WAV audio
            
        Returns:
            Transcribed text or empty string on error
        """
        debug_log(f"Transcribe called - use_realtime: {self.use_realtime}, is_connected: {self.is_connected}")
        
        if self.use_realtime:
            debug_log("Using real-time API for transcription")
            return self._transcribe_realtime(wav_io)
        else:
            debug_log("Using traditional API for transcription")
            return self._transcribe_traditional(wav_io)
    
    def _transcribe_realtime(self, wav_io: io.BytesIO) -> str:
        """Transcribe using real-time API."""
        try:
            debug_log("Starting real-time transcription")
            
            if not self._ensure_connected():
                debug_log("Real-time connection failed, falling back to traditional")
                # Fallback to traditional if real-time fails
                return self._transcribe_traditional(wav_io)
            
            audio_data = wav_io.getvalue()
            debug_log(f"Audio data size: {len(audio_data)} bytes")
            
            if len(audio_data) < 1000:
                debug_log("Audio data too small, skipping")
                return ""
            
            # Send audio to real-time API
            debug_log("Sending audio to real-time API")
            self._send_audio_data(audio_data)
            
            # Wait for transcription
            try:
                debug_log("Waiting for transcription response")
                result = self.transcription_queue.get(timeout=10.0)
                # Handle both old format (string) and new format (dict with text and item_id)
                if isinstance(result, dict):
                    transcript = result.get("text", "")
                    debug_log(f"Received transcript (dict format): {transcript}")
                else:
                    transcript = result
                    debug_log(f"Received transcript (string format): {transcript}")
                return transcript
            except queue.Empty:
                logger.warning("Real-time transcription timeout")
                debug_log("Real-time transcription timeout")
                return ""
                
        except Exception as e:
            logger.error(f"Real-time transcription error: {e}")
            debug_log(f"Real-time transcription error: {str(e)}")
            # Fallback to traditional
            return self._transcribe_traditional(wav_io)
    
    def _transcribe_traditional(self, wav_io: io.BytesIO) -> str:
        """Transcribe using traditional Whisper API."""
        try:
            from openai import OpenAI
            
            if not hasattr(self, 'openai_client'):
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            if len(wav_io.getvalue()) < 1000:
                return ""
            
            wav_io.name = "audio.wav"
            
            transcript = self.openai_client.audio.transcriptions.create(
                model=self.whisper_model,
                file=wav_io,
                prompt="This is a conversation. Listen for natural speech.",
                timeout=10.0
            )
            
            result = transcript.text.strip()
            return result if result else ""
            
        except Exception as e:
            logger.error(f"Traditional transcription error: {e}")
            return ""
    
    def _send_audio_data(self, audio_data: bytes):
        """Send audio to real-time API."""
        try:
            debug_log(f"Sending audio data: {len(audio_data)} bytes")
            
            if not self.is_connected or not self.websocket:
                debug_log("Not connected, skipping audio send")
                return
                
            # Convert raw PCM16 audio to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            debug_log(f"Audio encoded to base64: {len(audio_base64)} chars")
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            
            debug_log("Sending audio buffer append message")
            
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
                # Don't wait for completion to avoid blocking
                
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            debug_log(f"Audio send error: {str(e)}")
    
    def send_audio_frame(self, audio_frame: bytes):
        """Public method to send audio frame (alias for _send_audio_data)."""
        debug_log(f"send_audio_frame called: {len(audio_frame)} bytes")
        self._send_audio_data(audio_frame)
    
    def set_callbacks(self, partial_transcript_callback=None, response_delta_callback=None, 
                      audio_response_callback=None, speech_stopped_callback=None, speech_started_callback=None):
        """Set callback functions for real-time events."""
        debug_log("Setting callbacks")
        self.partial_transcript_callback = partial_transcript_callback
        self.response_delta_callback = response_delta_callback
        self.audio_response_callback = audio_response_callback
        self.speech_stopped_callback = speech_stopped_callback
        self.speech_started_callback = speech_started_callback
        debug_log("Callbacks set successfully")
    
    def start_streaming(self):
        """Start continuous audio streaming mode."""
        debug_log("start_streaming called")
        
        if not self.use_realtime or not self.is_connected:
            debug_log("Cannot start streaming - not using realtime or not connected")
            return False
            
        try:
            debug_log("Creating conversation item for streaming")
            # Create new conversation item for streaming
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "audio": ""  # Will be filled by streaming
                        # NOTE: content_index must not be included in client-sent events
                    }]
                }
            }
            
            debug_log("Sending conversation item create message")
            
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            


            self.is_streaming = True
            logger.info("Started continuous streaming mode")
            debug_log("Streaming started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            debug_log(f"Streaming start error: {str(e)}")
            return False
    
    def stop_streaming(self):
        """Stop continuous streaming and commit the audio buffer."""
        if not self.use_realtime or not self.is_connected or not self.is_streaming:
            return
            
        try:
            # Commit the audio buffer to trigger transcription
            message = {
                "type": "input_audio_buffer.commit"
            }
            
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            
            self.is_streaming = False
            logger.debug("Stopped streaming and committed audio buffer")
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
    
    def clear_audio_buffer(self):
        """Clear the input audio buffer without processing."""
        if not self.use_realtime or not self.is_connected:
            return
            
        try:
            message = {
                "type": "input_audio_buffer.clear"
            }
            
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            
            logger.debug("Cleared audio buffer")
            
        except Exception as e:
            logger.error(f"Error clearing audio buffer: {e}")
    
    def cancel_active_response(self):
        """Cancel any active response generation."""
        if not self.use_realtime or not self.is_connected or not self.active_response_id:
            return
            
        try:
            message = {
                "type": "response.cancel"
            }
            
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            
            logger.debug(f"Cancelled active response: {self.active_response_id}")
            self.active_response_id = None
            
        except Exception as e:
            logger.error(f"Error cancelling response: {e}")
    
    def trigger_response(self):
        """Trigger response generation from the assistant."""
        if not self.use_realtime or not self.is_connected:
            return
            
        try:
            # Ensure temperature is at least 0.6 for API compliance
            safe_temperature = max(0.6, 0.7)  # Default to 0.7 if not specified
            
            message = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "temperature": safe_temperature
                }
            }
            
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
            
            logger.debug("Triggered response generation")
            
        except Exception as e:
            logger.error(f"Error triggering response: {e}")
    
    def _manage_context_updates(self, conversation_manager) -> None:
        """
        Manage context updates and summary generation decisions.
        This is where all context strategy decisions are made.
        """
        from core.context_manager import ContextManager
        
        try:
            # Decision: Should we update the summary?
            if self._should_update_summary(conversation_manager):
                debug_log(f"Updating conversation summary")
                context_manager = ContextManager()
                context_manager.update_summary(conversation_manager)
                debug_log(f"Summary updated successfully")
        except Exception as e:
            debug_log(f"Error managing context updates: {e}")
    
    def _should_update_summary(self, conversation_manager) -> bool:
        """
        Decision logic: Should we update the conversation summary?
        """
        messages = conversation_manager.get_messages()
        min_messages = getattr(config, 'CONTEXT_SUMMARY_MIN_MESSAGES', 5)
        frequency = getattr(config, 'CONTEXT_SUMMARY_FREQUENCY', 5)
        
        # Only generate summaries if frequency is enabled (> 0)
        if frequency <= 0:
            return False
        
        # Check if conversation is long enough and at frequency interval
        # Trigger at min_messages threshold, then every frequency messages after that
        return (len(messages) >= min_messages and 
                (len(messages) == min_messages or (len(messages) - min_messages) % frequency == 0))

    def _sync_conversation_to_session(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """Sync conversation history to Realtime session with intelligent context management."""
        try:
            import config
            
            # Apply intelligent context management using new ContextManager interface
            if getattr(config, 'REALTIME_COST_OPTIMIZATION', True):
                # Create a temporary conversation manager for the context manager
                from core.context_manager import ContextManager
                
                class TempConversationManager:
                    def __init__(self, messages, system_prompt):
                        self.messages = messages
                        self.system_prompt = system_prompt
                    def get_messages(self):
                        return self.messages
                    def get_chat_minus_sys_prompt(self):
                        return self.messages
                
                temp_conv = TempConversationManager(messages, config.SYSTEM_PROMPT)
                context_manager = ContextManager()
                messages = context_manager.get_context_for_response(temp_conv, use_summary=True)
                debug_log(f"Using intelligent context: {len(messages)} messages")
            
            # Clear any existing conversation items first
            # According to OpenAI Realtime API, we need to truncate to a specific item
            # If we have conversation items, truncate to the system message (first item)
            if self.conversation_items:
                # Find the first item (should be system message or first conversation item)
                first_item_id = self.conversation_items[0]["id"]
                # Truncate to the first item – content_index should not be included per API docs
                clear_message = {"type": "conversation.item.truncate", "item_id": first_item_id}
                
                if self.loop:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket.send(json.dumps(clear_message)),
                            self.loop
                        )
                        debug_log(f"Conversation truncated to item: {first_item_id}")
                    except Exception as e:
                        logger.error(f"Error truncating conversation: {e}")
                        debug_log(f"Conversation truncate error: {str(e)}")
            
            # Add each message as a conversation item
            for i, message in enumerate(messages):
                if message.get("role") == "system":
                    # System message is already set in session configuration
                    continue
                    
                # Validate message content before creating item
                if not message.get("content"):
                    logger.warning(f"Empty message content for role {message.get('role')}")
                    continue
                
                item = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": message["role"],
                        "content": [
                            {
                                "type": "input_text",
                                "text": message["content"]
                            }
                        ]
                    }
                }
                
                if self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps(item)),
                        self.loop
                    )
                    
            logger.debug(f"Synced {len(messages)} messages to Realtime session")
            
        except Exception as e:
            logger.error(f"Error syncing conversation to session: {e}")
    
    def _convert_functions_to_tools(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI function format to Realtime API tools format with compression."""
        import config
        tools = []
        
        for func in functions:
            # Cost optimization: Compress tool descriptions
            if getattr(config, 'REALTIME_COST_OPTIMIZATION', True):
                # Shorten description to first sentence or 50 chars
                desc = func["description"]
                if '.' in desc:
                    desc = desc.split('.')[0] + '.'
                elif len(desc) > 50:
                    desc = desc[:50] + '...'
                
                tool = {
                    "type": "function",
                    "name": func["name"],
                    "description": desc,
                    "parameters": func["parameters"]
                }
            else:
                tool = {
                    "type": "function",
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"]
                }
            tools.append(tool)
        return tools
    
    def execute_function_call_realtime(self, function_call: Dict[str, Any], mcp_server) -> Optional[Dict[str, Any]]:
        """Execute a function call from Realtime API and return result to session."""
        try:
            func_name = function_call.get("name")
            func_args_str = function_call.get("arguments", "{}")
            call_id = function_call.get("call_id")
            
            if not func_name or not call_id:
                logger.error("Invalid function call format")
                return None
            
            # Parse arguments
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid function arguments JSON: {e}")
                return None
            
            # Execute the function using MCP server
            tool_result = mcp_server.execute_tool(func_name, func_args)
            
            # Send function result back to Realtime session
            function_result = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(tool_result)
                }
            }
            
            if self.loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps(function_result)),
                        self.loop
                    )
                    debug_log(f"Function result sent successfully: {call_id}")
                except Exception as e:
                    logger.error(f"Error sending function result: {e}")
                    debug_log(f"Function result send error: {str(e)}")
            
            logger.debug(f"Executed function {func_name} and sent result to session")
            return {"call_id": call_id, "result": tool_result}
            
        except Exception as e:
            logger.error(f"Error executing function call: {e}")
            return None
    
    def check_for_function_calls(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Check if there are pending function calls to execute."""
        try:
            function_call = self.function_call_queue.get(timeout=timeout)
            return function_call
        except queue.Empty:
            return None
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None,
                       item_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get chat completion - uses real-time API if available, otherwise traditional.
        
        Args:
            messages: Conversation history
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
            functions: Optional list of available functions
            item_id: Optional conversation item ID to respond to
            
        Returns:
            Dict with 'content' key, or None on error
        """
        debug_log(f"Chat completion called - use_realtime: {self.use_realtime}, is_connected: {self.is_connected}")
        debug_log(f"Parameters - max_tokens: {max_tokens}, temperature: {temperature}, functions: {len(functions) if functions else 0}")
        debug_log(f"Messages count: {len(messages)}")
        
        if self.use_realtime and self.is_connected:
            debug_log("Using real-time API for chat completion")
            # Use Realtime API for both regular chat and function calls
            return self._chat_completion_realtime(messages, max_tokens, temperature, functions, item_id)
        else:
            debug_log("Using traditional API for chat completion")
            # Fall back to traditional API only if Realtime is not connected
            return self._chat_completion_traditional(messages, max_tokens, temperature, functions)
    
    def _chat_completion_realtime(self, messages, max_tokens, temperature, functions=None, item_id=None) -> Optional[Dict[str, Any]]:
        """Chat completion using real-time API with proper conversation context."""
        try:
            debug_log("Starting real-time chat completion")
            
            if not self._ensure_connected():
                logger.warning("Realtime API not connected, falling back to traditional")
                debug_log("Real-time API not connected, falling back to traditional")
                return self._chat_completion_traditional(messages, max_tokens, temperature, functions)
            
            # Cancel any active response first
            if self.active_response_id:
                logger.warning(f"Cancelling active response {self.active_response_id} before creating new one")
                debug_log(f"Cancelling active response: {self.active_response_id}")
                self.cancel_active_response()
                time.sleep(0.1)  # Brief pause to ensure cancellation is processed
            
            # Add conversation context to Realtime session
            debug_log("Syncing conversation to session")
            self._sync_conversation_to_session(messages, functions)
            
            # Ensure temperature is at least 0.6 for API compliance
            safe_temperature = max(0.6, temperature)
            debug_log(f"Temperature adjusted from {temperature} to {safe_temperature}")

            # Create response with proper configuration
            
            response_config = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "voice": self.tts_voice,
                    "output_audio_format": "pcm16",
                    "temperature": safe_temperature,
                    "max_output_tokens": max_tokens
                }
            }
            
            # Note: item_id and content_index are NOT used in client response.create events
            # They are used in server events that the API sends back to us
            # The API will automatically respond to the conversation context
            
            # Add tools/functions if provided
            if functions:
                debug_log(f"Adding {len(functions)} functions to response")
                response_config["response"]["tools"] = self._convert_functions_to_tools(functions)
            
            # Debug: Log the payload being sent
            debug_log("Sending real-time API payload", response_config)
            
            # Send response generation request
            if self.loop:
                debug_log("Sending response.create via event loop")
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(response_config)),
                    self.loop
                )
            
            # Wait for response completion, handling function calls if they occur
            try:
                # Check for function calls first if functions were provided
                if functions:
                    debug_log("Checking for function calls")
                    function_call = self.check_for_function_calls(timeout=5.0)
                    if function_call:
                        debug_log(f"Function call detected: {function_call}")
                        # Return function call for external execution
                        return {
                            "function_call": {
                                "name": function_call.get("name"),
                                "arguments": function_call.get("arguments", "{}")
                            },
                            "call_id": function_call.get("call_id")
                        }
                
                # Wait for text response
                debug_log("Waiting for text response")
                response = self.response_queue.get(timeout=20.0)
                debug_log(f"Received response: {response}")
                return response
            except queue.Empty:
                logger.warning("Real-time chat timeout")
                debug_log("Real-time chat timeout")
                return None
                
        except Exception as e:
            logger.error(f"Real-time chat error: {e}")
            debug_log(f"Real-time chat error: {str(e)}")
            # Only fall back if there's a real error, not just timeout
            if "timeout" not in str(e).lower():
                return self._chat_completion_traditional(messages, max_tokens, temperature, functions)
            return None
    
    def _chat_completion_traditional(self, messages, max_tokens, temperature, functions=None):
        """Chat completion using traditional API."""
        try:
            debug_log("Starting traditional chat completion")
            
            if not hasattr(self, 'model_provider'):
                debug_log("Initializing traditional components")
                # Initialize traditional components if not done
                self._init_traditional(self.openai_api_key, "openai", "gpt-4o", None)
            
            debug_log(f"Calling traditional model provider with {len(messages)} messages")
            result = self.model_provider.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions
            )
            debug_log(f"Traditional chat completion result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Traditional chat error: {e}")
            debug_log(f"Traditional chat error: {str(e)}")
            return None
    
    def _trigger_response(self, max_tokens, temperature):
        """Trigger response in real-time API."""
        try:
            if not self.is_connected or not self.websocket:
                return
                
            # Ensure temperature is at least 0.6 for API compliance
            safe_temperature = max(0.6, temperature)
                
            message = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "temperature": safe_temperature,
                    "max_output_tokens": max_tokens
                }
            }
            
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.loop
                )
                
        except Exception as e:
            logger.error(f"Error triggering response: {e}")
    
    def text_to_speech(self, text: str, play_immediately: bool = True) -> Optional[str]:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert to speech
            play_immediately: Whether to play the audio immediately
            
        Returns:
            Path to the generated audio file, or None if TTS is disabled/failed
        """
        if not self.tts_enabled:
            return None
            
        try:
            from openai import OpenAI
            
            if not hasattr(self, 'openai_client'):
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            response = self.openai_client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                response_format="mp3"
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                response.stream_to_file(temp_file.name)
                audio_file_path = temp_file.name
            
            if play_immediately:
                self.play_audio_file(audio_file_path)
            
            return audio_file_path
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def play_audio_file(self, file_path: str) -> None:
        """Play an audio file using system audio player."""
        try:
            subprocess.run(['afplay', file_path], check=True)
            
            if file_path.startswith('/tmp') or '/tmp/' in file_path:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio playback error: {e}")
        except Exception as e:
            logger.error(f"Unexpected audio error: {e}")
    
    def disconnect(self):
        """Disconnect from real-time API."""
        if not self.use_realtime:
            return
            
        try:
            if self.websocket and self.is_connected:
                if self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.close(), self.loop
                    )
                self.is_connected = False
                
            self._stop_event.set()
            
            if self.loop_thread and self.loop_thread.is_alive():
                if self.loop:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                self.loop_thread.join(timeout=2.0)
                
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'use_realtime') and self.use_realtime:
            self.disconnect()


class ConversationManager:
    """
    Pure conversation oracle. Stores and manages conversation data only.
    
    Responsibilities:
    - Store conversation messages
    - Provide data access methods
    - Manage conversation metadata (interruptions, etc.)
    
    Does NOT:
    - Make context decisions
    - Trigger context updates
    - Implement context strategies
    """
    
    def __init__(self, system_prompt: str):
        """Initialize conversation manager."""
        # For Realtime API, system prompt is set at session level, not as message
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = system_prompt  # Store for reference but don't add to messages
        
    def add_user_message(self, content: str) -> None:
        """Add user message to conversation."""
        self.messages.append({"role": "user", "content": content})
        
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation."""
        self.messages.append({"role": "assistant", "content": content})
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages
    
    def get_chat_minus_sys_prompt(self) -> List[Dict[str, str]]:
        """Get conversation history minus the system prompt."""
        # For Realtime API, no system prompt in messages, so return all messages
        return self.messages
    
    
    def get_recent_messages(self, count: int) -> List[Dict[str, str]]:
        """Get the last N messages from conversation."""
        # For Realtime API, no system prompt in messages, so return recent messages directly
        if len(self.messages) == 0:
            return []
        
        return self.messages[-count:] if count < len(self.messages) else self.messages
        
    def clear(self, system_prompt: str) -> None:
        """Clear conversation and reset with system prompt."""
        # For Realtime API, system prompt is set at session level, not as message
        self.messages = []
        self.system_prompt = system_prompt
    
    def mark_response_as_interrupted(self, assistant_message_id: str = None) -> None:
        """Mark the last assistant response as interrupted."""
        # Find the last assistant message and mark it as interrupted
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                # Add metadata to indicate this response was interrupted
                if "metadata" not in self.messages[i]:
                    self.messages[i]["metadata"] = {}
                self.messages[i]["metadata"]["interrupted"] = True
                self.messages[i]["metadata"]["interrupted_at"] = time.time()
                break
    
    def remove_incomplete_response(self) -> Optional[str]:
        """Remove the last assistant message if it was incomplete/interrupted. Returns the removed content."""
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                # Check if this response was marked as interrupted
                metadata = self.messages[i].get("metadata", {})
                if metadata.get("interrupted", False):
                    removed_message = self.messages.pop(i)
                    return removed_message.get("content", "")
                break
        return None
    
    def get_conversation_state_before_interruption(self) -> List[Dict]:
        """Get conversation state before the last interruption."""
        # Return all messages except interrupted assistant responses
        clean_messages = []
        for message in self.messages:
            if message.get("role") == "assistant":
                metadata = message.get("metadata", {})
                if not metadata.get("interrupted", False):
                    clean_messages.append(message)
            else:
                clean_messages.append(message)
        return clean_messages
    
 