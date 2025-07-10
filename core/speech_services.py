"""
Speech services for transcription and chat completion.
Handles Whisper STT and ChatGPT interactions.
"""

import io
import os
import subprocess
import tempfile
from typing import List, Dict, Optional
from openai import OpenAI
from .model_providers import ModelProvider, create_provider


class SpeechServices:
    """Handles speech-to-text and chat completion services."""
    
    def __init__(self, 
                 openai_api_key: str, 
                 whisper_model: str, 
                 chat_provider: str,
                 chat_model: str,
                 gemini_api_key: Optional[str] = None,
                 tts_enabled: bool = False,
                 tts_model: str = "tts-1",
                 tts_voice: str = "alloy"):
        """
        Initialize speech services.
        
        Args:
            openai_api_key: OpenAI API key (required for Whisper and TTS)
            whisper_model: Whisper model name (e.g., "whisper-1")
            chat_provider: "openai" or "gemini"
            chat_model: Chat model name for the selected provider
            gemini_api_key: Google API key (required if using Gemini)
            tts_enabled: Enable text-to-speech
            tts_model: TTS model name (e.g., "tts-1")
            tts_voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
        """
        # OpenAI client for Whisper and TTS (always needed)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.whisper_model = whisper_model
        
        # TTS configuration
        self.tts_enabled = tts_enabled
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        
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
        
    def transcribe(self, wav_io: io.BytesIO) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            wav_io: BytesIO buffer containing WAV audio
            
        Returns:
            Transcribed text or empty string on error
        """
        try:
            # Check if audio chunk is too small (likely just noise)
            if len(wav_io.getvalue()) < 1000:  # Less than 1KB
                print("⚠️  Audio chunk too small, skipping transcription")
                return ""
            
            # Give the file a proper name for the API
            wav_io.name = "audio.wav"
            
            print("🔄 Sending to Whisper API...")
            transcript = self.openai_client.audio.transcriptions.create(
                model=self.whisper_model,
                file=wav_io,
                prompt="This is a conversation. Listen for natural speech.",
                timeout=10.0  # 10 second timeout
            )
            
            result = transcript.text.strip()
            if not result:
                print("⚠️  Whisper returned empty transcription")
                return ""
                
            return result
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            print("🔄 This might be a temporary network issue, continuing...")
            return ""
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, any]]] = None) -> Optional[Dict[str, any]]:
        """
        Get chat completion from the configured provider.
        
        Args:
            messages: Conversation history
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
            functions: Optional list of available functions for function calling
            
        Returns:
            Dict with 'content' and optionally 'function_call' keys, or error message
        """
        print(f"🤖 Using {self.chat_provider.upper()} for chat completion...")
        return self.model_provider.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions
        )
    
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
            print(f"🔊 Converting text to speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate speech using OpenAI TTS
            response = self.openai_client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                response_format="mp3"
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                response.stream_to_file(temp_file.name)
                audio_file_path = temp_file.name
            
            print(f"🎵 Generated speech audio: {audio_file_path}")
            
            # Play immediately if requested
            if play_immediately:
                self.play_audio_file(audio_file_path)
            
            return audio_file_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
    
    def play_audio_file(self, file_path: str) -> None:
        """
        Play an audio file using system audio player.
        
        Args:
            file_path: Path to the audio file to play
        """
        try:
            print(f"🔊 Playing audio: {file_path}")
            # Use afplay on macOS, could be extended for other platforms
            subprocess.run(['afplay', file_path], check=True)
            
            # Clean up temporary file after playing
            if file_path.startswith('/tmp') or '/tmp/' in file_path:
                try:
                    os.unlink(file_path)
                    print(f"🗑️ Cleaned up temporary audio file")
                except OSError:
                    pass
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ Audio playback error: {e}")
        except Exception as e:
            print(f"❌ Unexpected audio error: {e}")


class ConversationManager:
    """Manages conversation history and message flow."""
    
    def __init__(self, system_prompt: str):
        """
        Initialize conversation manager.
        
        Args:
            system_prompt: Initial system prompt
        """
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
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
        return self.messages[1:]
    
    def get_tool_context(self, max_messages: int = 6) -> List[Dict[str, str]]:
        """
        Get focused context for tool selection - system prompt + recent messages.
        
        Args:
            max_messages: Maximum number of recent messages to include (default: 6)
            
        Returns:
            List of messages with system prompt + recent conversation
        """
        if len(self.messages) <= max_messages + 1:  # +1 for system prompt
            return self.messages
        
        # Always include system prompt + recent messages
        system_prompt = self.messages[0]
        recent_messages = self.messages[-(max_messages):]
        
        return [system_prompt] + recent_messages
    
    def get_response_context(self) -> List[Dict[str, str]]:
        """
        Get full context for response generation - complete conversation history.
        
        Returns:
            Complete conversation history including system prompt
        """
        return self.messages
    
    def get_recent_messages(self, count: int) -> List[Dict[str, str]]:
        """
        Get the last N messages from conversation.
        
        Args:
            count: Number of recent messages to get
            
        Returns:
            List of recent messages (excluding system prompt)
        """
        if len(self.messages) <= 1:  # Only system prompt
            return []
        
        # Get recent messages (excluding system prompt)
        user_messages = self.messages[1:]  # Skip system prompt
        return user_messages[-count:] if count < len(user_messages) else user_messages
        
    def clear(self, system_prompt: str) -> None:
        """Clear conversation and reset with system prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]