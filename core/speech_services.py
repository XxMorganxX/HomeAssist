"""
Speech services for transcription and chat completion.
Handles Whisper STT and ChatGPT interactions.
"""

import io
from typing import List, Dict, Optional
from openai import OpenAI


class SpeechServices:
    """Handles speech-to-text and chat completion services."""
    
    def __init__(self, api_key: str, whisper_model: str, chat_model: str):
        """
        Initialize speech services.
        
        Args:
            api_key: OpenAI API key
            whisper_model: Whisper model name (e.g., "whisper-1")
            chat_model: Chat model name (e.g., "gpt-3.5-turbo")
        """
        self.client = OpenAI(api_key=api_key)
        self.whisper_model = whisper_model
        self.chat_model = chat_model
        
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
            transcript = self.client.audio.transcriptions.create(
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
        Get chat completion from ChatGPT.
        
        Args:
            messages: Conversation history
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
            functions: Optional list of available functions for function calling
            
        Returns:
            Dict with 'content' and optionally 'function_call' keys, or error message
        """
        try:
            kwargs = {
                "model": self.chat_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add function calling if functions are provided
            if functions:
                kwargs["functions"] = functions
                kwargs["function_call"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs, timeout=15.0)
            message = response.choices[0].message
            
            result = {"content": message.content}
            if hasattr(message, 'function_call') and message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                }
            
            return result
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return {"content": "Sorry, I'm having trouble connecting to the chat service."}


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
        
    def clear(self, system_prompt: str) -> None:
        """Clear conversation and reset with system prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]