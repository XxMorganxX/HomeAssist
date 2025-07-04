"""
Model provider abstractions for different AI services.
Supports OpenAI and Google Gemini with a common interface.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from openai import OpenAI
import google.generativeai as genai


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""
    
    @abstractmethod
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        Get chat completion from the AI model.
        
        Args:
            messages: Conversation history in OpenAI format
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
            functions: Optional list of available functions for function calling
            
        Returns:
            Dict with 'content' and optionally 'function_call' keys, or error message
        """
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for chat completions."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Get chat completion from OpenAI."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add function calling if functions are provided
            if functions:
                # Convert functions to tools format for newer API
                tools = []
                for func in functions:
                    tools.append({
                        "type": "function",
                        "function": func
                    })
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs, timeout=15.0)
            message = response.choices[0].message
            
            result = {"content": message.content}
            
            # Handle new tool_calls format (supports multiple calls)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                result["tool_calls"] = []
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            # Backward compatibility: still support old function_call format
            elif hasattr(message, 'function_call') and message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                }
            
            return result
        except Exception as e:
            print(f"❌ OpenAI Chat error: {e}")
            return {"content": "Sorry, I'm having trouble connecting to the OpenAI service."}


class GeminiProvider(ModelProvider):
    """Google Gemini API provider for chat completions."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key
            model: Model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI message format to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have a system role, prepend to first user message
                continue
            elif msg["role"] == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [msg["content"]]
                })
            elif msg["role"] == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })
                
        return gemini_messages
    
    def _extract_system_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract system prompt from messages."""
        for msg in messages:
            if msg["role"] == "system":
                return msg["content"]
        return ""
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 150,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Get chat completion from Gemini."""
        try:
            # Extract system prompt and convert messages
            system_prompt = self._extract_system_prompt(messages)
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Start chat with system instruction if available
            if system_prompt:
                chat = self.model.start_chat(
                    history=gemini_messages[:-1] if gemini_messages else [],
                    system_instruction=system_prompt
                )
            else:
                chat = self.model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
            
            # Get the latest user message
            if gemini_messages:
                latest_message = gemini_messages[-1]["parts"][0]
            else:
                latest_message = "Hello"
            
            # Generate response
            response = chat.send_message(
                latest_message,
                generation_config=generation_config
            )
            
            # Note: Gemini function calling is more complex and would require additional setup
            # For now, we'll focus on basic chat completion
            if functions:
                print("⚠️  Function calling not yet implemented for Gemini provider")
            
            return {"content": response.text}
            
        except Exception as e:
            print(f"❌ Gemini Chat error: {e}")
            return {"content": "Sorry, I'm having trouble connecting to the Gemini service."}


def create_provider(provider_type: str, **kwargs) -> ModelProvider:
    """
    Factory function to create model providers.
    
    Args:
        provider_type: "openai" or "gemini"
        **kwargs: Provider-specific arguments
        
    Returns:
        ModelProvider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type.lower() == "gemini":
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")