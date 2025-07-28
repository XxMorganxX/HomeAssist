"""
Model provider abstractions for different AI services.
Supports OpenAI only.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from openai import OpenAI


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



def create_provider(provider_type: str, **kwargs) -> ModelProvider:
    """
    Factory function to create model providers.
    
    Args:
        provider_type: Only "openai" supported
        **kwargs: Provider-specific arguments
        
    Returns:
        ModelProvider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIProvider(**kwargs)
    else:
        raise ValueError(f"Only OpenAI provider supported, got: {provider_type}")