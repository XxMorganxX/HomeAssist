"""
Factory for creating provider instances based on configuration.
"""

from typing import Dict, Any
from pathlib import Path

try:
    # Try relative imports first (when used as package)
    from .interfaces import (
        TranscriptionInterface,
        ResponseInterface, 
        TextToSpeechInterface,
        ContextInterface,
        WakeWordInterface
    )
    from .providers.transcription_v2 import AssemblyAIAsyncProvider
    from .providers.response import OpenAIWebSocketResponseProvider
    from .providers.tts import GoogleTTSProvider, LocalTTSProvider
    from .providers.context import UnifiedContextProvider
    from .providers.wakeword_v2 import IsolatedOpenWakeWordProvider
except ImportError:
    # Fall back to absolute imports (when run as module)
    from assistant_framework.interfaces import (
        TranscriptionInterface,
        ResponseInterface, 
        TextToSpeechInterface,
        ContextInterface,
        WakeWordInterface
    )
    from assistant_framework.providers.transcription_v2 import AssemblyAIAsyncProvider
    from assistant_framework.providers.response import OpenAIWebSocketResponseProvider
    from assistant_framework.providers.tts import GoogleTTSProvider, LocalTTSProvider
    from assistant_framework.providers.context import UnifiedContextProvider
    from assistant_framework.providers.wakeword_v2 import IsolatedOpenWakeWordProvider


class ProviderFactory:
    """Factory for creating provider instances."""
    
    # Provider registries (v2 providers - segfault-safe)
    TRANSCRIPTION_PROVIDERS = {
        'assemblyai': AssemblyAIAsyncProvider,
    }
    
    RESPONSE_PROVIDERS = {
        'openai_websocket': OpenAIWebSocketResponseProvider,
    }
    
    TTS_PROVIDERS = {
        'google_tts': GoogleTTSProvider,
        'local_tts': LocalTTSProvider,
    }
    
    CONTEXT_PROVIDERS = {
        'unified': UnifiedContextProvider,
    }
    
    WAKEWORD_PROVIDERS = {
        'openwakeword': IsolatedOpenWakeWordProvider,
    }
    
    @classmethod
    def create_transcription_provider(cls, 
                                    provider_name: str, 
                                    config: Dict[str, Any]) -> TranscriptionInterface:
        """
        Create a transcription provider instance.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            TranscriptionInterface: Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        if provider_name not in cls.TRANSCRIPTION_PROVIDERS:
            available = ', '.join(cls.TRANSCRIPTION_PROVIDERS.keys())
            raise ValueError(f"Unsupported transcription provider: {provider_name}. Available: {available}")
        
        provider_class = cls.TRANSCRIPTION_PROVIDERS[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_response_provider(cls,
                               provider_name: str,
                               config: Dict[str, Any]) -> ResponseInterface:
        """
        Create a response provider instance.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            ResponseInterface: Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        if provider_name not in cls.RESPONSE_PROVIDERS:
            available = ', '.join(cls.RESPONSE_PROVIDERS.keys())
            raise ValueError(f"Unsupported response provider: {provider_name}. Available: {available}")
        
        provider_class = cls.RESPONSE_PROVIDERS[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_tts_provider(cls,
                           provider_name: str,
                           config: Dict[str, Any]) -> TextToSpeechInterface:
        """
        Create a TTS provider instance.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            TextToSpeechInterface: Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        if provider_name not in cls.TTS_PROVIDERS:
            available = ', '.join(cls.TTS_PROVIDERS.keys())
            raise ValueError(f"Unsupported TTS provider: {provider_name}. Available: {available}")
        
        provider_class = cls.TTS_PROVIDERS[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_context_provider(cls,
                               provider_name: str,
                               config: Dict[str, Any]) -> ContextInterface:
        """
        Create a context provider instance.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            ContextInterface: Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        if provider_name not in cls.CONTEXT_PROVIDERS:
            available = ', '.join(cls.CONTEXT_PROVIDERS.keys())
            raise ValueError(f"Unsupported context provider: {provider_name}. Available: {available}")
        
        provider_class = cls.CONTEXT_PROVIDERS[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_wakeword_provider(cls,
                                 provider_name: str,
                                 config: Dict[str, Any]) -> WakeWordInterface:
        """
        Create a wake word provider instance.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            WakeWordInterface: Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        if provider_name not in cls.WAKEWORD_PROVIDERS:
            available = ', '.join(cls.WAKEWORD_PROVIDERS.keys())
            raise ValueError(f"Unsupported wakeword provider: {provider_name}. Available: {available}")
        provider_class = cls.WAKEWORD_PROVIDERS[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_all_providers(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create all providers based on configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Dictionary containing all provider instances
        """
        providers = {}
        
        # Create each provider type
        if 'transcription' in config:
            transcription_config = config['transcription']
            provider_name = transcription_config.get('provider')
            provider_config = transcription_config.get('config', {})
            
            if provider_name:
                providers['transcription'] = cls.create_transcription_provider(
                    provider_name, provider_config
                )
        
        if 'response' in config:
            response_config = config['response']
            provider_name = response_config.get('provider')
            provider_config = response_config.get('config', {})
            
            if provider_name:
                providers['response'] = cls.create_response_provider(
                    provider_name, provider_config
                )
        
        if 'tts' in config:
            tts_config = config['tts']
            provider_name = tts_config.get('provider')
            provider_config = tts_config.get('config', {})
            
            if provider_name:
                providers['tts'] = cls.create_tts_provider(
                    provider_name, provider_config
                )
        
        if 'context' in config:
            context_config = config['context']
            provider_name = context_config.get('provider')
            provider_config = context_config.get('config', {})
            
            if provider_name:
                providers['context'] = cls.create_context_provider(
                    provider_name, provider_config
                )
        
        if 'wakeword' in config:
            wakeword_config = config['wakeword']
            provider_name = wakeword_config.get('provider')
            provider_config = wakeword_config.get('config', {})
            
            if provider_name:
                providers['wakeword'] = cls.create_wakeword_provider(
                    provider_name, provider_config
                )
        
        return providers
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, list]:
        """
        Get list of all available providers by type.
        
        Returns:
            Dictionary mapping provider types to available provider names
        """
        return {
            'transcription': list(cls.TRANSCRIPTION_PROVIDERS.keys()),
            'response': list(cls.RESPONSE_PROVIDERS.keys()),
            'tts': list(cls.TTS_PROVIDERS.keys()),
            'context': list(cls.CONTEXT_PROVIDERS.keys()),
            'wakeword': list(cls.WAKEWORD_PROVIDERS.keys()),
        }
    
    @classmethod
    def validate_provider_config(cls, provider_type: str, provider_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate a provider configuration.
        
        Args:
            provider_type: Type of provider ('transcription', 'response', 'tts', 'context')
            provider_name: Name of the provider
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If provider type/name is invalid or config is missing required fields
        """
        # Check if provider type is valid
        provider_registries = {
            'transcription': cls.TRANSCRIPTION_PROVIDERS,
            'response': cls.RESPONSE_PROVIDERS,
            'tts': cls.TTS_PROVIDERS,
            'context': cls.CONTEXT_PROVIDERS,
            'wakeword': cls.WAKEWORD_PROVIDERS,
        }
        
        if provider_type not in provider_registries:
            available_types = ', '.join(provider_registries.keys())
            raise ValueError(f"Invalid provider type: {provider_type}. Available: {available_types}")
        
        # Check if provider name is valid for the type
        registry = provider_registries[provider_type]
        if provider_name not in registry:
            available_names = ', '.join(registry.keys())
            raise ValueError(f"Invalid {provider_type} provider: {provider_name}. Available: {available_names}")
        
        # Basic validation - try to create the provider
        try:
            provider_class = registry[provider_name]
            provider_class(config)
            return True
        except Exception as e:
            raise ValueError(f"Provider configuration validation failed: {e}")


def create_transcription_provider(provider_name: str, config: Dict[str, Any]) -> TranscriptionInterface:
    """Convenience function to create transcription provider."""
    return ProviderFactory.create_transcription_provider(provider_name, config)


def create_response_provider(provider_name: str, config: Dict[str, Any]) -> ResponseInterface:
    """Convenience function to create response provider."""
    return ProviderFactory.create_response_provider(provider_name, config)


def create_tts_provider(provider_name: str, config: Dict[str, Any]) -> TextToSpeechInterface:
    """Convenience function to create TTS provider."""
    return ProviderFactory.create_tts_provider(provider_name, config)


def create_context_provider(provider_name: str, config: Dict[str, Any]) -> ContextInterface:
    """Convenience function to create context provider."""
    return ProviderFactory.create_context_provider(provider_name, config)