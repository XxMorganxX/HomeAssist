"""
Assistant Framework - Standardized interface for transcription, response, and TTS components.

This framework provides a clean abstraction layer for:
- Speech-to-text transcription (AssemblyAI)
- LLM response generation (OpenAI WebSocket with MCP)
- Text-to-speech synthesis (Google Cloud TTS)
- Conversation context management

Usage:
    from assistant_framework import create_orchestrator
    from assistant_framework.config import get_framework_config
    
    # Create and initialize orchestrator
    config = get_framework_config()
    orchestrator = await create_orchestrator(config)
    
    # Run full pipeline
    await orchestrator.run_full_pipeline()
    
    # Or use components individually
    async for result in orchestrator.run_transcription_only():
        print(result.text)
"""

from .orchestrator import AssistantOrchestrator, create_orchestrator
from .factory import ProviderFactory
from .config import get_framework_config
from . import interfaces
from . import models
from . import providers

__version__ = "1.0.0"

__all__ = [
    'AssistantOrchestrator',
    'create_orchestrator', 
    'ProviderFactory',
    'get_framework_config',
    'interfaces',
    'models',
    'providers'
]