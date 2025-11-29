"""
Assistant Framework - Standardized interface for transcription, response, and TTS components.

This framework provides a clean abstraction layer for:
- Speech-to-text transcription (AssemblyAI)
- LLM response generation (OpenAI WebSocket with MCP)
- Text-to-speech synthesis (Google Cloud TTS)
- Conversation context management
- Wake word detection (OpenWakeWord)
- Barge-in interruption support

Usage (v2 - recommended):
    from assistant_framework.orchestrator_v2 import RefactoredOrchestrator
    from assistant_framework.config import get_framework_config
    
    config = get_framework_config()
    orchestrator = RefactoredOrchestrator(config)
    await orchestrator.initialize()
    await orchestrator.run_continuous_loop()
"""

from .orchestrator_v2 import RefactoredOrchestrator
from .factory import ProviderFactory
from .config import get_framework_config
from . import interfaces
from . import models
from . import providers

__version__ = "2.0.0"

__all__ = [
    'RefactoredOrchestrator',
    'ProviderFactory',
    'get_framework_config',
    'interfaces',
    'models',
    'providers'
]