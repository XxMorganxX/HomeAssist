# Assistant Framework

A standardized interface architecture for voice assistant components with complete abstraction and easy provider swapping.

## Architecture Overview

```
assistant_framework/
├── interfaces/           # Abstract base classes
│   ├── transcription.py     # STT interface
│   ├── response.py          # LLM/API interface  
│   ├── text_to_speech.py   # TTS interface
│   ├── context.py           # Context management interface
│   └── wake_word.py         # Wake word detection interface
├── providers/           # Concrete implementations
│   ├── transcription/
│   │   └── assemblyai.py    # AssemblyAI WebSocket streaming
│   ├── response/
│   │   └── openai_websocket.py  # OpenAI Realtime API + MCP
│   ├── tts/
│   │   └── google_tts.py    # Google Cloud TTS
│   ├── context/
│   │   └── unified_context.py   # Token-aware context manager
│   └── wakeword/
│       └── openwakeword_provider.py  # OpenWakeWord-based detection
├── models/              # Common data structures
│   └── data_models.py       # TranscriptionResult, ResponseChunk, etc.
├── factory.py           # Provider instantiation
├── orchestrator.py      # Main pipeline orchestrator
├── config.py            # Static configuration
└── example.py           # Usage examples
```

## Key Features

- **Complete Abstraction**: Components don't know about each other's implementations
- **Easy Provider Swapping**: Change providers by updating config only
- **Streaming First**: Real-time transcription and response generation
- **Independent Components**: Each component can be used standalone
- **Type Safety**: Full type hints and data validation
- **Fail Fast**: Provider failures cause clean program exit
- **Context Management**: Swappable conversation context with token counting
- **MCP Integration**: Tool calling through Model Context Protocol

## Quick Start

### 1. Environment Setup
```bash
# Required environment variables
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google/service/account.json"
```

### 2. Basic Usage
```python
import asyncio
from assistant_framework import create_orchestrator
from assistant_framework.config import get_framework_config

async def main():
    # Get configuration
    config = get_framework_config()
    
    # Create and initialize orchestrator
    orchestrator = await create_orchestrator(config)
    
    # Run full pipeline (transcription → response → TTS)
    await orchestrator.run_full_pipeline()
    
    # Or process a single message
    result = await orchestrator.process_single_message(
        "What's the weather like today?",
        use_context=True,
        generate_tts=True
    )
    
    await orchestrator.cleanup()

asyncio.run(main())
```

### 3. Individual Components
```python
# Use components independently
async for transcription in orchestrator.run_transcription_only():
    print(f"Heard: {transcription.text}")

async for response_chunk in orchestrator.run_response_only("Hello"):
    print(response_chunk.content, end="")

audio = await orchestrator.run_tts_only("Hello Mr. Stuart")
```

## Configuration

Static provider selection in `config.py`:

```python
TRANSCRIPTION_PROVIDER = "assemblyai"
RESPONSE_PROVIDER = "openai_websocket"  
TTS_PROVIDER = "google_tts"
CONTEXT_PROVIDER = "unified"
```

## Supported Providers

### Transcription
- **assemblyai**: Real-time WebSocket streaming with turn formatting

### Response  
- **openai_websocket**: OpenAI Realtime API with MCP tool integration

### Text-to-Speech
- **google_tts**: Google Cloud TTS with HD voices and ffmpeg post-processing

### Context
- **unified**: Token-aware conversation management with auto-trimming

### Wake Word
- **openwakeword**: Local wake word detection using OpenWakeWord

## Adding New Providers

1. Implement the appropriate interface
2. Add to factory registry
3. Update configuration
4. No changes to existing code required

## Examples

Run the example script to see different usage patterns:

```bash
cd assistant_framework
python example.py full        # Full voice pipeline
python example.py message     # Single message processing  
python example.py components  # Individual components
python example.py context     # Context management
```

## Dependencies

- `assemblyai` - For AssemblyAI transcription
- `openai` - For OpenAI API integration
- `google-cloud-texttospeech` - For Google TTS
- `mcp` - For Model Context Protocol
- `pyaudio` - For microphone access
- `pygame` - For audio playback
- `aiohttp` - For WebSocket connections
- `tiktoken` - For token counting

## Error Handling

The framework uses a "fail fast" approach:
- Provider initialization failures stop the program
- Network errors are propagated up
- Clean resource cleanup on exit
- Detailed error messages and logging