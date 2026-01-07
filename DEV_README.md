# HomeAssist Developer Reference

This document provides detailed implementation documentation for developers working on or extending the HomeAssist codebase.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Provider Pattern](#provider-pattern)
3. [Orchestrator & State Machine](#orchestrator--state-machine)
4. [Audio Pipeline](#audio-pipeline)
5. [MCP Tool System](#mcp-tool-system)
6. [Memory Systems](#memory-systems)
7. [Configuration System](#configuration-system)
8. [Data Models](#data-models)
9. [Scheduled Jobs](#scheduled-jobs)
10. [Error Handling](#error-handling)
11. [Common Development Tasks](#common-development-tasks)

---

## Architecture Overview

### Directory Structure

```
HomeAssistV2/
├── assistant_framework/          # Core voice assistant engine
│   ├── __main__.py               # Package entry point
│   ├── main_v2.py                # CLI and startup logic
│   ├── orchestrator_v2.py        # Main coordination (1900 lines)
│   ├── config.py                 # All configuration (1185 lines)
│   ├── factory.py                # Provider instantiation
│   ├── interfaces/               # Abstract base classes (ABCs)
│   │   ├── transcription.py      # TranscriptionInterface
│   │   ├── response.py           # ResponseInterface  
│   │   ├── text_to_speech.py     # TextToSpeechInterface
│   │   ├── wake_word.py          # WakeWordInterface
│   │   ├── context.py            # ContextInterface
│   │   ├── termination.py        # TerminationInterface
│   │   ├── embedding.py          # EmbeddingInterface
│   │   └── vector_store.py       # VectorStoreInterface
│   ├── providers/                # Concrete implementations
│   │   ├── transcription_v2/     # AssemblyAI, OpenAI Whisper
│   │   ├── response/             # OpenAI WebSocket
│   │   ├── tts/                  # Piper, Google, Local, Chatterbox
│   │   ├── wakeword_v2/          # OpenWakeWord (process-isolated)
│   │   ├── context/              # UnifiedContextProvider
│   │   ├── termination/          # IsolatedTerminationProvider
│   │   ├── embedding/            # OpenAI embeddings
│   │   └── vector_store/         # Supabase pgvector
│   ├── models/                   # Data structures
│   │   └── data_models.py        # Dataclasses for all components
│   └── utils/                    # Shared utilities
│       ├── state_machine.py      # AudioStateMachine
│       ├── barge_in.py           # Interrupt detection
│       ├── persistent_memory.py  # Long-term fact storage
│       ├── vector_memory.py      # Semantic search
│       ├── conversation_summarizer.py
│       ├── briefing_manager.py   # Supabase briefing CRUD
│       ├── briefing_processor.py # LLM opener generation
│       ├── conversation_recorder.py  # Session logging
│       ├── console_logger.py     # Remote dashboard
│       ├── audio_manager.py      # Device handling
│       ├── device_manager.py     # Bluetooth/device detection
│       ├── tones.py              # Audio feedback sounds
│       ├── error_handling.py     # Recovery strategies
│       └── logging_config.py     # Verbose/quiet modes
│
├── mcp_server/                   # Model Context Protocol server
│   ├── server.py                 # FastMCP entry point
│   ├── tool_registry.py          # Dynamic tool discovery
│   ├── mcp_adapter.py            # BaseTool → FastMCP bridge
│   ├── base_tool.py              # Abstract tool base class
│   ├── tools_config.py           # Enable/disable tools
│   ├── config.py                 # MCP server settings
│   ├── tools/                    # Tool implementations
│   │   ├── weather.py            # Weather forecasts
│   │   ├── calendar.py           # Google Calendar
│   │   ├── spotify.py            # Music control
│   │   ├── kasa_lighting.py      # Smart lights
│   │   ├── sms.py                # macOS Messages
│   │   ├── notifications.py      # Email/news retrieval
│   │   ├── google_search.py      # Web search
│   │   ├── state_tool.py         # App state access
│   │   ├── system_info.py        # Self-documentation
│   │   └── cursor.py             # IDE integration
│   └── clients/                  # External service clients
│       ├── calendar_client.py    # Google Calendar API
│       ├── weather_client.py     # Open-Meteo API
│       ├── web_search_client.py  # Google search
│       └── kasa_lighting_client.py
│
├── scripts/scheduled/            # Background jobs (GitHub Actions)
│   ├── scheduled_events.py       # Job runner
│   ├── email_summarizer/         # Gmail → AI summary
│   ├── news_summary/             # NewsAPI → AI digest
│   ├── calendar_briefing/        # Reminder announcements
│   ├── weather_briefing/         # Unusual weather alerts
│   └── reminder_analyzer/        # Event analysis
│
├── state_management/             # Runtime state files
│   ├── app_state.json            # User prefs, notifications
│   ├── persistent_memory.json    # Long-term facts
│   └── conversation_summary.json # Current session
│
├── audio_data/                   # Model files
│   ├── wake_word_models/         # OpenWakeWord .onnx files
│   ├── piper_models/             # Piper TTS .onnx files
│   └── chatterbox_models/        # Chatterbox neural TTS
│
└── google_creds/                 # OAuth credentials
    ├── google_creds_*.json       # Client secrets per user
    └── token_*.json              # Access tokens per user
```

### Execution Flow

```
1. python -m assistant_framework.main_v2 continuous

2. main_v2.py:
   └── ensure_first_time_setup()     # Interactive if app_state.json missing
   └── get_framework_config()         # Assemble all config
   └── RefactoredOrchestrator(config) # Create orchestrator
   └── orchestrator.initialize()      # Start all providers + MCP
   └── orchestrator.run_continuous_loop()

3. Continuous loop:
   ┌─────────────────────────────────────────────────────────────────┐
   │  WAKE_WORD_LISTENING                                            │
   │  └── IsolatedOpenWakeWordProvider runs in subprocess            │
   │  └── Yields WakeWordEvent on detection                          │
   ├─────────────────────────────────────────────────────────────────┤
   │  [Optional] SYNTHESIZING (briefing announcements)               │
   │  └── BriefingManager.get_pending_briefings_with_opener()        │
   │  └── TTS speaks pre-generated opener (no LLM latency)           │
   ├─────────────────────────────────────────────────────────────────┤
   │  TRANSCRIBING (with parallel termination detection)             │
   │  └── AssemblyAIAsyncProvider streams to WebSocket               │
   │  └── Detects send phrases ("sir", "send it")                    │
   │  └── Returns final text                                         │
   ├─────────────────────────────────────────────────────────────────┤
   │  PROCESSING_RESPONSE                                            │
   │  └── UnifiedContextProvider adds message to history             │
   │  └── VectorMemoryManager.get_context_enrichment() (parallel)    │
   │  └── OpenAIWebSocketResponseProvider.stream_response()          │
   │      └── Persistent WebSocket to OpenAI Realtime API            │
   │      └── MCP tool discovery and execution                       │
   │      └── Composed tool calling (multi-step tasks)               │
   ├─────────────────────────────────────────────────────────────────┤
   │  SYNTHESIZING (with barge-in detection)                         │
   │  └── PiperTTSProvider.synthesize() or chunked                   │
   │  └── BargeInDetector monitors microphone                        │
   │  └── User speech interrupts and captures audio                  │
   └─────────────────────────────────────────────────────────────────┘
   
4. On conversation end:
   └── ConversationRecorder.end_session()
   └── UnifiedContextProvider.on_conversation_end()
       └── PersistentMemoryManager.update_after_conversation()
       └── VectorMemoryManager.store_conversation()
```

---

## Provider Pattern

### Interface System

All components implement abstract interfaces defined in `assistant_framework/interfaces/`. This enables swappable implementations.

**TranscriptionInterface** (`interfaces/transcription.py`)
```python
class TranscriptionInterface(ABC):
    @abstractmethod
    async def initialize(self) -> bool: ...
    
    @abstractmethod
    async def start_streaming(self) -> AsyncIterator[TranscriptionResult]: ...
    
    @abstractmethod
    async def stop_streaming(self) -> None: ...
    
    @property
    @abstractmethod
    def is_active(self) -> bool: ...
    
    @abstractmethod
    async def cleanup(self) -> None: ...
```

**ResponseInterface** (`interfaces/response.py`)
```python
class ResponseInterface(ABC):
    @abstractmethod
    async def initialize(self) -> bool: ...
    
    @abstractmethod
    async def stream_response(
        self, 
        message: str,
        context: Optional[List[Dict]] = None,
        tool_context: Optional[List[Dict]] = None
    ) -> AsyncIterator[ResponseChunk]: ...
    
    @abstractmethod
    async def get_available_tools(self) -> List[Dict]: ...
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict) -> str: ...
    
    @abstractmethod
    async def cleanup(self) -> None: ...
```

### Provider Factory

`factory.py` maps provider names to classes:

```python
class ProviderFactory:
    TRANSCRIPTION_PROVIDERS = {
        'assemblyai': AssemblyAIAsyncProvider,
        'openai_whisper': OpenAIWhisperProvider,
    }
    
    TTS_PROVIDERS = {
        'google_tts': GoogleTTSProvider,
        'local_tts': LocalTTSProvider,
        'chatterbox': ChatterboxTTSProvider,
        'piper': PiperTTSProvider,
    }
    
    # ... other provider types
    
    @classmethod
    def create_all_providers(cls, config: Dict) -> Dict[str, Any]:
        """Create all providers from config dict."""
        providers = {}
        # Iterates config keys, calls appropriate create_*_provider()
        return providers
```

### Provider Selection

Selected in `config.py`:

```python
TRANSCRIPTION_PROVIDER = "assemblyai"      # or "openai_whisper"
RESPONSE_PROVIDER = "openai_websocket"     # Only option currently
TTS_PROVIDER = "piper"                     # or "google_tts", "local_tts", "chatterbox"
CONTEXT_PROVIDER = "unified"               # Only option currently
WAKEWORD_PROVIDER = "openwakeword"         # Only option currently
```

### Adding a New Provider

1. **Create interface** (if new type) in `interfaces/`
2. **Implement provider** in `providers/<type>/`
3. **Register in factory.py**:
   ```python
   TRANSCRIPTION_PROVIDERS = {
       'assemblyai': AssemblyAIAsyncProvider,
       'openai_whisper': OpenAIWhisperProvider,
       'my_new_provider': MyNewProvider,  # Add here
   }
   ```
4. **Add config** in `config.py`
5. **Select via** `TRANSCRIPTION_PROVIDER = "my_new_provider"`

---

## Orchestrator & State Machine

### RefactoredOrchestrator

The orchestrator (`orchestrator_v2.py`, ~1900 lines) coordinates all components. Key responsibilities:

- **Provider lifecycle** — Creates, initializes, and cleans up all providers
- **State transitions** — Uses AudioStateMachine for safe component switching
- **Conversation flow** — Wake word → transcription → response → TTS loop
- **Barge-in handling** — Interrupt TTS when user speaks
- **Termination detection** — Parallel "over out" phrase monitoring
- **Recording** — Logs sessions to Supabase
- **Memory management** — Triggers memory updates on conversation end

### AudioStateMachine

`utils/state_machine.py` manages component lifecycle:

```python
class AudioState(Enum):
    IDLE = auto()
    WAKE_WORD_LISTENING = auto()
    TRANSCRIBING = auto()
    PROCESSING_RESPONSE = auto()
    SYNTHESIZING = auto()
    TRANSITIONING = auto()
    ERROR = auto()
```

**Valid Transitions (Production):**

```
IDLE → WAKE_WORD_LISTENING, ERROR
WAKE_WORD_LISTENING → TRANSCRIBING, PROCESSING_RESPONSE, SYNTHESIZING, IDLE, ERROR
TRANSCRIBING → PROCESSING_RESPONSE, IDLE, ERROR
PROCESSING_RESPONSE → SYNTHESIZING, IDLE, ERROR
SYNTHESIZING → IDLE, WAKE_WORD_LISTENING, TRANSCRIBING, ERROR
ERROR → IDLE
```

**Key Methods:**

```python
async def transition_to(
    self,
    target_state: AudioState,
    component: Optional[str] = None,  # "wakeword", "transcription", etc.
    metadata: Optional[Dict] = None
) -> bool:
    """
    1. Validates transition is allowed
    2. Calls cleanup handler for previous component
    3. Waits for audio settling (if audio components switching)
    4. Updates state
    """

async def emergency_reset(self):
    """Force to IDLE, cleanup all components. Idempotent."""
```

**Cleanup Handlers:**

Registered during orchestrator init:

```python
self.state_machine.register_cleanup_handler("wakeword", cleanup_wakeword)
self.state_machine.register_cleanup_handler("transcription", cleanup_transcription)
self.state_machine.register_cleanup_handler("response", cleanup_response)
self.state_machine.register_cleanup_handler("tts", cleanup_tts)
```

Handlers are **idempotent** — safe to call multiple times:

```python
async def cleanup_transcription():
    if self._transcription:
        if getattr(self._transcription, 'is_active', False):
            await self._transcription.stop_streaming()
```

### Process Isolation

Wake word and termination detection run in **separate processes** to prevent model corruption:

```python
# IsolatedOpenWakeWordProvider
class IsolatedOpenWakeWordProvider(WakeWordInterface):
    """
    Runs OpenWakeWord in a subprocess via multiprocessing.
    
    Why process isolation?
    - ONNX models can corrupt when audio streams are interrupted
    - Subprocess crash doesn't affect main application
    - Models stay loaded between conversations (warm mode)
    """
```

**Warm Mode** (default): Subprocess stays alive, audio stream pauses/resumes (~200ms restart)

**Cold Mode**: Full subprocess termination and restart (~2-3s)

---

## Audio Pipeline

### Transcription Flow

**AssemblyAI (default):**

```python
class AssemblyAIAsyncProvider(TranscriptionInterface):
    """
    Real-time streaming transcription via WebSocket.
    
    Flow:
    1. preconnect() — Establish WebSocket during TTS (overlapped)
    2. start_streaming() — Begin audio capture + streaming
    3. Yields TranscriptionResult (partial/final)
    4. stop_streaming() — Close WebSocket, stop audio
    """
```

**Prefill Audio** (for barge-in):

```python
def set_prefill_audio(self, audio_bytes: bytes):
    """
    Pre-captured audio from barge-in detector.
    Sent immediately when transcription starts.
    """
    self._prefill_audio = audio_bytes
```

### Barge-In Detection

`utils/barge_in.py` enables interrupting TTS:

```python
class BargeInDetector:
    """
    Energy-based speech detection during TTS playback.
    
    Config:
    - energy_threshold: Voice detection sensitivity (0.03-0.15)
    - min_speech_duration: Required speech before triggering
    - cooldown_after_tts_start: Ignore speech briefly after TTS begins
    - buffer_seconds: Audio to capture before barge-in
    - capture_after_trigger: Extra capture after detection
    """
    
    async def start(self, on_barge_in: Callable):
        """Start monitoring. Calls on_barge_in() when speech detected."""
    
    def get_captured_audio_bytes(self) -> Optional[bytes]:
        """Get audio captured during barge-in for transcription prefill."""
```

**Early Barge-In** (within 3 seconds of response start):
- Appends new message to previous message
- Enables natural corrections: "No wait, I meant..."

### TTS Pipeline

**Piper (default):**

```python
class PiperTTSProvider(TextToSpeechInterface):
    """
    Fast local neural TTS using ONNX.
    ~50x realtime, 15-100MB models, CPU-only.
    
    Methods:
    - synthesize(text) → bytes
    - play_audio_async(audio) → None
    - synthesize_and_play_chunked(text) → None (for long messages)
    - stop_audio() → None (for barge-in)
    """
```

**Chunked Synthesis** (for latency reduction):

```python
PIPER_TTS_CONFIG = {
    "chunked_synthesis_threshold": 150,  # Chars above which to chunk
    "chunk_max_length": 150,             # Max chars per chunk
}
# First chunk plays while rest synthesizes → faster perceived response
```

### Device Management

`utils/device_manager.py` handles audio device detection:

```python
def get_audio_device_config() -> AudioDeviceConfig:
    """
    Auto-detect audio device and return optimized settings.
    
    Returns:
        AudioDeviceConfig with:
        - device_index: Input device ID
        - device_name: Human-readable name
        - is_bluetooth: True for Bluetooth/Meta Ray-Bans
        - sample_rate: 16000 (always)
        - blocksize: Optimized for device type
        - latency: 'high' for Bluetooth, 'low' otherwise
    """
```

**Bluetooth Handling:**

- Lower energy threshold (mic quality drops during playback)
- Higher latency setting
- Suppress overflow warnings (bursty audio is expected)

---

## MCP Tool System

### Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OpenAI Realtime API                                        │
│  └── Sends tool call request                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  OpenAIWebSocketResponseProvider                            │
│  └── MCP Client (stdio transport)                           │
│  └── mcp_session.call_tool(name, arguments)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Server (mcp_server/server.py)                          │
│  └── FastMCP framework                                      │
│  └── ToolRegistry discovers tools from tools/               │
│  └── MCPToolAdapter bridges BaseTool → FastMCP              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  BaseTool.execute(params)                                   │
│  └── Tool-specific logic                                    │
│  └── Returns Dict[str, Any] with success/error              │
└─────────────────────────────────────────────────────────────┘
```

### BaseTool Abstract Class

All tools extend `base_tool.py`:

```python
class BaseTool(ABC):
    name: str = None           # Tool name for MCP
    description: str = None    # LLM sees this
    version: str = "1.0.0"
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with params, return result dict."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for parameters."""
        pass
    
    def to_fastmcp_function(self):
        """
        Convert to FastMCP-compatible async function.
        Generates dynamic function with proper signature and docstring.
        """
```

### Creating a New Tool

1. **Create file** `mcp_server/tools/my_tool.py`:

```python
from typing import Any, Dict
from mcp_server.base_tool import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    version = "1.0.0"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "This is required"
                },
                "optional_param": {
                    "type": "integer",
                    "description": "This is optional",
                    "default": 10
                }
            },
            "required": ["required_param"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Do work
            result = f"Processed {params['required_param']}"
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

2. **Enable in** `mcp_server/tools_config.py`:

```python
TOOL_CONFIG = {
    "my_tool": {
        "enabled": True,
        # Optional tool-specific config
    }
}
```

3. **Tool is auto-discovered** — No registration needed

### Tool Discovery

`tool_registry.py` auto-discovers tools:

```python
class ToolRegistry:
    def discover_tools(self) -> List[str]:
        """
        Scans mcp_server/tools/*.py
        Finds classes extending BaseTool
        Checks tools_config.py for enabled status
        """
    
    def get_tool_instance(self, tool_name: str) -> BaseTool:
        """Get or create singleton tool instance."""
```

### Composed Tool Calling

Multi-step task execution (`orchestrator_v2.py`):

```python
async def _iterative_tool_execution(
    self,
    user_message: str,
    context: List[Dict],
    initial_tool_calls: List[ToolCall],
    instructions: str = ""
) -> tuple:
    """
    Execute tools iteratively for multi-step tasks.
    
    Example: "Find the weather and text it to me"
    1. Initial: weather tool → gets forecast
    2. Check: "Need more tools?" → Yes, need SMS
    3. Execute: SMS tool with weather as content
    4. Check: "Need more tools?" → No
    5. Compose final answer
    
    Safeguards:
    - max_tool_iterations (default: 5)
    - Duplicate detection (same tool + args blocked)
    - Calendar write dedup (only 1 create_event per request)
    """
```

---

## Memory Systems

### Three-Tier Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  1. Conversation Context (within session)                   │
│     └── UnifiedContextProvider                              │
│     └── Stores: Messages, token counts, summaries           │
│     └── Lifetime: Single conversation session               │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Persistent Memory (across all sessions)                 │
│     └── PersistentMemoryManager                             │
│     └── Stores: Facts, preferences, patterns                │
│     └── Lifetime: Permanent (state_management/persistent_memory.json) │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Vector Memory (semantic search)                         │
│     └── VectorMemoryManager                                 │
│     └── Stores: Conversation embeddings (3072-dim)          │
│     └── Lifetime: 90 days (Supabase pgvector)               │
└─────────────────────────────────────────────────────────────┘
```

### Conversation Context

`providers/context/unified_context.py`:

```python
class UnifiedContextProvider(ContextInterface):
    """
    Manages conversation history within a session.
    
    Key methods:
    - add_message(role, content) — Add user/assistant message
    - get_recent_for_response() — System prompt + last N messages
    - get_tool_context() — Compact context for tool decisions
    - auto_trim_if_needed() — Enforce max_messages limit
    - reset() — Start new session (injects persistent memory)
    """
    
    # Context retrieval strategies:
    def get_full_history(self) -> List[Dict]:
        """All messages (for summarization)."""
    
    def get_recent_for_response(self) -> List[Dict]:
        """System + last 8 messages (default)."""
    
    def get_tool_context(self, max_user=3, max_assistant=2) -> List[Dict]:
        """Compact context for tool decisions."""
```

**Summarization** (background, non-blocking):

```python
"summarization": {
    "enabled": True,
    "first_summary_at": 8,    # Messages before first summary
    "summarize_every": 4,      # Re-summarize every N after first
    "gemini_model": "gemini-2.0-flash"
}
```

### Persistent Memory

`utils/persistent_memory.py`:

```python
class PersistentMemoryManager:
    """
    Extracts lasting facts from conversation summaries.
    
    Updated at end of each conversation via LLM extraction.
    Injected into system prompt on session start.
    
    Storage format (persistent_memory.json):
    {
        "user_profile": {
            "name": "Morgan",
            "location": "Ithaca, NY",
            "preferences": {"lighting": "warm", "music": "jazz"}
        },
        "known_facts": [
            "Works in AI research",
            "Has a dog named Max"
        ],
        "patterns": [
            {"pattern": "checks weather every morning", "strength": "strong"},
            {"pattern": "prefers concise answers", "strength": "moderate"}
        ]
    }
    """
    
    def update_after_conversation(self, summary: str):
        """
        Background thread: 
        1. Send summary + existing memory to LLM
        2. Extract new facts, preferences, patterns
        3. Merge (avoiding duplicates, handling removals)
        4. Save to JSON
        """
    
    def get_memory_summary(self) -> str:
        """Format memory for system prompt injection."""
```

**Pattern Strength Levels:**

| Level | Meaning | Promotion Criteria |
|-------|---------|-------------------|
| `weak` | Speculative, single instance | New observation |
| `moderate` | Some evidence | 2-3 occurrences |
| `strong` | Clear recurring behavior | Multiple data points |
| `confirmed` | Practically a fact | Overwhelming evidence |

### Vector Memory

`utils/vector_memory.py`:

```python
class VectorMemoryManager:
    """
    Semantic search over past conversations.
    
    Flow:
    1. Conversation ends → summary generated
    2. Summary → OpenAI embedding (3072-dim)
    3. Embedding stored in Supabase pgvector
    4. On new conversation: query finds relevant past contexts
    5. Injected as system message before response
    
    Features:
    - K-fold partitioning for long conversations
    - Local numpy cache for fast search (~1-5ms vs ~50-200ms remote)
    - Smart retrieval (top 2 if similar, else top 1)
    - 90-day expiry
    """
    
    async def store_conversation(self, summary: str):
        """Embed and store summary."""
    
    async def get_context_enrichment(self, user_message: str) -> str:
        """Search for relevant past conversations, format for prompt."""
    
    async def preload_recent_vectors(self) -> bool:
        """Pre-warm cache during idle for faster first query."""
```

**Supabase Table:**

```sql
CREATE TABLE conversation_memories (
    id TEXT PRIMARY KEY,
    embedding vector(3072),
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Configuration System

### config.py Structure

The configuration file (~1185 lines) is organized into numbered sections:

| Section | Purpose |
|---------|---------|
| 0 | Logging verbosity |
| 0B | Dynamic user config (from app_state.json) |
| 1 | Environment & credentials |
| 2 | Provider selection |
| 3 | Transcription config |
| 4 | Response/LLM config |
| 5 | TTS config |
| 6 | Wake word config |
| 6A | Termination detection |
| 6B | Briefing processor |
| 7 | Context/memory config |
| 8 | MCP server paths |
| 9 | Conversation flow (phrases, timeouts) |
| 10 | Recording (Supabase) |
| 11 | Barge-in config |
| 11B | Latency/turnaround tuning |
| 12 | Framework assembly (`get_framework_config()`) |
| 13 | Environment presets |
| 14 | Runtime provider switching |
| 15 | Validation & diagnostics |

### System Prompt Builder

`build_system_prompt()` converts structured dict to formatted string:

```python
SYSTEM_PROMPT_CONFIG = {
    "name": "Sol",
    "name_full": "Solas",
    "role": "philosophical_mentor",
    "vibe": ["natural conversation", "thoughtful", "curious"],
    "north_star": "Leave the user with clearer thinking...",
    "voice": {
        "tone": "calm, grounded, quietly sharp",
        "cadence": "loose paragraphs like natural speech"
    },
    "formatting": {...},
    "metaphor": {...},
    "profanity": {...},
    "tools": {...},
    "behavior": [...],
    "transparency": {...},
    "response_shape": [...],
    "example": "..."
}

# Automatically includes current date/time and user from app_state.json
SYSTEM_PROMPT = build_system_prompt(SYSTEM_PROMPT_CONFIG)
```

### Configuration Presets

```python
CONFIG_PRESET = "dev"  # "default", "dev", "prod", "test"

def get_config_for_preset(preset: str) -> Dict:
    """
    dev: enable_debug=True, max_tokens=1000
    prod: enable_debug=False, max_tokens=2000
    test: enable_debug=True, max_tokens=500, max_messages=5
    """
```

### Runtime Provider Switching

```python
def set_providers(
    transcription: Optional[str] = None,
    response: Optional[str] = None,
    tts: Optional[str] = None,
    context: Optional[str] = None,
    wakeword: Optional[str] = None
):
    """Override providers at runtime (before initialization)."""
```

### Validation

```python
def validate_environment() -> Dict[str, Any]:
    """
    Returns:
    {
        'valid': bool,
        'errors': ['Missing OPENAI_API_KEY', ...],
        'warnings': ['MCP server not found', ...],
        'info': ['Google credentials: /path/to/file', ...]
    }
    """

def print_config_summary():
    """Print formatted config status to console."""
```

---

## Data Models

All shared data structures in `models/data_models.py`:

### Core Types

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    PCM16 = "pcm16"
    OGG = "ogg"
    FLAC = "flac"
```

### Transcription

```python
@dataclass
class TranscriptionResult:
    text: str
    is_final: bool
    timestamp: float
    confidence: Optional[float] = None
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Response/LLM

```python
@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    result: Optional[str] = None

@dataclass
class ResponseChunk:
    content: str
    is_complete: bool
    timestamp: float
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_tool_calls(self) -> bool: ...
    def get_full_content(self) -> str: ...
```

### Audio

```python
@dataclass
class AudioOutput:
    audio_data: bytes
    format: AudioFormat
    sample_rate: int
    duration: Optional[float] = None
    voice: Optional[str] = None
```

### Events

```python
@dataclass
class WakeWordEvent:
    model_name: str
    score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TerminationEvent:
    phrase_name: str  # "over_out", "stop_listening"
    score: float
    timestamp: float
    interrupted_state: Optional[str] = None  # "SYNTHESIZING", etc.
```

### Conversation

```python
@dataclass
class ConversationMessage:
    role: MessageRole
    content: str
    timestamp: float
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMessage': ...
```

---

## Scheduled Jobs

### Job Runner

`scripts/scheduled/scheduled_events.py`:

```python
def main():
    """
    1. cleanup_ephemeral_data() — Clear ephemeral_data/ folders
    2. run_email_summarizer() — Subprocess email_summarizer/main.py
    3. run_news_summary() — Subprocess news_summary/main.py
    4. export_notifications() — Write to app_state.json
    """
```

### Email Summarizer

`scripts/scheduled/email_summarizer/`:

1. `fetch_mail.py` — Gmail API fetch (OAuth)
2. `mail_summary.py` — Gemini summarization
3. `mail_main.py` — Orchestrates + stores to Supabase `notification_sources`

**Output:** `ephemeral_data/email_summaries.json`, Supabase rows

### News Summary

`scripts/scheduled/news_summary/`:

1. `news_scraper.py` — NewsAPI fetch
2. `news_ai.py` — OpenAI analysis
3. `news_main.py` — Orchestrates + stores to Supabase

**Output:** `ephemeral_data/news_summary.json`, Supabase rows

### Calendar Briefing

`scripts/scheduled/calendar_briefing/`:

1. Fetches 7 days of calendar events
2. Filters already-processed (via `calendar_event_cache` table)
3. AI analyzes optimal reminder timing
4. Creates `briefing_announcements` with pre-generated openers
5. Uses `{{TIME_UNTIL_EVENT}}` placeholder for dynamic timing

### Weather Briefing

`scripts/scheduled/weather_briefing/`:

1. Auto-detects location (IP geolocation or configured)
2. Fetches 7-day forecast from Open-Meteo
3. Analyzes for unusual conditions relative to weekly average
4. Only creates briefings when alerts detected

**Alert Types:**
- Rain/snow/thunderstorms/freezing precipitation
- Extreme heat/cold (seasonally adjusted)
- High winds (>40mph gusts)

---

## Error Handling

### Recovery Strategies

`utils/error_handling.py`:

```python
class ErrorSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    RECOVERABLE = auto()
    FATAL = auto()

@dataclass
class ComponentError:
    component: str  # "wakeword", "transcription", etc.
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None

class ErrorHandler:
    def register_recovery(self, component: str, handler: Callable):
        """Register component-specific recovery."""
    
    async def handle_error(self, error: ComponentError):
        """
        1. Log error
        2. Call recovery handler if registered
        3. Track error counts for circuit breaking
        """
```

**Orchestrator Recovery Strategies:**

```python
# Registered during init
async def recover_wakeword(error):
    if self._wakeword:
        await self._wakeword.cleanup()
        self._wakeword = None
    await asyncio.sleep(1.0)
    # Recreated on next use

async def recover_transcription(error):
    if self._transcription:
        await self._transcription.cleanup()
        self._transcription = None
    await asyncio.sleep(1.0)
```

### State Machine Safety

Invalid transitions raise `ValueError`:

```python
try:
    await self.state_machine.transition_to(AudioState.TRANSCRIBING)
except ValueError as e:
    # Invalid transition (e.g., PROCESSING_RESPONSE → TRANSCRIBING)
    await self.state_machine.emergency_reset()
```

### Process Isolation Safety

Wake word/termination subprocess crashes don't affect main:

```python
# In subprocess
except Exception as e:
    # Crashes here are isolated
    pass

# Main process
try:
    async for event in self._wakeword.start_detection():
        ...
except Exception as e:
    # Main continues, can reinitialize
    await self._wakeword.cleanup()
    self._wakeword = await self._create_wakeword_provider()
```

---

## Common Development Tasks

### Running the Assistant

```bash
# Continuous mode (production)
python -m assistant_framework.main_v2 continuous

# Single conversation
python -m assistant_framework.main_v2 single

# Test wake word only
python -m assistant_framework.main_v2 wakeword

# Test transcription only
python -m assistant_framework.main_v2 transcribe

# Show config
python -m assistant_framework.main_v2 config
```

### Running MCP Server Standalone

```bash
cd mcp_server
./run.sh  # Creates venv, installs deps, starts server

# Or manually
python server.py --host 127.0.0.1 --port 3000 --transport http
```

### Testing Tools

```bash
# Check tool status
python scripts/show_tool_status.py

# Test vector memory
python scripts/test_vector_memory.py
```

### Running Scheduled Jobs

```bash
# All jobs
python scripts/scheduled/scheduled_events.py

# Individual jobs
python scripts/scheduled/email_summarizer/mail_main.py
python scripts/scheduled/news_summary/news_main.py
python scripts/scheduled/calendar_briefing/main.py
python scripts/scheduled/weather_briefing/main.py
```

### Debugging Tips

1. **Enable verbose logging:**
   ```bash
   export VERBOSE_LOGGING=true
   ```

2. **Check config validation:**
   ```python
   from assistant_framework.config import print_config_summary
   print_config_summary()
   ```

3. **View state machine history:**
   ```python
   history = orchestrator.state_machine.get_transition_history(last_n=20)
   for t in history:
       print(f"{t.from_state.name} → {t.to_state.name} ({t.component})")
   ```

4. **Check memory contents:**
   ```bash
   cat state_management/persistent_memory.json | python -m json.tool
   ```

5. **View vector cache stats:**
   ```python
   if orchestrator._context and orchestrator._context._vector_memory:
       stats = orchestrator._context._vector_memory.get_cache_stats()
       print(stats)
   ```

### Adding Environment Variables

1. Add to `.env`
2. Load in `config.py`:
   ```python
   MY_NEW_VAR = os.getenv("MY_NEW_VAR", "default")
   ```
3. Update `SETUP.md` (per workspace rule):
   ```markdown
   - `MY_NEW_VAR` — Description of what this enables
   ```

### Database Migrations

Supabase tables are documented in `SETUP.md`. Key tables:

- `conversation_memories` — Vector storage (3072-dim pgvector)
- `notification_sources` — Email/news summaries
- `briefing_announcements` — Wake word briefings
- `calendar_event_cache` — Processed event dedup

---

## Key Implementation Details

### OpenAI Realtime WebSocket

`providers/response/openai_websocket.py`:

- **Persistent connection** — Reused across requests (~300-500ms saved)
- **Session updates** — System prompt set via `session.update`, not message role
- **Tool execution** — MCP client via stdio transport
- **Composed calling** — Iterative tool execution with gpt-4o-mini for decisions
- **Token tracking** — Composition API calls tracked separately

### Barge-In Early Detection

When user interrupts within 3 seconds of response start:

```python
if elapsed < self._early_barge_in_threshold:
    self._early_barge_in = True
    # Next message appends to previous
```

This enables natural corrections without losing context.

### Briefing Pre-Generation

`utils/briefing_processor.py`:

```python
# Input sources call this after inserting briefings:
processor = BriefingProcessor()
await processor.process_pending_briefings(user="Morgan", briefing_manager=manager)

# On wake word, assistant fetches pre-generated opener:
opener = manager.get_combined_opener(briefings_with_opener)
await self.run_tts(opener)  # TTS only, no LLM latency
```

### Audio Feedback Tones

`utils/tones.py` provides system sounds:

- `beep_wake_detected()` — Wake word activated
- `beep_listening_start()` — Recording started
- `beep_send_detected()` — Send phrase recognized
- `beep_ready_to_listen()` — Ready for next question
- `beep_shutdown()` — Conversation ending
- `beep_tool_success()` — Tool executed successfully
- `beep_tool_failure()` — Tool failed

Auto-detected from tool response JSON:
```python
if '"success": true' in result or no error field:
    beep_tool_success()
else:
    beep_tool_failure()
```

---

## Performance Considerations

### Latency Optimization

| Optimization | Impact | Config |
|--------------|--------|--------|
| Persistent WebSocket | ~300-500ms saved per request | Automatic |
| Transcription preconnect | Overlapped with wake word stop | Automatic |
| Vector cache | ~1-5ms vs ~50-200ms remote | `local_cache_enabled: True` |
| Chunked TTS | Reduced perceived latency | `chunked_synthesis_threshold: 150` |
| Warm mode wake word | ~200ms vs ~2-3s restart | `wake_word_warm_mode: True` |
| Briefing pre-generation | No LLM latency on wake | Use BriefingProcessor |

### Memory Usage

| Component | Typical Usage |
|-----------|---------------|
| Wake word subprocess | ~200-400MB (ONNX models) |
| Vector cache (10k vectors) | ~120MB |
| Main process | ~100-200MB |
| Total | ~400-700MB |

### Concurrent Operations

| Phase | Parallel Tasks |
|-------|----------------|
| Wake word detection | Preconnect transcription, warm WebSocket, preload vector cache |
| TTS playback | Barge-in detection, termination detection, transcription preconnect |
| Response generation | Vector memory query (before context prep) |
| Conversation end | Recording end, memory update, termination stop (fire-and-forget) |

---

*Last updated: January 2026*

