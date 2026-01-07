# HomeAssist

Voice Controlled Personal AI infrastructure. Always listening. Low latency. Actually useful.

HomeAssist is a voice-first assistant tailored to my life, workflows, and data—modular, agentic, and built to feel like a real companion/mentor (not a reactive chatbot). It doesn’t just answer: it **proactively** surfaces what matters, delivers briefings, and nudges me at the right moment.

Say the wake word and speak naturally—HomeAssist handles lights, music, calendar, questions, and more. The first steps on the road to building a real-life Jarvis.

---

## What It Does

HomeAssist is a hands-free voice interface that:

- **Listens continuously** for a customizable wake word
- **Transcribes speech** in real-time with streaming ASR
- **Thinks with LLMs** to understand requests and generate responses
- **Speaks responses** via neural text-to-speech
- **Controls smart home devices** through an extensible tool system
- **Remembers you** across sessions with persistent and semantic memory

You can interrupt it mid-sentence (barge-in), ask multi-step questions, and have natural back-and-forth conversations.

---

## Capabilities

### Voice Interaction
| Feature | Description |
|---------|-------------|
| Wake word detection | Hands-free activation with custom trigger phrases |
| Real-time transcription | Low-latency streaming speech-to-text |
| Neural TTS | Natural-sounding responses (multiple voice options) |
| Barge-in | Interrupt the assistant by speaking |
| Send phrases | "Send it", "Sir" to submit your message |
| Auto-send | Automatic submission after silence timeout |

### Smart Home & Services
| Tool | What It Controls |
|------|------------------|
| **Lighting** | Kasa smart lights—on/off, brightness, color, scenes |
| **Spotify** | Music playback, search, queue management |
| **Calendar** | Google Calendar—view events, create appointments |
| **Weather** | Current conditions and forecasts |
| **Web Search** | Google search with AI-summarized results |
| **SMS** | Send text messages via macOS Messages |
| **Notifications** | Email summaries, news digests, custom alerts |

### Memory & Context
| System | Purpose |
|--------|---------|
| **Conversation context** | Maintains history within a session |
| **Persistent memory** | Remembers facts, preferences, and patterns about you |
| **Vector memory** | Semantic search over past conversations |
| **Briefing announcements** | Proactive updates spoken on wake word |

### Tool Chaining
Multi-step requests are handled automatically:

> "Find that song we talked about and text me the link"

The assistant recognizes this needs multiple tools (search → SMS), executes them in sequence, and confirms completion.

---

## Architecture

### Design Philosophy

1. **Modular providers** — Each component (transcription, TTS, wake word, LLM) is swappable via configuration
2. **State machine coordination** — Audio components are orchestrated through explicit state transitions to prevent conflicts
3. **Process isolation** — Wake word detection runs in a separate process to prevent model corruption
4. **Tool abstraction** — Smart home control via MCP (Model Context Protocol) for clean separation

### System Flow

```
┌─────────────────┐     Wake word      ┌─────────────────┐
│  Microphone     │───────────────────►│  Transcription  │
│  (always on)    │                    │  (streaming)    │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                ▼
┌─────────────────┐     Response       ┌─────────────────┐
│  TTS Speaker    │◄───────────────────│  LLM + Tools    │
│  (neural voice) │                    │  (orchestrator) │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │  MCP Server     │
                                       │  (tool actions) │
                                       └─────────────────┘
```

### Component Overview

| Component | Provider Options | Default |
|-----------|------------------|---------|
| Wake Word | OpenWakeWord | `hey_honey_v2` |
| Transcription | AssemblyAI, OpenAI Whisper | AssemblyAI |
| Response/LLM | OpenAI Realtime API | `gpt-4o-realtime-preview` |
| TTS | Piper, macOS, Google Cloud, Chatterbox | Piper |
| Tools | MCP Server | HTTP on `localhost:3000` |

### LLM Strategy

| Scenario | Model | Rationale |
|----------|-------|-----------|
| Direct conversation | `gpt-4o-realtime-preview` | Quality prose for thoughtful responses |
| Tool decisions | `gpt-4o-realtime-preview` | Accurate intent classification |
| Tool chaining | `gpt-4o-mini` | Cost-effective for "need more tools?" checks |
| Final answer (after tools) | `gpt-4o-mini` | Just composing results, premium model unnecessary |

---

## Project Structure

```
HomeAssist/
├── assistant_framework/       # Core voice assistant
│   ├── orchestrator_v2.py     # Main coordination logic
│   ├── config.py              # All configuration
│   ├── providers/             # Pluggable implementations
│   │   ├── transcription_v2/  # Speech-to-text
│   │   ├── response/          # LLM responses
│   │   ├── tts/               # Text-to-speech
│   │   ├── wakeword_v2/       # Wake word detection
│   │   └── context/           # Conversation history
│   ├── utils/                 # Shared utilities
│   │   ├── state_machine.py   # Audio lifecycle management
│   │   ├── barge_in.py        # Interrupt detection
│   │   ├── briefing_manager.py
│   │   └── briefing_processor.py
│   └── interfaces/            # Abstract base classes
│
├── mcp_server/                # Tool server
│   ├── server.py              # MCP entry point
│   ├── tools/                 # Tool implementations
│   │   ├── kasa_lighting.py
│   │   ├── spotify.py
│   │   ├── calendar.py
│   │   ├── weather.py
│   │   └── ...
│   └── tools_config.py        # Enable/disable tools
│
├── scripts/scheduled/         # Background jobs
│   ├── email_summarizer/      # Email digest pipeline
│   ├── news_summary/          # News summary pipeline
│   └── calendar_briefing/     # Calendar reminder announcements
│
├── audio_data/                # Model files
│   ├── wake_word_models/
│   └── piper_models/
│
└── state_management/          # Runtime state
    ├── persistent_memory.json
    └── conversation_summary.json
```

---

## Quick Start

### 1. Configure environment

```bash
cp env.example .env
# Edit .env with your API keys
```

Required keys:
- `OPENAI_API_KEY` — LLM responses
- `ASSEMBLYAI_API_KEY` — Transcription
- `SUPABASE_URL` + `SUPABASE_KEY` — Memory storage

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
python3 -m assistant_framework.main_v2 continuous
```

### 4. Interact

1. Say **"Hey Honey"** (or your configured wake word)
2. Speak your request
3. Say **"send it"** or wait for auto-send
4. Listen to the response (interrupt anytime by speaking)

---

## Configuration

All settings are in `assistant_framework/config.py`. Key sections:

| Section | Controls |
|---------|----------|
| Provider Selection | Which implementation for each component |
| Wake Word | Trigger phrases, sensitivity, multiple wake words |
| Transcription | ASR provider settings |
| Response/LLM | Model selection, temperature, system prompt |
| TTS | Voice selection, speed, chunking |
| Barge-In | Interrupt sensitivity |
| Memory | Persistent facts, vector search settings |
| Briefing Processor | Proactive announcement generation |

See [SETUP.md](SETUP.md) for detailed configuration reference.

---

## Extending

### Adding a new tool

1. Create `mcp_server/tools/your_tool.py` implementing the tool interface
2. Register in `mcp_server/tools_config.py`
3. The assistant automatically discovers and can use the tool

### Adding a new provider

1. Implement the appropriate interface in `assistant_framework/interfaces/`
2. Create provider class in `assistant_framework/providers/`
3. Register in `assistant_framework/factory.py`
4. Select via config

### Custom wake word

1. Train a model using [OpenWakeWord](https://github.com/dscripka/openWakeWord)
2. Place `.onnx` file in `audio_data/wake_word_models/`
3. Update `WAKEWORD_CONFIG` in config

---

## Requirements

- Python 3.10+
- macOS or Linux with audio support
- Microphone and speakers
- API keys for cloud services (OpenAI, AssemblyAI, etc.)

---

## License

MIT License — See LICENSE file for details.
