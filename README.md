# HomeAssist V2

HomeAssist is a fully local, always-listening voice assistant that brings your smart home to life. Just say the wake word and speak naturally—it transcribes your voice in real-time, thinks with GPT-4, and responds out loud. Control your lights, play music on Spotify, check your calendar, get the weather, and more, all hands-free. You can even interrupt it mid-sentence and it'll stop to listen. Version 2 was completely rebuilt to be rock-solid stable, fixing the crashes that plagued the original by using a smarter audio architecture that never fights with itself.

Key improvements:
- **No more segfaults** – Audio resources are properly managed with explicit ownership
- **Barge-in support** – Interrupt the assistant mid-response by speaking
- **Process isolation** – Wake word detection runs in a separate process to prevent model corruption
- **Clean state transitions** – A state machine ensures components don't conflict
- **Persistent memory** – Remembers facts, preferences, and patterns about you across sessions
- **Vector memory** – Semantic search of past conversations using 3072-dim embeddings

---

## Quick Start

### 1. Set up environment

Create a `.env` file in the project root:

```
ASSEMBLYAI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here
```

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the assistant

```bash
PYTHONFAULTHANDLER=1 python3 -m assistant_framework.main_v2 continuous
```

The `PYTHONFAULTHANDLER=1` flag helps diagnose any issues by printing a traceback on crashes.

### 4. Using the assistant

1. Say the wake word (**"Alexa"**) to activate
2. Speak your request
3. Say **"sir"** or **"send it"** when you're done speaking
4. The assistant will respond via text-to-speech
5. You can interrupt the response by speaking (barge-in)

---

## Project Structure

```
HomeAssistV2/
│
├── assistant_framework/          # Core assistant logic
│   ├── main_v2.py                # Entry point
│   ├── orchestrator_v2.py        # Coordinates all components
│   ├── config.py                 # All configuration settings
│   │
│   ├── providers/                # Pluggable component implementations
│   │   ├── transcription_v2/     # Speech-to-text (AssemblyAI)
│   │   ├── response/             # AI responses (OpenAI Realtime API)
│   │   ├── tts/                  # Text-to-speech (Google Cloud or local)
│   │   ├── wakeword_v2/          # Wake word detection (OpenWakeWord)
│   │   └── context/              # Conversation history management
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── state_machine.py      # Manages audio component lifecycle
│   │   ├── audio_manager.py      # Ensures only one component uses audio
│   │   ├── barge_in.py           # Detects when user interrupts
│   │   └── device_manager.py     # Audio device selection
│   │
│   ├── interfaces/               # Abstract base classes for providers
│   └── legacy/                   # Old v1 code (for reference)
│
├── mcp_server/                   # Tool server for smart home control
│   ├── server.py                 # MCP server entry point
│   ├── config.py                 # Tool settings (lights, users, etc.)
│   ├── tools_config.py           # Enable/disable individual tools
│   └── tools/                    # Tool implementations
│       ├── kasa_lighting.py      # Smart light control
│       ├── spotify_tool.py       # Music playback
│       ├── calendar_tool.py      # Google Calendar
│       ├── weather_tool.py       # Weather forecasts
│       └── ...
│
├── audio_data/                   # Wake word models
├── .env                          # API keys (not in repo)
└── requirements.txt              # Python dependencies
```

---

## Configuration

All settings live in `assistant_framework/config.py`, organized into sections:

| Section | What it controls |
|---------|------------------|
| Provider Selection | Which implementation to use for each component |
| Transcription | AssemblyAI settings (sample rate, etc.) |
| Response/LLM | OpenAI model, temperature, system prompt |
| TTS | Voice selection, speed, pitch |
| Wake Word | Detection threshold, cooldown timing |
| Barge-In | Interrupt sensitivity and buffering |
| Conversation Flow | Trigger phrases ("send it", "scratch that") |
| Persistent Memory | Facts, preferences, and patterns extraction |
| Vector Memory | Semantic search of past conversations (Supabase + pgvector) |

### Changing the TTS provider

In `config.py`, change:
```python
TTS_PROVIDER = "local_tts"  # Uses macOS 'say' command
# or
TTS_PROVIDER = "google_tts"  # Uses Google Cloud TTS (requires billing)
```

### Adjusting wake word sensitivity

In the `WAKEWORD_CONFIG` section:
```python
"threshold": 0.2,  # Lower = more sensitive, higher = fewer false positives
```

---

## MCP Tools

The assistant can control smart home devices and access services through MCP (Model Context Protocol) tools:

- **Lighting** – Turn lights on/off, adjust brightness, set scenes
- **Spotify** – Play music, control playback, search tracks
- **Calendar** – View and create Google Calendar events
- **Weather** – Get current conditions and forecasts
- **Notifications** – Read and manage notifications
- **Google Search** – Search the web with AI summaries

Tools can be enabled/disabled in `mcp_server/tools_config.py`.

---

## Memory Systems

HomeAssist has two complementary memory systems:

### Persistent Memory
Extracts and stores structured information:
- **User profile** – Name, location, preferences
- **Known facts** – Explicit user information ("my sister is a dog whisperer")
- **Patterns** – Behavioral observations with strength levels (weak → confirmed)

Stored in `state_management/persistent_memory.json`.

### Vector Memory
Semantic search over past conversations:
- **3072-dim embeddings** via OpenAI `text-embedding-3-large`
- **K-fold partitioning** – Long conversations stored as overlapping windows
- **Smart retrieval** – Returns top 1-2 results based on similarity gap
- **Context injection** – Relevant past conversations added to each prompt

Requires Supabase with pgvector extension. See SETUP.md for database setup.

---

## Troubleshooting

**Assistant doesn't respond to wake word**
- Check that your microphone is working and selected as the default input
- Try lowering the `threshold` value in wake word config

**Audio cuts out or sounds choppy**
- Increase `AUDIO_HANDOFF_DELAY` in config.py
- Check for other applications using the microphone

**"Missing API key" errors**
- Ensure your `.env` file exists and contains valid keys
- Keys should not have quotes around them

**Segmentation fault (should be rare in v2)**
- Run with `PYTHONFAULTHANDLER=1` to get a traceback
- Check that no other Python processes are using the audio device
