# HomeAssist V2

HomeAssist (codename **Sol**) is a fully local, always-listening voice assistant that brings your smart home to life. Just say the wake word and speak naturally—it transcribes your voice in real-time, thinks with GPT-4, and responds out loud. Control your lights, play music on Spotify, check your calendar, get the weather, and more, all hands-free. You can even interrupt it mid-sentence and it'll stop to listen. Version 2 was completely rebuilt to be rock-solid stable, fixing the crashes that plagued the original by using a smarter audio architecture that never fights with itself.

Key improvements:
- **No more segfaults** – Audio resources are properly managed with explicit ownership
- **Barge-in support** – Interrupt the assistant mid-response by speaking
- **Process isolation** – Wake word detection runs in a separate process to prevent model corruption
- **Clean state transitions** – A state machine ensures components don't conflict
- **Persistent memory** – Remembers facts, preferences, and patterns about you across sessions
- **Vector memory** – Semantic search of past conversations using 3072-dim embeddings
- **Briefing announcements** – Proactive updates spoken on wake word (e.g., "Hey Honey, what's new?")

---

## Quick Start

### 1. Set up environment

Create a `.env` file in the project root (recommended: start from `env.example`):

```
OPENAI_API_KEY=your_key_here
OPENAI_KEY=your_key_here  # optional alias used by some scripts
ASSEMBLYAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Supabase (required for vector memory, conversation recording, and notifications)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key

# Optional integrations
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here

# Optional scheduled summaries
NEWS_API_KEY_1=your-newsapi-key
EMAIL_NOTIFICATION_RECIPIENT=Morgan
NEWS_NOTIFICATION_RECIPIENT=Morgan
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

### 4. (Optional) Run the MCP server standalone

The assistant will start MCP automatically for `single` / `continuous` modes, but you can also run it manually:

```bash
./mcp_server/run.sh
```

By default it starts on `127.0.0.1:3000` (override with `HOST`, `PORT`, `TRANSPORT`).

### 5. Using the assistant

1. Say the wake word (**"Hey Honey"**) to activate
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
│   │   ├── transcription_v2/     # Speech-to-text (AssemblyAI or Whisper)
│   │   ├── response/             # AI responses (OpenAI Realtime API)
│   │   ├── tts/                  # Text-to-speech (Piper, Google, Chatterbox, local)
│   │   ├── wakeword_v2/          # Wake word detection (OpenWakeWord)
│   │   └── context/              # Conversation history management
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── state_machine.py      # Manages audio component lifecycle
│   │   ├── audio_manager.py      # Ensures only one component uses audio
│   │   ├── barge_in.py           # Detects when user interrupts
│   │   ├── device_manager.py     # Audio device selection
│   │   ├── briefing_manager.py   # Fetches/updates briefing announcements
│   │   └── briefing_processor.py # Pre-generates briefing openers via LLM
│   │
│   └── interfaces/               # Abstract base classes for providers
│
├── mcp_server/                   # Tool server for smart home control
│   ├── server.py                 # MCP server entry point
│   ├── config.py                 # Tool settings (lights, users, etc.)
│   ├── tools_config.py           # Enable/disable individual tools
│   └── tools/                    # Tool implementations
│       ├── kasa_lighting.py      # Smart light control
│       ├── spotify.py            # Music playback
│       ├── calendar.py           # Google Calendar
│       ├── weather.py            # Weather forecasts
│       ├── notifications.py      # Notification management
│       ├── google_search.py      # Web search with AI summaries
│       ├── sms.py                # Send text messages (macOS)
│       ├── state_tool.py         # System state management
│       └── system_info.py        # Assistant self-documentation
│
├── audio_data/                   # Model files
│   ├── wake_word_models/         # Wake word detection models
│   ├── piper_models/             # Piper TTS voice models
│   └── chatterbox_models/        # Chatterbox TTS models
│
├── state_management/             # Runtime state files
├── .env                          # API keys (not in repo)
└── requirements.txt              # Python dependencies
```

---

## Configuration

All settings live in `assistant_framework/config.py`, organized into sections:

| Section | What it controls |
|---------|------------------|
| Provider Selection | Which implementation to use for each component |
| Transcription | AssemblyAI or OpenAI Whisper settings |
| Response/LLM | OpenAI model, temperature, system prompt |
| TTS | Voice selection, speed, pitch (4 providers) |
| Wake Word | Detection threshold, cooldown, multiple wake words |
| Briefing Processor | Opener generation model, tokens, prompt |
| Barge-In | Interrupt sensitivity and buffering |
| Conversation Flow | Trigger phrases ("send it", "scratch that") |
| Persistent Memory | Facts, preferences, and patterns extraction |
| Vector Memory | Semantic search of past conversations (Supabase + pgvector) |
| Presets | Dev/prod/test configuration profiles |

### Changing the TTS provider

In `config.py`, change:
```python
TTS_PROVIDER = "piper"       # Fast local neural TTS (default)
# or
TTS_PROVIDER = "local_tts"   # Uses macOS 'say' command
# or
TTS_PROVIDER = "google_tts"  # Google Cloud TTS (requires billing)
# or
TTS_PROVIDER = "chatterbox"  # Resemble AI's local neural TTS with voice cloning
```

### Adjusting wake word sensitivity

In the `WAKEWORD_CONFIG` section:
```python
"threshold": 0.2,  # Lower = more sensitive, higher = fewer false positives
```

---

## MCP Tools

The assistant can control smart home devices and access services through MCP (Model Context Protocol) tools:

- **Lighting** – Turn lights on/off, adjust brightness, set scenes (Kasa devices)
- **Spotify** – Play music, control playback, search tracks
- **Calendar** – View and create Google Calendar events
- **Weather** – Get current conditions and forecasts
- **Notifications** – Read and manage notifications
- **Google Search** – Search the web with AI summaries
- **SMS** – Send text messages via macOS Messages (macOS only)
- **State** – Manage system state (active user, lighting scenes, volume)
- **System Info** – Explain the assistant's own architecture and capabilities

Tools can be enabled/disabled in `mcp_server/tools_config.py`.

### Notifications (email + news)

- **Storage**: Scheduled email/news pipelines write to Supabase `notification_sources`.
- **Context policy**: The assistant only pulls the **most recent** email batch and/or news summary into context. Older rows remain as historical data.
- **Read status**: When a notification is retrieved (sent to the LLM as context), its `read_status` is set to `read`.
- **Seen messaging**: Each returned notification includes a `previously_seen` flag so the LLM can say whether you've already been told about it.

### Briefing Announcements

Proactive briefings are spoken to the user when they trigger a designated wake word:

- **Storage**: Briefings are stored in Supabase `briefing_announcements` table.
- **Pre-generated openers**: `BriefingProcessor` generates natural conversation openers via LLM ahead of time, so wake word activation only requires TTS (no LLM latency).
- **Selective triggering**: Configure which wake words trigger briefings in `WAKEWORD_CONFIG`:
  ```python
  "model_names": ["hey_honey_v2", "hey_honey_whats_new"],
  "briefing_wake_words": ["hey_honey_whats_new"],  # Only this triggers briefings
  ```
- Say "Hey Honey" for quick commands, or "Hey Honey, what's new?" to hear pending announcements first.

---

## Model Architecture

HomeAssist uses a two-model approach optimized for cost and quality:

| Scenario | Model | Why |
|----------|-------|-----|
| **Direct conversation** | `gpt-4o-realtime-preview` | Higher quality prose for thoughtful responses |
| **Initial tool decision** | `gpt-4o-realtime-preview` | Decides if tools are needed |
| **Tool chaining** | `gpt-4o-mini` | Checks if additional tools are needed after each execution |
| **Final answer (after tools)** | `gpt-4o-mini` | Composes response from tool results |

When you ask a conversational question ("What do you think about X?"), you get the premium model directly. When tools are involved, the cheaper model handles composition since it's just reporting results.

---

## Tool Chaining

Multi-step requests are handled automatically. When you say something like:

> "Find me that song and text me the link"

The assistant:
1. Recognizes this requires **two tools** (search + SMS)
2. Executes the first tool (Google search)
3. Checks if the request is fully satisfied (it's not—no text sent yet)
4. Calls the second tool (SMS) with the search result
5. Composes a final response confirming both actions

This loop continues until all parts of your request are fulfilled.

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
- Increase `state_transition_delay` in `TURNAROUND_CONFIG`
- Check for other applications using the microphone

**"Missing API key" errors**
- Ensure your `.env` file exists and contains valid keys
- Keys should not have quotes around them

**Segmentation fault (should be rare in v2)**
- Run with `PYTHONFAULTHANDLER=1` to get a traceback
- Check that no other Python processes are using the audio device
