# HomeAssist Setup Guide

A modular voice assistant framework with wake word detection, real-time transcription, LLM-powered responses, and smart home integration.

---

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT responses and realtime voice processing |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key for real-time speech transcription |

### Optional - AI Services

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (used for conversation summarization and memory extraction if provider is set to gemini) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud service account JSON (for Google TTS). Auto-detected if placed in `assistant_framework/google_creds/` |

### Optional - Smart Home & Integrations

| Variable | Description |
|----------|-------------|
| `SPOTIFY_CLIENT_ID` | Spotify OAuth client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify OAuth client secret |

### Optional - Data Persistence

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL (for conversation recording & dashboard sync) |
| `SUPABASE_KEY` | Supabase service role key |
| `CONSOLE_TOKEN` | Unique identifier for dashboard log isolation (see Dashboard section) |

### Optional - Email Summarizer (Scheduled Scripts)

| Variable | Description |
|----------|-------------|
| `GMAIL_CLIENT_ID` / `GOOGLE_CLIENT_ID` | Google OAuth client ID for Gmail access |
| `GMAIL_CLIENT_SECRET` / `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `GMAIL_REFRESH_TOKEN` / `GOOGLE_REFRESH_TOKEN` | OAuth refresh token for unattended access |
| `EMAIL_NOTIFICATION_RECIPIENT` | Name for email notifications (default: "Morgan") |
| `EMAIL_SUMMARIZER_HEADLESS` | Set to `1` for headless OAuth flow |

### Optional - Runtime

| Variable | Description |
|----------|-------------|
| `WAKEWORD_MODEL_DIR` | Custom path to wake word models (default: `./audio_data/wake_word_models`) |
| `DEFAULT_TIME_ZONE` | IANA timezone for calendar operations (default: `America/New_York`) |
| `LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `QUIET_IMPORT` | Set to suppress config summary on import |

---

## Configuration Overview

All runtime configuration lives in `assistant_framework/config.py`, organized into sections:

### Provider Selection (Section 2)

```python
TRANSCRIPTION_PROVIDER = "assemblyai"      # Speech-to-text
RESPONSE_PROVIDER = "openai_websocket"     # LLM backend
TTS_PROVIDER = "local_tts"                 # "google_tts" or "local_tts"
CONTEXT_PROVIDER = "unified"               # Conversation context management
WAKEWORD_PROVIDER = "openwakeword"         # Wake word detection
```

### System Prompt (Section 4)

`SYSTEM_PROMPT` defines the assistant's personality, capabilities, and behavioral guidelines. Edit this to customize how your assistant responds.

### Wake Word (Section 6)

```python
WAKEWORD_CONFIG = {
    "model_name": "hey_honey",    # Available: alexa_v0.1, hey_jarvis_v0.1, hey_honey
    "threshold": 0.2,             # Detection sensitivity (lower = more sensitive)
    "cooldown_seconds": 2.0,      # Minimum time between activations
}
```

Custom wake word models (`.onnx` format) can be added to `audio_data/wake_word_models/`.

### Conversation Memory (Section 7)

**Summarization**: Automatically summarizes long conversations to maintain context without exceeding token limits.

**Persistent Memory**: Extracts and stores lasting facts about the user across sessions (name, preferences, patterns). Configurable backend:

```python
PERSISTENT_MEMORY_CONFIG = {
    "provider": "openai",           # "openai" or "gemini"
    "openai_model": "gpt-5-nano",   # Model for memory extraction
}
```

### Conversation Flow (Section 9)

```python
TERMINATION_PHRASES = ["over out", "stop listening"]  # End session
SEND_PHRASES = ["send it", "sir"]                     # Send message to LLM
AUTO_SEND_SILENCE_TIMEOUT = 6.0                       # Auto-send after N seconds silence
```

### Barge-In Detection (Section 11)

Allows interrupting the assistant mid-speech:

```python
BARGE_IN_CONFIG = {
    "energy_threshold": 0.055,        # Voice detection sensitivity
    "early_barge_in_threshold": 3.0,  # Seconds - early interruptions append to previous message
    "min_speech_duration": 0.2,       # Required speech duration to trigger
}
```

### Latency Tuning (Section 11B)

```python
TURNAROUND_CONFIG = {
    "state_transition_delay": 0.05,   # Component switch delay
    "barge_in_resume_delay": 0.05,    # Resume delay after interruption
}
```

---

## MCP Server (Tool Integration)

The MCP (Model Context Protocol) server provides tool capabilities to the assistant. Configuration lives in `mcp_server/config.py`:

### Smart Lighting (Kasa)

```python
LIGHT_IPS = {
    "morgans_led": "192.168.1.49",
    "living_room_lamp": "192.168.1.165",
}
```

### Spotify Users

```python
SPOTIFY_USERS = {
    "Morgan": {"username": "..."},
}
```

### Google Calendar

Calendar credentials are stored in `mcp_server/google_creds/`:
- `google_creds_<user>.json` - OAuth client secrets
- `token_<user>.json` - Access tokens (generated on first auth)

---

## Dashboard Communication

HomeAssist integrates with a remote dashboard for real-time monitoring via REST API endpoints:

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/console/log` | POST | Add a console log entry |
| `/api/console/log` | GET | Retrieve logs (query: `token`, `since`) |
| `/api/console/log` | DELETE | Clear logs for a token |
| `/api/sessions` | POST | Create conversation session |
| `/api/sessions` | GET | List recent sessions (limit 100) |
| `/api/sessions/[id]/end` | POST | End a session |
| `/api/messages` | POST | Add message to session |
| `/api/tool-calls` | POST | Record tool execution |
| `/api/webhook` | POST/GET | Generic webhook handler |

### Console Token

The `CONSOLE_TOKEN` environment variable isolates data per user/instance:

- **Console Logs**: Namespaces logs so multiple users don't see each other's data
- **Database Records**: Stored in session/message metadata for realtime filtering
- **Dashboard Filtering**: Enables the dashboard to show only relevant sessions

This allows multiple devices or users to share the same dashboard infrastructure without collision.

### Data Flow

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  HomeAssist     │◄──────────────────►│  AssemblyAI     │
│  (Voice Loop)   │                    │  (Transcription)│
└────────┬────────┘                    └─────────────────┘
         │
         │ REST API
         ▼
┌─────────────────┐     Realtime       ┌─────────────────┐
│  Supabase       │◄──────────────────►│  Dashboard      │
│  (PostgreSQL)   │                    │  (Next.js)      │
└─────────────────┘                    └─────────────────┘
```

---

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the assistant**
   ```bash
   # Continuous conversation mode (wake word → transcribe → respond → repeat)
   python -m assistant_framework.main_v2 continuous
   
   # Single interaction mode
   python -m assistant_framework.main_v2 single
   ```

4. **Validate configuration**
   ```python
   from assistant_framework.config import validate_environment, print_config_summary
   print_config_summary()
   ```

---

## State Files

| File | Purpose |
|------|---------|
| `state_management/app_state.json` | Runtime application state |
| `state_management/conversation_summary.json` | Current session summary |
| `state_management/persistent_memory.json` | Long-term user memory |
| `state_management/session_summary.json` | Session metadata |

---

## Troubleshooting

### Audio Device Issues
- Set `input_device_index` in `WAKEWORD_CONFIG` to specify a microphone
- Increase `AUDIO_HANDOFF_DELAY` if experiencing device conflicts

### Wake Word Not Detecting
- Lower `threshold` in `WAKEWORD_CONFIG` (more sensitive)
- Ensure microphone is not muted and correct device is selected

### Barge-In Too Sensitive / Not Sensitive Enough
- Adjust `energy_threshold` in `BARGE_IN_CONFIG`
- Higher values = less sensitive, lower values = more sensitive

### Memory Not Persisting
- Check `persistent_memory.json` is writable
- Verify API key for chosen provider (OpenAI or Gemini)
