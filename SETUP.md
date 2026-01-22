# HomeAssist Setup Guide

A modular voice assistant framework featuring wake word detection, real-time transcription, LLM-powered responses, persistent memory, and smart home integration.

**V3 improvements:** Dramatically faster response times and proactive jump-in briefings that speak immediately on wake word detection.

---

## 📋 Overview

HomeAssist is a privacy-focused voice assistant that runs locally on your machine. It connects to cloud APIs for transcription and AI responses while keeping your conversation data under your control.

### ✨ Key Features

- 🎤 **Wake Word Detection** — Hands-free activation with customizable trigger phrases
- 🗣️ **Real-time Transcription** — Low-latency speech-to-text via AssemblyAI
- 🤖 **LLM Responses** — Natural conversation powered by OpenAI's Realtime API
- 🧠 **Persistent Memory** — Remembers user preferences and facts across sessions
- 🏠 **Smart Home Control** — Lights, music, calendar, and more via MCP tools
- ⚡ **Barge-In Support** — Interrupt the assistant mid-speech naturally
- 🔔 **Audio Feedback** — Distinct sounds for tool success/failure and system events

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- macOS (for local TTS) or Linux with audio support
- Microphone and speakers

### Step 1: Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/HomeAssistV3.git
cd HomeAssistV3
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys (see Configuration section below).

### Step 5: Run the Assistant

```bash
# Continuous conversation mode (recommended)
python -m assistant_framework.main continuous

# Single interaction mode
python -m assistant_framework.main single
```

> 💡 **Tip:** On first run, a configuration summary will print showing which components are active and any missing credentials.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with your API keys.

#### Required Variables

- `OPENAI_API_KEY` — OpenAI API key (LLM responses, OpenAI TTS if enabled, some scheduled jobs)
- `ASSEMBLYAI_API_KEY` — AssemblyAI key (real-time transcription)
- `GEMINI_API_KEY` — Gemini API key (email summarization + optional conversation summarization)
- `SUPABASE_URL` — Supabase project URL (vector memory, conversation recording, notifications)
- `SUPABASE_KEY` — Supabase service role key (server-side access for writes)

#### Optional — AI Services

- `OPENAI_KEY` — Optional alias used by some scripts (fallback to `OPENAI_API_KEY`)
- `GEMINI_MODEL` — Gemini model name for summarizers (default: `gemini-2.5-flash`)
- `BRIEFING_PROCESSOR_MODEL` — Model for briefing opener generation (default: `gpt-4o-mini`)
- `GOOGLE_APPLICATION_CREDENTIALS` — Path to Google Cloud service account JSON (Google Cloud TTS, if enabled)

#### Optional — Integrations

- `SPOTIFY_CLIENT_ID` — Spotify OAuth client ID
- `SPOTIFY_CLIENT_SECRET` — Spotify OAuth client secret
- `CONSOLE_TOKEN` — Dashboard log isolation token

#### Optional — Notifications / Scheduled Summaries

- `EMAIL_NOTIFICATION_RECIPIENT` — Target user for email notifications (default: `Morgan`)
- `NEWS_NOTIFICATION_RECIPIENT` — Target user for news summaries (default: `Morgan`)
- `NEWS_API_KEY_1` — NewsAPI.org key (used by the news summarizer; supports rotation with `_2`…`_5`)
- `NEWS_API_KEY_2` — Additional NewsAPI key (optional)
- `NEWS_API_KEY_3` — Additional NewsAPI key (optional)
- `NEWS_API_KEY_4` — Additional NewsAPI key (optional)
- `NEWS_API_KEY_5` — Additional NewsAPI key (optional)
- `REMINDER_LOOKAHEAD_DAYS` — Days ahead to analyze calendar events for reminders (default: `7`)
- `REMINDER_CALENDAR_USERS` — Comma-separated list of calendar users to analyze (default: `morgan_personal`)

#### Optional — Google Calendar (CI/GitHub Actions)

For running calendar briefing in CI environments, credentials can be provided via environment variables:

- `GOOGLE_CREDENTIALS_JSON` — Base64-encoded OAuth client credentials JSON
- `GOOGLE_CALENDAR_TOKEN_JSON` — Base64-encoded OAuth token JSON (with calendar scopes)
- `GOOGLE_TOKEN_JSON` — Fallback token (may have Gmail scopes only)

> 💡 **Tip:** The calendar client auto-detects base64-encoded vs raw JSON secrets.

#### Optional — Gmail OAuth (Email Summarizer)

- `GMAIL_CLIENT_ID` — Gmail OAuth client id (CI-friendly auth; no files needed)
- `GMAIL_CLIENT_SECRET` — Gmail OAuth client secret
- `GMAIL_REFRESH_TOKEN` — Gmail OAuth refresh token
- `GOOGLE_CLIENT_ID` — Alias for `GMAIL_CLIENT_ID` (optional)
- `GOOGLE_CLIENT_SECRET` — Alias for `GMAIL_CLIENT_SECRET` (optional)
- `GOOGLE_REFRESH_TOKEN` — Alias for `GMAIL_REFRESH_TOKEN` (optional)

> ⚠️ **Important:** Never commit your `.env` file to version control!

---

### Config File Reference

All runtime settings live in `assistant_framework/config.py`, organized by feature:

#### Provider Selection

Choose which implementation to use for each component:

```python
TRANSCRIPTION_PROVIDER = "assemblyai"      # "assemblyai" or "openai_whisper"
RESPONSE_PROVIDER = "openai_websocket"     # LLM backend
TTS_PROVIDER = "piper"                     # "google_tts", "local_tts", "piper", "chatterbox", "openai_tts"
WAKEWORD_PROVIDER = "openwakeword"         # Wake word engine
```

#### Transcription Providers

**AssemblyAI** (default) — Real-time streaming via WebSocket:

```python
ASSEMBLYAI_CONFIG = {
    "api_key": os.getenv("ASSEMBLYAI_API_KEY"),
    "sample_rate": 16000,
    "format_turns": True,
}
```

**OpenAI Whisper** — Chunked transcription via Whisper API:

```python
OPENAI_WHISPER_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "whisper-1",
    "chunk_duration": 3.0,        # Seconds per API call
    "language": "en",             # Supports many languages
    "silence_threshold": 0.01,    # Silence detection sensitivity
    "silence_duration": 1.5,      # Silence to trigger final result
}
```

> 💡 **Tip:** AssemblyAI has lower latency (true streaming). Whisper is useful if you already have OpenAI credentials and don't want another API key.

#### Text-to-Speech Providers

**Piper TTS** (recommended) — Fast local neural TTS using ONNX:

```python
PIPER_TTS_CONFIG = {
    "voice": "en_US-lessac-high",         # Voice model name
    "model_dir": "./audio_data/piper_models",  # Auto-downloads models
    "speed": 1.2,                            # Speech rate (0.5-2.0)
    "chunked_synthesis_threshold": 150,      # Enable chunked synthesis for text > N chars
    "chunk_max_length": 150,                 # Max chars per chunk
}
```

Available voices: `en_US-lessac-high`, `en_US-lessac-medium`, `en_US-ryan-medium`, `en_US-amy-medium`, `en_GB-alan-medium`, `en_GB-jenny_dioco-medium`

> 💡 **Benefits:** Very fast (~50x realtime), small models (15-100MB), runs entirely on CPU, no API costs.

**OpenAI TTS** — Cloud-based with true streaming for lowest latency:

```python
OPENAI_TTS_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),  # Required
    "model": "gpt-4o-mini-tts",               # "tts-1" (fast), "tts-1-hd" (high quality), or "gpt-4o-mini-tts" (latest)
    "voice": "alloy",                         # alloy, echo, fable, onyx, nova, shimmer
    "speed": 1.0,                             # Speed modifier (0.25-4.0)
    "response_format": "mp3",                 # mp3, opus, aac, flac, wav, pcm
    "stream_chunk_size": 4096,                # Bytes per streaming chunk
}
```

Available models: `tts-1` (fast, low latency), `tts-1-hd` (higher quality), `gpt-4o-mini-tts` (latest multimodal model)

Available voices: `alloy` (neutral), `echo` (male), `fable` (expressive), `onyx` (deep male), `nova` (female), `shimmer` (soft female)

> 💡 **Streaming:** Audio playback begins as soon as the first chunks arrive from OpenAI, dramatically reducing perceived latency. Requires `ffplay` (from ffmpeg) for true streaming; falls back to buffered playback if unavailable.

> ⚠️ **Note:** OpenAI TTS does not support pitch adjustment. Requires API key and internet connection.

**Google Cloud TTS** — Premium cloud TTS with HD voices:

```python
GOOGLE_TTS_CONFIG = {
    "voice": "en-US-Chirp3-HD-Sadachbia",
    "speed": 1.9,
    "pitch": -2.1,                           # Pitch adjustment in semitones
    "language_code": "en-US",
    "audio_encoding": "MP3"
}
```

> 💡 **Requires:** `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to service account JSON.

**Local TTS** — macOS native TTS:

```python
LOCAL_TTS_CONFIG = {
    "voice_id": 132,    # Samantha (US English female)
    "rate": 199,        # Words per minute
    "volume": 0.9       # Volume 0.0 to 1.0
}
```

> 💡 **Note:** macOS only, uses the built-in `say` command.

**Chatterbox TTS** — Local neural TTS with paralinguistic features:

```python
CHATTERBOX_TTS_CONFIG = {
    "model_type": "turbo",          # "turbo" (fast) or "standard" (creative)
    "model_dir": "./audio_data/chatterbox_models",
    "device": "auto",               # "auto", "mps" (Apple Silicon), "cuda", "cpu"
    "voice_prompt_path": None,      # Optional voice cloning
}
```

Supports paralinguistic tags: `[chuckle]`, `[laugh]`, `[sigh]`, `[cough]`, `[sniffle]`, `[groan]`, `[yawn]`, `[gasp]`

> 💡 **Install:** `pip install chatterbox-tts torchaudio` and run `huggingface-cli login` on first use.

#### Wake Word Settings

```python
WAKEWORD_CONFIG = {
    "model_name": "hey_honey_v2",         # Primary wake word (backward compat)
    "model_names": ["hey_honey_v2"],      # Multiple wake words (add more here)
    "briefing_wake_words": [],            # Wake words that trigger briefing announcements
    "threshold": 0.2,                     # Sensitivity (lower = more sensitive)
    "cooldown_seconds": 2.0,              # Min time between activations
}
```

> 💡 **Tip:** Custom wake word models (`.onnx`) go in `audio_data/wake_word_models/`

**Multiple Wake Words Example:**

```python
WAKEWORD_CONFIG = {
    "model_names": ["hey_honey_v2", "hey_honey_whats_new"],
    "briefing_wake_words": ["hey_honey_whats_new"],  # Only this triggers briefings
    # ...
}
```

This lets you say "Hey Honey" for quick commands, or "Hey Honey, what's new?" to hear pending briefing announcements first.

#### Conversation Flow

```python
TERMINATION_PHRASES = ["over out", "stop listening"]
SEND_PHRASES = ["send it", "sir"]
AUTO_SEND_SILENCE_TIMEOUT = 6.0  # Auto-send after 6s of silence
AUTO_SEND_SILENCE_TIMEOUT_DURING_TOOLS = 12.0  # Extended timeout after tool calls
```

> 💡 **Tip:** After the assistant executes tool calls (calendar, weather, etc.), the silence timeout automatically increases to give you more time to think about the results before your next question.

#### Parallel Termination Detection

Enables instant conversation termination by detecting phrases like "over out" in parallel with other operations. Unlike text-based termination phrases (above) that require transcription, this uses a dedicated wake word model that can detect termination phrases even while the assistant is speaking.

```python
TERMINATION_DETECTION_CONFIG = {
    "enabled": True,                   # Enable/disable feature
    "model_name": "over_out",          # Custom trained OpenWakeWord model
    "threshold": 0.5,                  # Detection sensitivity (higher = fewer false positives)
    "cooldown_seconds": 1.0,           # Min time between detections
    "verbose": False,                  # Debug output
}
```

**Key benefits:**

- ⚡ **Instant response** — Wake word model detects in ~100-200ms vs waiting for transcription
- 🗣️ **Works during TTS** — Say "over out" while assistant is speaking to immediately stop
- 🔄 **Parallel execution** — Runs alongside transcription, response generation, and TTS
- 🛡️ **Process isolated** — Crashes don't affect main application

**Training a custom model:**

The termination detection requires a trained OpenWakeWord model. Train one using [OpenWakeWord's training guide](https://github.com/dscripka/openWakeWord#training-new-models) with phrases like:

- "over out"
- "over and out"  
- "stop listening"
- "end session"

Place the trained model at: `audio_data/wake_word_models/over_out.onnx`

> 💡 **Tip:** If the model doesn't exist, termination detection gracefully degrades to text-based detection only.

#### Barge-In Detection

Allows interrupting the assistant while it's speaking:

```python
BARGE_IN_CONFIG = {
    "energy_threshold": 0.055,        # Voice detection sensitivity
    "early_barge_in_threshold": 3.0,  # Early interrupts append to previous message
    "min_speech_duration": 0.2,       # Required speech to trigger
}
```

#### Conversation Summarization

The assistant automatically summarizes conversations using Gemini for efficient context management:

```python
"summarization": {
    "enabled": True,
    "first_summary_at": 8,       # Trigger first summary at this many messages
    "summarize_every": 4,         # Re-summarize every N messages after first
    "gemini_model": "gemini-2.0-flash",
}
```

**How it works:**
- Summaries are generated in the background without blocking conversation
- Each conversation session starts fresh (no carryover from previous sessions)
- Summaries are stored in `state_management/conversation_summary.json`
- Used to feed context to persistent memory extraction and vector memory

> 💡 **Tip:** Requires `GEMINI_API_KEY` environment variable. If not set, summarization is skipped.

#### Persistent Memory

The assistant remembers facts about you across sessions:

```python
PERSISTENT_MEMORY_CONFIG = {
    "enabled": True,
    "provider": "openai",           # "openai" or "gemini"
    "openai_model": "gpt-5-nano",   # Extraction model
}
```

Memory is stored in `state_management/persistent_memory.json` and includes:

- ✅ User profile (name, location)
- ✅ Known facts ("prefers warm lighting")
- ✅ Behavioral patterns with strength levels

#### Vector Memory (Semantic Search)

Long-term semantic memory that stores conversation summaries as high-dimensional embeddings:

```python
VECTOR_MEMORY_CONFIG = {
    "enabled": True,
    "embedding_model": "text-embedding-3-large",  # 3072 dimensions
    "embedding_dimensions": 3072,
    "retrieve_top_k": 3,           # Initial retrieval count
    "relevance_threshold": 0.0,    # No minimum (smart filtering instead)
    "max_age_days": 90,            # Ignore old memories
}
```

**Features:**

- 🔍 **Semantic search** — Find past conversations by meaning, not keywords
- 📐 **3072-dim embeddings** — Higher quality than small model (64.6% vs 62.3% MTEB)
- 🪟 **K-fold partitioning** — Long conversations (>4 turns) stored as overlapping windows
- 🎯 **Smart retrieval** — Returns top 2 if similar scores, otherwise just top 1 (6% gap threshold)
- 📤 **Context injection** — Relevant memories added as system message before each response

**Supabase Setup:**

1. Enable pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Create the memories table (3072 dimensions, no index for simplicity):
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

> 💡 **Note:** No index is needed for < 10k vectors.

**Notification Sources Table (Email & News Summaries):**

Scheduled email and news summary processes store their outputs to Supabase for persistent storage:

```sql
CREATE TABLE notification_sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,           -- 'email', 'news', etc.
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    read_status TEXT DEFAULT 'unread',   -- 'unread', 'read', 'dismissed'
    priority TEXT DEFAULT 'normal',      -- 'high', 'normal', 'low'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    source_generated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    batch_id TEXT
);

-- Indexes for common queries
CREATE INDEX idx_notifications_user_type ON notification_sources(user_id, source_type);
CREATE INDEX idx_notifications_user_unread ON notification_sources(user_id) WHERE read_status = 'unread';
CREATE INDEX idx_notifications_created ON notification_sources(created_at DESC);
CREATE INDEX idx_notifications_metadata ON notification_sources USING gin (metadata);
CREATE INDEX idx_notifications_batch ON notification_sources(batch_id);
```

- `EMAIL_NOTIFICATION_RECIPIENT` — Target user for email notifications (default: primary user)
- `NEWS_NOTIFICATION_RECIPIENT` — Target user for news summaries (default: primary user)

💡 **Notification context policy**
- The assistant only loads the **most recent** email batch and/or the **most recent** news summary into conversation context.
- Older rows remain in Supabase as historical data but are not re-injected into context.

💡 **Read semantics**
- When a notification is retrieved (sent to the LLM as context), its `read_status` is updated to `read`.
- Returned notifications include `previously_seen` so the LLM can say whether you've already been told about it.

**Briefing Announcements Table (Wake-Word Briefings):**

Briefing announcements are reported proactively to the user when they trigger the wake word. Briefings persist until explicitly dismissed.

```sql
CREATE TABLE briefing_announcements (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content JSONB NOT NULL,            -- { message: str, llm_instructions?: str, meta?: {...} }
    opener_text TEXT,                  -- Pre-generated conversation opener (via BriefingProcessor)
    priority TEXT DEFAULT 'normal',    -- 'high', 'normal', 'low'
    status TEXT DEFAULT 'pending',     -- 'pending', 'delivered', 'dismissed', 'skipped'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    dismissed_at TIMESTAMPTZ
);

CREATE INDEX idx_briefings_user_status ON briefing_announcements(user_id, status);
CREATE INDEX idx_briefings_created ON briefing_announcements(created_at DESC);

-- To add opener_text to existing table:
-- ALTER TABLE briefing_announcements ADD COLUMN opener_text TEXT;
```

Content structure:
```json
{
  "message": "Your package was delivered at 2pm",
  "llm_instructions": "Mention this casually at the start of conversation",
  "meta": {
    "timestamp": "2026-01-03T14:00:00Z",
    "source": "delivery_tracker",
    "event_datetime_iso": "2026-01-03T14:00:00Z"
  }
}
```

💡 **Wake-word briefing behavior**
- Briefings can be inserted via Supabase dashboard or programmatically by any input source.
- The `BriefingProcessor` utility (`assistant_framework/utils/briefing_processor.py`) pre-generates `opener_text` via LLM.
- On wake word, the assistant fetches briefings with `opener_text` and speaks via TTS only (no LLM latency).
- If briefings don't have openers yet, falls back to LLM generation at wake time.
- After speaking, briefings are marked `delivered` (remain until explicitly `dismissed`).
- If `event_datetime_iso` is in the past when fetched, the briefing is automatically marked `skipped` (event already happened).

💡 **Multiple Wake Words for Selective Briefings**

You can configure different wake words to control whether briefings are announced:

```python
# In assistant_framework/config.py
WAKEWORD_CONFIG = {
    "model_names": ["hey_honey_v2", "hey_honey_whats_new"],  # Load both models
    "briefing_wake_words": ["hey_honey_whats_new"],  # Only this triggers briefings
    # ...
}
```

| Wake Word | Briefings | Use Case |
|-----------|-----------|----------|
| `hey_honey_v2` | ❌ Skipped | Quick questions, commands |
| `hey_honey_whats_new` | ✅ Announced | "What's new?" - get updates |

💡 **BriefingProcessor Configuration**

The opener generation is configured in `config.py`:

```python
BRIEFING_PROCESSOR_CONFIG = {
    "model": "gpt-4o-mini",           # Fast, cheap model for opener generation
    "max_tokens_single": 150,         # Max tokens for single briefing
    "max_tokens_combined": 200,       # Max tokens for multiple briefings
    "temperature": 0.7,               # Creativity level (0.0-1.0)
    "system_prompt": "..."            # Customize the opener generation prompt
}
```

- Override model via `BRIEFING_PROCESSOR_MODEL` environment variable
- System prompt controls tone and style of generated openers

💡 **Pre-generating openers (recommended)**

Briefing sources should call `BriefingProcessor` after inserting briefings:
```python
from assistant_framework.utils.briefing_processor import BriefingProcessor
from assistant_framework.utils.briefing_manager import BriefingManager

processor = BriefingProcessor()
manager = BriefingManager()

# Process all pending briefings without openers
await processor.process_pending_briefings(user="Morgan", briefing_manager=manager)
```

**Calendar Event Cache Table (Deduplication):**

The calendar briefing system tracks processed events to avoid duplicate announcements:

```sql
CREATE TABLE IF NOT EXISTS calendar_event_cache (
    event_id TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calendar_event_cache_first_seen 
ON calendar_event_cache(first_seen);
```

- Events are cached for 30 days, then auto-pruned
- Cache uses Supabase when available, falls back to local file
- Prevents re-announcing the same calendar events on each run

**Local Caching (Fast Search):**

Vector memory includes an in-memory cache for fast local search:

```python
VECTOR_MEMORY_CONFIG = {
    # ... other settings ...
    "local_cache_enabled": True,      # Use numpy cache for fast search
    "max_cached_vectors": 10000,      # Max vectors in memory (~120MB)
    "sync_interval_seconds": 300,     # Background sync interval
    "preload_on_startup": True,       # Load all vectors on startup
}
```

**Performance comparison:**

| Method | Latency | Notes |
|--------|---------|-------|
| Supabase (remote) | ~50-200ms | Network round-trip |
| Local cache (numpy) | ~1-5ms | In-memory cosine similarity |

**How it works:**
1. On startup, all vectors are loaded from Supabase into memory
2. Searches use numpy matrix multiplication (very fast)
3. New vectors are written to both cache and Supabase
4. Cache stats available via `vector_memory.get_cache_stats()`

**Console output:**
```
📦 Vector cache loaded: 150 vectors (1.8MB) in 0.45s
⚡ Cache search: 3 results in 0.8ms
```

---

## 🏠 Smart Home Setup

### MCP Server

The MCP (Model Context Protocol) server provides tool capabilities. Configuration is in `mcp_server/config.py`.

💡 **Running the MCP server**
- Use `./mcp_server/run.sh` to create/activate a venv and install `mcp_server/requirements.txt`.
- Defaults can be overridden via environment variables:
  - `HOST` — Bind host (default: `127.0.0.1`)
  - `PORT` — Bind port (default: `3000`)
  - `TRANSPORT` — Transport (`http` or `stdio`, default: `http`)

#### System Info Tool

The `system_info` tool allows the assistant to explain its own architecture and capabilities when users ask questions like "how do you work?" or "what can you do?". It reads the project README files and returns relevant documentation sections.

- Enabled by default in `mcp_server/tools_config.py`
- No configuration required
- Supports section filtering: `overview`, `structure`, `tools`, `memory`, `config`, `framework`, `troubleshooting`, or `all`

#### Kasa Smart Lights

```python
LIGHT_IPS = {
    "morgans_led": "192.168.1.49",
    "living_room_lamp": "192.168.1.165",
}
```

#### Google Calendar

Place credentials in `creds/`:

- `google_creds_<user>.json` — OAuth client secrets
- `token_<user>.json` — Access tokens (auto-generated on first auth)

**Calendar Selection Behavior:**

- **READ operations** — Default to `primary` which queries ALL calendars (shows events from Morgan Stuart, Birthdays, Family, HomeAssist, etc.)
- **WRITE operations** — Default to `homeassist` calendar to keep assistant-created events separate

> 💡 **Tip:** Create a calendar named "HomeAssist" in Google Calendar to keep assistant events organized separately from your main calendar.

#### Spotify

Set environment variables and configure users:

```python
SPOTIFY_USERS = {
    "Morgan": {"username": "your_spotify_username"},
}
```

---

## 🔔 Audio Feedback

HomeAssist provides distinct audio cues for every state transition and tool execution, giving you real-time feedback without looking at the terminal.

### State Transition Sounds

Every unique state transition plays a distinct sound automatically. This helps you know exactly what the system is doing.

| Transition | Sound (macOS) | Config Key | Description |
|------------|---------------|------------|-------------|
| **IDLE → Wake Word** | Tink | `idle_to_wakeword` | System entering wake word detection mode |
| **Wake Word → Transcribing** | Ping | `wakeword_to_transcribing` | Wake word detected, now recording |
| **Wake Word → Processing** | Submarine | `wakeword_to_processing` | Proactive response triggered |
| **Wake Word → Synthesizing** | Pop | `wakeword_to_synthesizing` | Pre-generated briefing starting |
| **Wake Word → IDLE** | Purr | `wakeword_to_idle` | Wake word detection cancelled |
| **Transcribing → Processing** | Hero | `transcribing_to_processing` | Transcription complete, processing request |
| **Transcribing → IDLE** | Blow | `transcribing_to_idle` | Transcription cancelled/timeout |
| **Processing → Synthesizing** | Morse | `processing_to_synthesizing` | Response ready, starting speech |
| **Processing → Transcribing** | Glass | `processing_to_transcribing` | Barge-in during processing |
| **Processing → IDLE** | Sosumi | `processing_to_idle` | Processing cancelled |
| **Synthesizing → IDLE** | Bottle | `synthesizing_to_idle` | TTS complete, system idle |
| **Synthesizing → Wake Word** | Frog | `synthesizing_to_wakeword` | TTS done, ready for wake word |
| **Synthesizing → Transcribing** | Ping | `synthesizing_to_transcribing` | Barge-in during TTS |
| **Any → ERROR** | Basso | `*_to_error` | System entering error state |
| **ERROR → IDLE** | Purr | `error_to_idle` | Error recovery complete |

### Customizing Transition Sounds

All transition sounds are configurable in `assistant_framework/config.py`:

```python
# Disable all transition beeps
ENABLE_TRANSITION_BEEPS = False

# Or customize individual sounds (macOS)
TRANSITION_SOUNDS = {
    "wakeword_to_transcribing": "Glass",  # Change wake word sound
    "synthesizing_to_idle": None,          # Disable this specific beep
    # ... other transitions
}
```

**Available macOS Sounds:**
`Basso`, `Blow`, `Bottle`, `Frog`, `Funk`, `Glass`, `Hero`, `Morse`, `Ping`, `Pop`, `Purr`, `Sosumi`, `Submarine`, `Tink`

> 💡 **Tip:** Only transitions listed in `TRANSITION_SOUNDS` will play a sound. Remove a transition from the dict to silence it, or set it to `None` to explicitly disable it.

### Tool Execution Feedback

**✅ Success Sound** (Pop/Glass):
- Tool executed successfully
- Response contains `"success": true` or no error field
- Pleasant, confirmatory tone

**❌ Failure Sound** (Funk/Basso):
- Tool execution failed
- Response contains `"success": false` or `"error"` field
- Warning tone to alert you of issues

**Examples:**

```json
// Triggers success sound ✅
{"success": true, "result": "Light turned on"}

// Triggers failure sound ❌
{"success": false, "error": "Device not found"}
```

> 💡 **Tip:** The system automatically detects success/failure from tool responses. No manual configuration needed.

### Event-Specific Sounds

These sounds play for specific events (not state transitions):

| Event | Sound | Description |
|-------|-------|-------------|
| **Send Phrase** | Hero/Funk | Send phrase detected in transcription |
| **Shutdown** | Blow | Assistant shutting down gracefully |

### TTS Announcements

In addition to audio beeps, HomeAssist can speak brief TTS announcements for key phase transitions. This provides clear spoken feedback for audio-only interactions.

**Announcements:**

| Event | Spoken Text | Description |
|-------|-------------|-------------|
| **Conversation Start** | "Listening" | Wake word detected, ready for input |
| **Conversation End** | "Conversation ended" | Termination phrase detected |
| **Tool Success** | "{tool} success" | Tool executed successfully |
| **Tool Failure** | "{tool} failed" | Tool execution failed |

**Pre-cached for Instant Playback:**

All announcements are pre-generated and cached at boot time for zero-latency playback. The cache is stored in `audio_data/announcement_cache/` and persists across restarts. On first run (or after clearing cache), the announcements are synthesized and saved. Subsequent boots load from cache instantly.

To clear the cache (e.g., after changing TTS voice):
```python
from assistant_framework.utils.audio.tts_announcements import clear_announcement_cache
clear_announcement_cache()
```

Or simply delete the `audio_data/announcement_cache/` folder.

**Configuration:**

```python
# In config.py (or via environment variable)
ENABLE_TTS_ANNOUNCEMENTS = True   # Enable spoken announcements (default)
ENABLE_TTS_ANNOUNCEMENTS = False  # Disable (beeps only)
```

**Environment variable:**
```bash
export ENABLE_TTS_ANNOUNCEMENTS=false  # Disable TTS announcements
```

> 💡 **Tip:** TTS announcements play alongside audio beeps. To have beeps only, set `ENABLE_TTS_ANNOUNCEMENTS=false`. To have TTS only, set `ENABLE_TRANSITION_BEEPS=false`.

---

## 📊 Token Tracking

HomeAssist provides detailed token usage tracking for API calls, helping you understand costs and optimize context usage.

### What's Tracked

| Component | Description |
|-----------|-------------|
| **Instructions** | System prompt + persistent memory |
| **Messages** | Conversation history context |
| **Tools** | Tool definitions/schemas |
| **Output** | Assistant response tokens |

### Console Output

Every API request logs a detailed token breakdown:

```
📊 API Input: 3,245 tokens (instructions: 1,200, messages: 845, tools: 1,200)
```

At session end, you'll see a full summary:

```
✅ Ended session: abc123...
   📊 5 messages | 4,500 total tokens
   📥 Input: 3,500 (context: 2,100, tools: 1,200)
   📤 Output: 1,000
```

### Token Tracking Methods

**For API requests** (includes full context):
```python
recorder.record_api_request(
    system_prompt="...",           # Instructions sent to model
    context_messages=[...],        # Conversation history
    tool_definitions=[...],        # Tool schemas
    user_message="...",            # Current user message
    vector_context="..."           # Semantic memory context
)
```

**Session statistics:**
```python
stats = recorder.session_token_stats
# {
#   "input_tokens": 3500,
#   "output_tokens": 1000,
#   "context_tokens": 2100,
#   "tool_tokens": 1200,
#   "total_tokens": 4500
# }
```

---

## 📊 Dashboard Integration

HomeAssist syncs with a remote dashboard for real-time monitoring.

### API Endpoints

- `/api/console/log` (POST/GET) — Console logging
- `/api/sessions` (POST/GET) — Manage sessions
- `/api/messages` (POST) — Record messages
- `/api/tool-calls` (POST) — Log tool usage

### Console Token

The `CONSOLE_TOKEN` isolates data per user/device:

- 🔹 Namespaces console logs
- 🔹 Filters database records
- 🔹 Enables multi-user dashboards

### Data Flow

```text
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  HomeAssist     │◄──────────────────►│  AssemblyAI     │
│  (Voice Loop)   │                    │  (Transcription)│
└────────┬────────┘                    └─────────────────┘
         │
         │ REST API
         ▼
┌─────────────────┐     Realtime       ┌─────────────────┐
│  Supabase       │◄──────────────────►│  Dashboard      │
│  (PostgreSQL)   │                    │  (Web UI)       │
└─────────────────┘                    └─────────────────┘
```

---

## 🔧 Troubleshooting

### ❌ Wake Word Not Detecting

**Symptoms:** Saying the wake phrase doesn't activate the assistant.

**Solutions:**

1. Lower the threshold (more sensitive):

```python
WAKEWORD_CONFIG = {
    "threshold": 0.15,  # Default is 0.2
}
```

2. Check your microphone is not muted

3. Specify the correct input device:

```python
WAKEWORD_CONFIG = {
    "input_device_index": 2,  # Find with: python -m sounddevice
}
```

---

### ❌ Barge-In Too Sensitive / Not Working

**Symptoms:** Assistant interrupts on background noise, or won't interrupt at all.

**Solutions:**

Adjust the energy threshold:

```python
BARGE_IN_CONFIG = {
    "energy_threshold": 0.08,   # Higher = less sensitive
    # "energy_threshold": 0.03, # Lower = more sensitive
}
```

---

### ❌ Audio Device Conflicts

**Symptoms:** Errors about device busy, or audio cutting out.

**Solutions:**

Increase handoff delay between components:

```python
TURNAROUND_CONFIG = {
    "state_transition_delay": 0.2,  # Default is 0.05
}
```

---

### ❌ Memory Not Persisting

**Symptoms:** Assistant forgets information between sessions.

**Solutions:**

1. Verify the file is writable:

```bash
ls -la state_management/persistent_memory.json
```

2. Check API key for memory provider:

```bash
# If using OpenAI for memory extraction
echo $OPENAI_API_KEY

# If using Gemini
echo $GEMINI_API_KEY
```

3. View memory extraction logs — look for `🧠 Persistent memory updated` in console

---

### ❌ Vector Memory Not Working

**Symptoms:** Past conversations not being retrieved, or "Failed to store in vector memory" errors.

**Solutions:**

1. Ensure Supabase has pgvector enabled. Run in Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Create the memories table. Get the SQL with:

```python
from assistant_framework.utils.vector_memory import VectorMemoryManager
from assistant_framework.config import VECTOR_MEMORY_CONFIG

vm = VectorMemoryManager(VECTOR_MEMORY_CONFIG)
print(vm.get_setup_sql())
```

3. Check Supabase credentials:

```bash
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

4. Verify vector memory is enabled in config:

```python
VECTOR_MEMORY_CONFIG = {
    "enabled": True,  # Must be True
    ...
}
```

---

### ❌ Termination Detection Not Working

**Symptoms:** Saying "over out" during TTS doesn't stop the conversation.

**Solutions:**

1. Check if the model exists:
```bash
ls audio_data/wake_word_models/over_out.onnx
```

2. If missing, you need to train a custom model or use text-based termination only:
```python
# Disable parallel termination (use text-based only)
TERMINATION_DETECTION_CONFIG = {
    "enabled": False,
}
```

3. Lower the threshold for more sensitivity:
```python
TERMINATION_DETECTION_CONFIG = {
    "threshold": 0.3,  # Default is 0.5
}
```

4. Check console for initialization message:
```
✅ Termination detection ready (PID: 12345)
```
or
```
⚠️  Termination detection unavailable (model not found)
```

---

### ❌ Configuration Validation

Run the built-in validator:

```python
from assistant_framework.config import validate_environment, print_config_summary

# Print full config status
print_config_summary()

# Get validation results
results = validate_environment()
print(results)
```

---

## 🎯 Advanced Features

### Custom System Prompt

Edit `SYSTEM_PROMPT_CONFIG` in `config.py` to change the assistant's personality. The prompt is built from a structured dictionary for easier customization.

> 💡 **Tip:** The system prompt automatically includes the current date and year, so the assistant always knows what day it is when scheduling events or interpreting relative dates.

### Pattern-Based Memory

The persistent memory system tracks behavioral patterns with strength levels:

- **weak** — Speculative, single instance
- **moderate** — Some supporting evidence
- **strong** — Clear recurring behavior
- **confirmed** — Practically a fact

Patterns upgrade/downgrade based on evidence and inform what gets stored as known facts.

### Verbose Logging

Control terminal output verbosity for cleaner production logs:

```python
# In config.py (or via environment variable)
VERBOSE_LOGGING = True   # Show all status messages (default)
VERBOSE_LOGGING = False  # Minimal output: only errors, warnings, and key events
```

**Environment variable:**
```bash
export VERBOSE_LOGGING=false  # Disable verbose logging
```

**What shows when `VERBOSE_LOGGING=False`:**

| Always Shown | Hidden |
|-------------|--------|
| ❌ Errors | 🔄 Status updates |
| ⚠️ Warnings | ✅ Success confirmations |
| 🎯 Wake word detected | 🎤 Audio stream details |
| 👤 User message | 🧹 Cleanup messages |
| 🤖 Assistant message | ⏳ Waiting messages |
| 💀 Critical errors | 📋 Config summaries |

**Using in code:**
```python
from assistant_framework.utils.logging_config import vprint, eprint

vprint("🔄 Processing...")     # Only shows if VERBOSE_LOGGING=True
print("❌ Connection failed")  # Always shows (use for errors)
eprint("🎯 Wake word!")        # Shows if verbose OR has essential prefix (❌⚠️🎯👤🤖)
```

### Latency Tuning

For faster response times, adjust turnaround delays:

```python
TURNAROUND_CONFIG = {
    "state_transition_delay": 0.02,   # Component switches
    "barge_in_resume_delay": 0.02,    # After interruption
    "transcription_stop_delay": 0.1,  # After transcription ends
}
```

> ⚠️ **Warning:** Very low values may cause audio device conflicts on some systems.

### Fast Reboot (Warm Mode)

Dramatically reduce the time to restart wake word detection after a conversation ends:

```python
TURNAROUND_CONFIG = {
    # ... other settings ...
    "wake_word_warm_mode": True,      # Keep subprocess alive (~2s faster restart)
    "post_conversation_delay": 0.0,   # No delay after conversation ends
    "wake_word_stop_delay": 0.0,      # No delay after pausing wake word
}
```

**How it works:**

| Mode | Restart Time | Description |
|------|-------------|-------------|
| **Warm mode** (default) | ~200ms | Subprocess stays alive, audio stream pauses/resumes |
| **Cold mode** | ~2-3s | Full subprocess termination and restart with model loading |

**Warm mode benefits:**

- ⚡ **Faster restart** — No model reloading between conversations
- 💾 **Lower memory churn** — Models stay loaded in the subprocess
- 🎯 **Same reliability** — Process isolation still prevents model corruption

**When to disable warm mode:**

- Memory-constrained systems where you need to free subprocess memory
- Debugging wake word detection issues (full restart gives cleaner state)

```python
# Disable warm mode (cold restarts)
TURNAROUND_CONFIG = {
    "wake_word_warm_mode": False,
    "post_conversation_delay": 0.2,   # Add settling time for cold restart
    "wake_word_stop_delay": 0.1,      # Wait for process termination
}
```

💡 **Tip:** Monitor console logs for `⚡ Resuming wake word detection (warm mode)` to confirm warm mode is active.

---

## 📁 State Files

- `state_management/app_state.json` — Runtime state
- `state_management/conversation_summary.json` — Current session summary
- `state_management/persistent_memory.json` — Long-term user memory

---

## ⏰ Scheduled Jobs (GitHub Actions)

HomeAssist includes scheduled background jobs that run via GitHub Actions cron. These create briefing announcements and notifications that the assistant delivers on wake word.

### Workflow Overview

The workflow (`.github/workflows/scheduled_events_cron.yml`) runs daily and includes:

| Job | Description | Output |
|-----|-------------|--------|
| **Email Summarizer** | Summarizes unread emails | `notification_sources` table |
| **News Summarizer** | Curates daily news digest | `notification_sources` table |
| **Calendar Briefing** | Creates reminder announcements | `briefing_announcements` table |
| **Weather Briefing** | Alerts for unusual weather | `briefing_announcements` table |

### Required GitHub Secrets

| Secret | Used By | Description |
|--------|---------|-------------|
| `GEMINI_API_KEY` | Email, Calendar | Google Gemini API key |
| `OPENAI_API_KEY` | News | OpenAI API key |
| `SUPABASE_URL` | All | Supabase project URL |
| `SUPABASE_KEY` | All | Supabase service role key |
| `NEWS_API_KEY_1` | News | NewsAPI.org key |

### Gmail Secrets (Email Summarizer)

| Secret | Description |
|--------|-------------|
| `GMAIL_CLIENT_ID` | OAuth client ID from Google Cloud Console |
| `GMAIL_CLIENT_SECRET` | OAuth client secret |
| `GMAIL_REFRESH_TOKEN` | Refresh token (get via `reauth_gmail.py`) |

### Calendar Secrets (Calendar Briefing)

**Option 1: Service Account (Recommended for CI - No User Interaction)**

| Secret | Description |
|--------|-------------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Base64-encoded service account JSON key |

**Setting up a Service Account:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Click **Create Service Account**
   - Name: `calendar-briefing`
   - Role: None needed (calendar access is per-calendar)
5. Click on the new service account → **Keys** → **Add Key** → **JSON**
6. Download the JSON key file
7. **Share your calendar** with the service account email:
   - Open Google Calendar
   - Click the gear icon on your calendar → **Settings and sharing**
   - Under "Share with specific people", add the service account email
   - Permission: "See all event details" or "Make changes to events"
8. Base64 encode and add to GitHub secrets:

```bash
cat service-account-key.json | base64
```

9. Add as GitHub secret: `GOOGLE_SERVICE_ACCOUNT_JSON`

> ✅ **Advantages:** Never expires, no user interaction, works perfectly in CI/CD

---

**Option 2: OAuth (Requires Initial User Interaction)**

| Secret | Description |
|--------|-------------|
| `GOOGLE_CREDENTIALS_JSON` | Base64-encoded OAuth client credentials |
| `GOOGLE_CALENDAR_TOKEN_JSON` | Base64-encoded token with calendar scopes |

**Creating Calendar Token:**

1. Run locally to trigger OAuth (if needed):
```bash
python -c "from mcp_server.clients.calendar_client import CalendarComponent; c = CalendarComponent(user='morgan_personal')"
```

2. Base64 encode your token:
```bash
cat creds/token_morgan.json | base64
```

3. Add as GitHub secret: `GOOGLE_CALENDAR_TOKEN_JSON`

> ⚠️ **Note:** OAuth tokens can expire if unused for 6+ months. Service accounts are more reliable for CI.

### Weather Variables (Weather Briefing)

| Variable | Description | Default |
|----------|-------------|---------|
| `WEATHER_ZIP_CODE` | US ZIP code for weather location | Auto-detected |
| `WEATHER_USER_ID` | User to receive weather briefings | `Morgan` |

> 💡 **Note:** Weather briefing auto-detects location from IP address if `WEATHER_ZIP_CODE` is not set. Only creates alerts when unusual weather is detected (rain, snow, extreme temps, high winds).

### Manual Trigger

Run the workflow manually from GitHub Actions UI with options to skip specific jobs:
- `skip_email` — Skip email summarizer
- `skip_news` — Skip news summarizer  
- `skip_calendar` — Skip calendar briefing
- `skip_weather` — Skip weather briefing

### Calendar Briefing Details

The calendar briefing job (`scripts/scheduled/calendar_briefing/`):

1. **Fetches** upcoming calendar events (7 days ahead by default)
2. **Filters** out already-processed events (via `calendar_event_cache` table)
3. **Analyzes** events with AI to determine optimal reminder timing
4. **Creates** briefing announcements with pre-generated opener text
5. **Stores** to `briefing_announcements` table in Supabase

**Dynamic Time Calculation:**

Calendar briefings use `{{TIME_UNTIL_EVENT}}` placeholder that gets replaced with the actual time remaining when delivered:

- Pre-generated opener: `"Your meeting is in {{TIME_UNTIL_EVENT}}."`
- At delivery (e.g., 45 min before): `"Your meeting is in about 45 minutes."`
- At delivery (e.g., 2 hours before): `"Your meeting is in about 2 hours."`

This ensures accurate timing regardless of when the briefing window opens.

Configuration:
- `REMINDER_LOOKAHEAD_DAYS` — Days ahead to look (default: 7)
- `REMINDER_CALENDAR_USERS` — Comma-separated calendar users (default: `morgan_personal`)

### Weather Briefing Details

The weather briefing job (`scripts/scheduled/weather_briefing/`):

1. **Auto-detects** location from IP (or uses cached/configured coordinates)
2. **Fetches** 7-day forecast from Open-Meteo (no API key required)
3. **Analyzes** full week to establish context (averages, patterns)
4. **Alerts** only on TODAY (next 24 hours) for unusual conditions
5. **Creates** briefing announcements only if alerts are detected
6. **Stores** to `briefing_announcements` table in Supabase

The system uses the broader forecast to detect anomalies - for example, if today is 15°F colder than the week average, it will alert even if it doesn't hit absolute thresholds.

**Alert Types:**
- 🌧️ **Rain** — Precipitation > 50% probability
- ❄️ **Snow** — Any snow in forecast  
- ⛈️ **Thunderstorms** — Especially with hail
- 🥶 **Freezing precipitation** — Ice/freezing rain
- 🔥 **Extreme heat** — Seasonally adjusted (95°F+ in summer)
- 🧊 **Extreme cold** — Seasonally adjusted (20°F- in winter)
- 💨 **High winds** — Gusts > 40 mph

**Severity Levels:**
- `severe` — Dangerous conditions, take action
- `significant` — Notable weather, plan accordingly
- `moderate` — Minor inconvenience, be aware

Configuration:
- `WEATHER_ZIP_CODE` — US ZIP code for location
- `WEATHER_USER_ID` — Target user for briefings

---

## 📝 License

MIT License — See LICENSE file for details.
