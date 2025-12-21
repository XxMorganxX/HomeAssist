# HomeAssist Setup Guide

A modular voice assistant framework featuring wake word detection, real-time transcription, LLM-powered responses, persistent memory, and smart home integration.

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

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- macOS (for local TTS) or Linux with audio support
- Microphone and speakers

### Step 1: Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/HomeAssistV2.git
cd HomeAssistV2
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
python -m assistant_framework.main_v2 continuous

# Single interaction mode
python -m assistant_framework.main_v2 single
```

> 💡 **Tip:** On first run, a configuration summary will print showing which components are active and any missing credentials.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with your API keys.

#### Required Variables

- `OPENAI_API_KEY` — OpenAI API key for GPT responses
- `ASSEMBLYAI_API_KEY` — AssemblyAI key for transcription

#### Optional — AI Services

- `GEMINI_API_KEY` — Google Gemini (for summarization)
- `GOOGLE_APPLICATION_CREDENTIALS` — Path to Google Cloud JSON

#### Optional — Integrations

- `SPOTIFY_CLIENT_ID` — Spotify OAuth client ID
- `SPOTIFY_CLIENT_SECRET` — Spotify OAuth client secret
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_KEY` — Supabase service role key
- `CONSOLE_TOKEN` — Dashboard log isolation token

> ⚠️ **Important:** Never commit your `.env` file to version control!

---

### Config File Reference

All runtime settings live in `assistant_framework/config.py`, organized by feature:

#### Provider Selection

Choose which implementation to use for each component:

```python
TRANSCRIPTION_PROVIDER = "assemblyai"      # "assemblyai" or "openai_whisper"
RESPONSE_PROVIDER = "openai_websocket"     # LLM backend
TTS_PROVIDER = "local_tts"                 # "google_tts" or "local_tts"
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

#### Wake Word Settings

```python
WAKEWORD_CONFIG = {
    "model_name": "hey_honey",    # Trigger phrase model
    "threshold": 0.2,             # Sensitivity (lower = more sensitive)
    "cooldown_seconds": 2.0,      # Min time between activations
}
```

> 💡 **Tip:** Custom wake word models (`.onnx`) go in `audio_data/wake_word_models/`

#### Conversation Flow

```python
TERMINATION_PHRASES = ["over out", "stop listening"]
SEND_PHRASES = ["send it", "sir"]
AUTO_SEND_SILENCE_TIMEOUT = 6.0  # Auto-send after 6s of silence
```

#### Barge-In Detection

Allows interrupting the assistant while it's speaking:

```python
BARGE_IN_CONFIG = {
    "energy_threshold": 0.055,        # Voice detection sensitivity
    "early_barge_in_threshold": 3.0,  # Early interrupts append to previous message
    "min_speech_duration": 0.2,       # Required speech to trigger
}
```

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

> 💡 **Note:** No index is needed for < 10k vectors. Search latency is ~50ms which is negligible for voice.

---

## 🏠 Smart Home Setup

### MCP Server

The MCP (Model Context Protocol) server provides tool capabilities. Configuration is in `mcp_server/config.py`.

#### Kasa Smart Lights

```python
LIGHT_IPS = {
    "morgans_led": "192.168.1.49",
    "living_room_lamp": "192.168.1.165",
}
```

#### Google Calendar

Place credentials in `mcp_server/google_creds/`:

- `google_creds_<user>.json` — OAuth client secrets
- `token_<user>.json` — Access tokens (auto-generated on first auth)

#### Spotify

Set environment variables and configure users:

```python
SPOTIFY_USERS = {
    "Morgan": {"username": "your_spotify_username"},
}
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

Edit `SYSTEM_PROMPT` in `config.py` to change the assistant's personality:

```python
SYSTEM_PROMPT = """
You're a helpful assistant named Jarvis.
Be formal and precise in your responses.
"""
```

### Pattern-Based Memory

The persistent memory system tracks behavioral patterns with strength levels:

- **weak** — Speculative, single instance
- **moderate** — Some supporting evidence
- **strong** — Clear recurring behavior
- **confirmed** — Practically a fact

Patterns upgrade/downgrade based on evidence and inform what gets stored as known facts.

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

---

## 📁 State Files

- `state_management/app_state.json` — Runtime state
- `state_management/conversation_summary.json` — Current session summary
- `state_management/persistent_memory.json` — Long-term user memory

---

## 📝 License

MIT License — See LICENSE file for details.
