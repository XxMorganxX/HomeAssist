HomeAssist – Voice Assistant System

Overview

HomeAssist is a local-first voice assistant system that combines wake-word detection, real‑time speech transcription, LLM‑powered responses, and text‑to‑speech playback. It includes a modular assistant framework with swappable providers (AssemblyAI STT, OpenAI Realtime responses with MCP tools, Google Cloud TTS) and a separate MCP server exposing smart‑home tools (lights, calendar, Spotify, weather, etc.).

Key Features

- Wake word → transcription → response → TTS loop
- Provider abstraction and easy swapping via config
- OpenAI Realtime API support with MCP tool calling
- Google Cloud Text‑to‑Speech playback (afplay on macOS)
- Shared audio resource manager to avoid device conflicts
- Robust session controls (send phrases, termination phrases, timeouts)

Repo Structure

- `main.py`: Top-level app orchestrating wake word → transcription → response → TTS
- `assistant_framework/`: Modular framework, CLI, providers, orchestrator
  - `interfaces/`: Abstractions for STT, response, TTS, context, wake word
  - `providers/`: Implementations (AssemblyAI, OpenAI WebSocket + MCP, Google TTS, OpenWakeWord)
  - `orchestrator.py`: Wires components together
  - `config.py`: Provider selection, system prompt, presets, validation
  - `main.py`: CLI entrypoint (`python -m assistant_framework.main ...`)
  - `utils/`: Audio manager and non-blocking tones
- `mcp_server/`: FastMCP server exposing home tools (HTTP or stdio)
- `core/`: Legacy/alternate streaming chatbot components (Whisper, traditional chat)

Prerequisites

- Python 3.11+ (3.11 recommended; Python 3.13 may cause NumPy wheel issues)
- macOS with `afplay` for audio playback (built-in) or ffmpeg for post-processing (optional)
- Microphone access (PyAudio)
- Accounts/keys where used:
  - AssemblyAI (streaming STT)
  - OpenAI (Realtime + Chat Completions + optional TTS in core/)
  - Google Cloud service account JSON for Google TTS

Quick Start

1) Create and activate a virtual environment

```bash
cd /Users/morgannstuart/Desktop/HomeAssist
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Set environment variables (example)

```bash
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/google_service_account.json"
```

You can also place a `.env` file at the project root to be read by `python-dotenv` where applicable.

3) Run the main Home Assistant loop

```bash
python main.py
```

You should see initialization logs, then “Listening for wake word…”. Say the configured wake word; speak; use a send phrase (e.g., “send message”) to trigger the response; use a termination phrase (e.g., “stop listening”) to end the transcription phase.

assistant_framework CLI

The framework provides a CLI to run individual components or full pipelines.

Examples:

```bash
# Full pipeline (mic → response → TTS)
python -m assistant_framework.main pipeline

# Transcription only
python -m assistant_framework.main transcribe --final-only

# Response only (single message)
python -m assistant_framework.main respond --message "Hello" --no-context

# TTS only (Google TTS via assistant_framework)
python -m assistant_framework.main tts --text "Hello Mr. Stuart" --no-play --save speech_audio/hello.mp3

# Single message through pipeline
python -m assistant_framework.main single --message "Turn on the living room lights"

# Wake word only
python -m assistant_framework.main wakeword

# Tool discovery via the active response provider (if MCP configured)
python -m assistant_framework.main tools
```

Common flags:

- `--env`: `default` (from preset), `dev`, `prod`, `test`
- `--transcription`, `--response`, `--tts`, `--context`: override provider names at runtime

Configuration

- Primary config for the framework is in `assistant_framework/config.py`.
- Default providers: AssemblyAI STT, OpenAI WebSocket (Realtime) response with MCP, Google TTS, Unified context, OpenWakeWord.
- System prompt and operational phrases:
  - Termination phrases: `TERMINATION_PHRASES = ["over out", "stop listening", "end session", "over, out"]`
  - Send phrases: `SEND_PHRASES = ["send message", "process this", "respond to this", "send this", "send it", "sir", "king"]`
- Presets: `CONFIG_PRESET` (`prod` | `dev` | `test` | `default`) with helpers to set active preset at runtime.

Required environment variables (typical)

- `ASSEMBLYAI_API_KEY` (required for streaming STT)
- `OPENAI_API_KEY` (required for OpenAI Realtime/tool composition)
- `GOOGLE_APPLICATION_CREDENTIALS` (path to Google service account JSON for TTS)

Optional environment variables

- `AF_CONFIG_PRESET` to choose preset (`prod`, `dev`, `test`)
- `QUIET_IMPORT=1` to suppress config summary on import
- `GOOGLE_TTS_TRANSPORT=rest` to force REST transport for Google TTS

MCP Server

The MCP server exposes smart‑home tools (calendar, lights, Spotify, weather) and can be used by the OpenAI Realtime response provider via stdio transport.

Install MCP server deps and run:

```bash
# In a shell with the project venv activated
pip install -r mcp_server/requirements.txt

# Option A: HTTP transport (default)
python -m mcp_server.server --host 127.0.0.1 --port 3000 --transport http

# Option B: stdio transport (for the Realtime provider integration)
python -m mcp_server.server --transport stdio

# Or use the helper script (creates/uses a venv if needed)
bash mcp_server/run.sh
```

The assistant framework auto‑detects the MCP server path (`assistant_framework/config.py`) and will attempt to start/connect via stdio when using the OpenAI WebSocket provider.

Audio Notes

- PyAudio is used for mic input. On macOS, if device conflicts occur, the app uses a shared audio manager to cleanly transition between wake word and transcription.
- Google TTS playback uses `afplay`. If ffmpeg is installed, HD voices can be speed/pitch adjusted via a post‑processing step.

Troubleshooting

- Missing environment variables
  - Ensure `ASSEMBLYAI_API_KEY` and `OPENAI_API_KEY` are set. For Google TTS, set `GOOGLE_APPLICATION_CREDENTIALS` to a readable JSON file.

- Audio device busy / conflicts
  - The app performs forced cleanup and handoff delays between components. If issues persist, stop other apps using the mic and try again.

- MCP tools not available
  - If MCP path is not detected or server fails to start, the framework will continue without tools. Check `mcp_server/run.sh` and the server logs.

- Google TTS errors
  - Set `GOOGLE_TTS_TRANSPORT=rest` to avoid gRPC issues. Verify the credentials path.

- OpenAI Realtime timeouts
  - Network hiccups can cause timeouts; re‑run the command. Ensure the API key is valid.

- NumPy C-extension import error on Python 3.13
  - Symptom: `ImportError: No module named 'numpy._core._multiarray_umath'` when importing NumPy (e.g., via OpenWakeWord).
  - Fix (recommended): use Python 3.11. Example:
    ```bash
    brew install python@3.11 # if not installed
    cd /Users/morgannstuart/Desktop/HomeAssist
    rm -rf venv
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```
  - Alternative: try a clean reinstall of NumPy in your existing venv (may still fail on 3.13):
    ```bash
    source venv/bin/activate
    pip install --force-reinstall --no-cache-dir numpy
    ```

License

Private project. All rights reserved.


