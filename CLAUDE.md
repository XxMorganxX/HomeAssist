# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

HomeAssistV3 is a modular voice assistant framework for macOS. It runs a real-time audio pipeline (wake word → transcription → LLM → tool execution → TTS → speaker) with a Discord text interface as an alternate frontend. Tools are served via a persistent MCP server.

## Running the Project

```bash
# Primary entry point (voice assistant with Bluetooth auto-connect)
./homeassist run

# Individual components
python -m assistant_framework.main continuous   # Voice assistant only
python -m assistant_framework.main single       # Single interaction
./homeassist mcp start                          # MCP tool server (port 3000)
./homeassist discord                            # Discord bot

# Service management
./homeassist start    # Start all background services (launchd)
./homeassist stop     # Stop all services
./homeassist restart  # Restart assistant
./homeassist logs     # Tail live logs
```

## Running Tests

```bash
cd tests/
pytest test_vector_memory.py          # Individual test file
python run_websocket_tests.py         # WebSocket integration tests
```

No linter or formatter is configured.

## Architecture

### Provider Pattern (core abstraction)

Every major component follows an ABC → concrete implementation → factory pattern:
- Define an ABC in `interfaces/`
- Implement in `providers/<domain>/`
- Wire via factory that maps config name → class
- Call sites import only the ABC

Current providers configured in `assistant_framework/config.py`:

| Slot | Config Key | Default |
|------|-----------|---------|
| Transcription | `TRANSCRIPTION_PROVIDER` | `assemblyai` |
| Response | `RESPONSE_PROVIDER` | `openai_websocket` |
| TTS | `TTS_PROVIDER` | `openai_tts` |
| Wake Word | `WAKEWORD_PROVIDER` | `openwakeword` |
| Tool Routing | `TOOL_ROUTING_PROVIDER` | `tool_calling_mini` |
| Context | `CONTEXT_PROVIDER` | `unified` |

### Orchestrator

`assistant_framework/orchestrator/` contains `RefactoredOrchestrator` (~1900 lines) — the central state machine coordinating the audio pipeline. States: IDLE → WAKE_WORD_LISTENING → TRANSCRIBING → PROCESSING_RESPONSE → SYNTHESIZING → ERROR.

### Tool Routing Layer

Tool routing is decoupled from the response provider:
- `ToolRoutingInterface` ABC with `route()` and `check_additional_tools()` for iterative calling
- `ToolCallingMiniProvider` — uses a fine-tuned Qwen3 model via external API with rotating HMAC-SHA256 API keys
- `OpenAIToolRoutingProvider` — GPT-4o-mini fallback
- Tool *execution* stays in the response provider (MCP invocation), routing just selects which tools to call

### MCP Tool Server

`mcp_server/` runs a FastMCP server on port 3000 (SSE transport). Tools include Spotify, Kasa lighting, calendar, weather, search, SMS, etc. Tool enable/disable is in `mcp_server/tools_config.py`.

### Discord Bot

`discord_bot/` provides a text interface reusing the same tool routing + response providers. Listens on a single channel (`DISCORD_CHANNEL_ID`). Runs in background alongside voice when `DISCORD_BOT_TOKEN` is set.

### Memory System

Supabase-backed with pgvector for semantic search. Includes persistent facts (JSON), conversation summarization, and vector memory with a local numpy cache for fast lookups.

### Process Isolation

- Wake word detection runs in a separate process
- Fast termination detection runs in a parallel process
- MCP server is a separate persistent process

## Key Design Rules

1. **`homeassist` script is the single source of truth** for all runtime logic (Bluetooth, error detection, restarts). The launchd plist only calls `homeassist run` — never duplicate logic in plist files.

2. **Tool Signal Mode** (`TOOL_SIGNAL_MODE`): The realtime model outputs a brief tool signal; o4-mini then handles full parameter construction and execution. Reduces latency.

3. **Configuration** lives in `assistant_framework/config.py` (~1185 lines). All provider settings, audio pipeline timings, memory config, and feature flags are centralized there. Environment variables come from `.env` (see `env.example`).

## Environment Setup

Required keys: `OPENAI_API_KEY`, `ASSEMBLYAI_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`

Optional: `SPOTIFY_CLIENT_ID/SECRET`, `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `INFERENCE_REFRESH_TOKEN`, `NEWS_API_KEY_1` through `NEWS_API_KEY_5`

See `env.example` for the full list.
