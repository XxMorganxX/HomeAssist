# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a voice-based home assistant system for Morgan, Spencer, and guests that uses OpenAI's Realtime API for low-latency voice interactions. The system provides home automation control (lights, music, calendar) with continuous conversation capability.

## Key Architecture Components

### Core Files
- `main.py` - Entry point for the voice assistant
- `core/streaming_chatbot_realtime.py` - Primary WebSocket-based Realtime API implementation
- `core/speech_services_realtime.py` - Real-time streaming chatbot with continuous audio processing
- `core/context_manager.py` - Pure context data provider for conversation management
- `config.py` - Complete system configuration

### Critical Technical Concepts

#### 1. Realtime API vs Traditional API
- **Realtime API**: WebSocket-based streaming with session-level system prompts
- **Traditional API**: REST-based with message-level system prompts
- The system can fall back to traditional API if Realtime API fails

#### 2. Context Management Architecture
- **ConversationManager**: Pure conversation oracle - stores and retrieves messages
- **ContextManager**: Pure context data provider - formats context for different use cases
- **SpeechServices**: Makes intelligent decisions about when to use context

#### 3. Session Summary + Sliding Window
- Once conversation > `CONTEXT_SUMMARY_MIN_MESSAGES` (5), use AI-generated summary + recent messages
- Recent messages window size: `REALTIME_SLIDING_WINDOW_SIZE` (6)
- Summary includes User_Summary and Response_Summary fields
- Context is lossless: all messages are either in summary or sliding window

#### 4. System Prompt Handling
- **Realtime API**: System prompt sent at session level only (via session.update)
- **Traditional API**: System prompt sent as first message
- Temperature minimum 0.6 required for Realtime API instruction following

## Key Configuration Variables

### Realtime API Settings
```python
USE_REALTIME_API = True  # Enable/disable Realtime API
REALTIME_STREAMING_MODE = True  # Continuous streaming vs chunk-based
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"
REALTIME_VOICE = "alloy"
REALTIME_USE_SUMMARY_CONTEXT = True  # Enable summary + sliding window
REALTIME_SLIDING_WINDOW_SIZE = 6  # Recent messages with summary
```

### Context Management
```python
CONTEXT_SUMMARY_MIN_MESSAGES = 5  # Start summaries after N messages
CONTEXT_SUMMARY_FREQUENCY = 5  # Generate summary every N messages
CONTEXT_SUMMARY_MODEL = "gpt-4o-mini"  # Model for generating summaries
RESPONSE_TEMPERATURE = 0.6  # Minimum for Realtime API
```

### Console Output Configuration
```python
DETAILED_FUNCTION_LOGGING = False  # Show detailed function call result data (for debugging)
```

## Important System Behaviors

### 1. System Prompt Requirements
- Uses imperative sentences, not bullet points
- NEVER end with engagement phrases ("feel free to ask", "let me know")
- Stop immediately after providing answers
- Always use tools for real-time information

### 2. Tool Call Restrictions
- NEVER query multiple people's calendars or notifications in one request
- For calendar day summaries, user MUST say "today" - don't assume they mean today
- Default to the current user if no specific person is mentioned

### 3. Context Flow
1. Short conversations (≤5 messages): Full context sent
2. Long conversations (>5 messages): Summary + 6 recent messages
3. Summary generation triggered after assistant messages
4. All context decisions made in SpeechServices layer

### 4. Session State Management
- Session summary stored in `core/state_management/session_summary.json`
- State management file: `core/state_management/app_state.json`
- Only User_Summary and Response_Summary are passed to Realtime API

## MCP Tool Architecture

### Tool Discovery and Registration
- Tools are automatically discovered from `mcp_server/tools/`
- Each tool inherits from `BaseTool` and implements required methods
- No manual registration needed - tools are loaded dynamically

### Available Tools
- `state_manager` - Read/update system state
- `get_notifications` - Query user notifications (single user only)
- `batch_light_control` - Control multiple Kasa smart lights
- `lighting_scene` - Apply lighting scenes (mood, party, etc.)
- `spotify_playback` - Control Spotify playback
- `calendar_data` - Query Google Calendar (single user, explicit "today" for day summaries)

### Tool Development Pattern
```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "Tool description"
    version = "1.0.0"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation
        return {"success": True, "result": "..."}
```

## Audio Processing Pipeline

### Voice Activity Detection (VAD)
- **Dual VAD System**: Server-side (OpenAI) + Client-side (WebRTC)
- **Cost Optimization**: Filters silence before sending to API
- **Interruption Detection**: Transcription-based (not immediate VAD-based)

### Audio Enhancement (Configurable)
- **AEC (Acoustic Echo Cancellation)**: NLMS adaptive filtering
- **Feedback Detection**: Automatic feedback loop prevention
- **Noise Reduction**: Ambient noise filtering and speech enhancement

### Configuration
```python
AEC_ENABLED = False  # Enable/disable AEC processing
FEEDBACK_DETECTION_ENABLED = False  # Enable automatic feedback detection
REALTIME_CLIENT_VAD_ENABLED = True  # Use client-side VAD filtering
```

## Common Development Commands

### Running the System
```bash
# Main voice assistant
python main.py

# Realtime streaming version  
python core/streaming_chatbot_realtime.py

# Terminal test mode for MCP tools
python main.py --test
```

### Testing and Development
```bash
# Test MCP tools interactively
python examples/interactive_tool_chat.py

# Run chat classifier as script
python scripts/chat_classifier.py

# Check microphone setup
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Configuration Management
```bash
# Environment setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Required environment variables
echo "OPENAI_KEY=your-key-here" > .env
```

## Database and State Management

### Database Schema
- `sys_prompt` column stores system prompts (TEXT format)
- Chat data stored separately from system prompts
- Genre classification tracks conversation types

### State Files
- `core/state_management/app_state.json` - Current system state
- `core/state_management/session_summary.json` - AI-generated conversation summaries

## Debugging Features

### Audio Processing Debug
```python
REALTIME_API_DEBUG = True  # WebSocket message debugging
REALTIME_DEBUG = True  # General realtime debugging
REALTIME_VAD_DEBUG = True  # VAD debugging output
```

### Function Call Debug
```python
DETAILED_FUNCTION_LOGGING = True  # Show detailed tool execution data
```

### Context Debug
```python
DISPLAY_CONTEXT = True  # Context debugging
```

## Important Notes for Development

### Permission-Based Logging
- Always ask user permission before adding detailed logging
- Use `config.request_logging_permission(description)` function
- Default to minimal console output
- **User preference: Concise terminal output** - Keep console messages brief and essential only

### Tool Call Best Practices
- Only query one person at a time for calendar/notifications
- Require explicit "today" mention for day summaries
- Use semantic understanding, not keyword matching

### Realtime API Specifics
- WebSocket connection required for full functionality
- Session-level instructions only (no message-level system prompts)
- Temperature must be ≥0.6 for proper instruction following
- Supports both streaming and chunk-based modes

### Context Drift Prevention
- Track conversation item IDs for debugging
- Never clear conversation queues (breaks session context)
- Debug with item ID logging when responses don't match questions

## File Structure
```
core/
├── streaming_chatbot_realtime.py   # Main Realtime API implementation
├── speech_services_realtime.py     # WebSocket speech services
├── context_manager.py              # Pure context data provider
├── components.py                   # Component orchestration
├── audio_processing.py             # VAD, AEC, and audio filtering
├── state_management/
│   ├── session_summary.json        # AI-generated conversation summaries
│   └── app_state.json              # System state persistence
└── db/
    └── db_connect.py                # Database operations

mcp_server/
├── server.py                       # MCP tool server
├── base_tool.py                    # Abstract tool base class
├── tool_registry.py               # Automatic tool discovery
└── tools/                          # Smart home tools
    ├── calendar_data.py
    ├── get_notifications.py
    ├── spotify_playback.py
    └── batch_light_control.py

config.py                           # Complete system configuration
main.py                            # Main entry point
```

This system provides a robust, low-latency voice assistant with intelligent context management and seamless integration with home automation tools.