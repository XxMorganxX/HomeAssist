# CLAUDE.md - OpenAI Realtime API Voice Assistant

## Project Overview
This is a voice-based home assistant system for Morgan, Spencer, and guests that uses OpenAI's Realtime API for low-latency voice interactions. The system provides home automation control (lights, music, calendar) with continuous conversation capability.

## Key Architecture Components

### Core Files
- `main.py` - Entry point for the voice assistant
- `core/speech_services_realtime.py` - Primary WebSocket-based Realtime API implementation
- `core/streaming_chatbot_realtime.py` - Real-time streaming chatbot with continuous audio processing
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
- Once conversation > `CONTEXT_SUMMARY_MIN_MESSAGES` (3), use AI-generated summary + recent messages
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
CONTEXT_SUMMARY_MIN_MESSAGES = 3  # Start summaries after N messages
CONTEXT_SUMMARY_FREQUENCY = 3  # Generate summary every N messages
CONTEXT_SUMMARY_MODEL = "gpt-4o-mini"  # Model for generating summaries
RESPONSE_TEMPERATURE = 0.6  # Minimum for Realtime API
```

## Important System Behaviors

### 1. System Prompt Requirements
- Uses imperative sentences, not bullet points
- NEVER end with engagement phrases ("feel free to ask", "let me know")
- Stop immediately after providing answers
- Always use tools for real-time information

### 2. Context Flow
1. Short conversations (≤3 messages): Full context sent
2. Long conversations (>3 messages): Summary + 6 recent messages
3. Summary generation triggered after assistant messages
4. All context decisions made in SpeechServices layer

### 3. Session State Management
- Session summary stored in `core/state_management/session_summary.json`
- State management file: `core/state_management/app_state.json`
- Only User_Summary and Response_Summary are passed to Realtime API

## Common Development Tasks

### Testing Context Management
```bash
# Check current context being sent to API
# Look for DISPLAY_CONTEXT debug output in logs

# Test summary generation
# Have a conversation with >3 messages and check session_summary.json
```

### Debugging Realtime API
```python
# Enable debug output in config.py
REALTIME_API_DEBUG = True  # WebSocket message debugging
REALTIME_DEBUG = True  # General realtime debugging
DISPLAY_CONTEXT = True  # Context debugging
```

### Running the System
```bash
# Main voice assistant
python main.py

# Realtime streaming version
python core/streaming_chatbot_realtime.py

# Test complete realtime functionality
python test_complete_realtime.py
```

## Recent Major Changes

### Context Architecture Refactor
- Separated ContextManager (pure data provider) from ConversationManager (pure oracle)
- Moved all context decision logic to SpeechServices layer
- Implemented sliding window + summary for optimal context management

### System Prompt Improvements
- Removed duplicate system prompt from response.create (was being sent twice)
- Reformatted from bullet points to imperative sentences
- Set minimum temperature to 0.6 for better instruction following

### Summary Integration
- Fixed summary generation not triggering during real conversations
- Added summary update calls after assistant messages
- Implemented lossless context coverage (summary + recent messages)

## File Structure
```
core/
├── speech_services_realtime.py     # Main Realtime API implementation
├── streaming_chatbot_realtime.py   # Streaming chatbot interface
├── context_manager.py              # Pure context data provider
├── components.py                   # Tool implementations
├── state_management/
│   ├── session_summary.json        # AI-generated conversation summaries
│   └── app_state.json          # System state persistence
└── __init__.py

config.py                           # Complete system configuration
main.py                            # Main entry point
requirements.txt                    # Python dependencies
```

## Key Dependencies
- `websockets` - For Realtime API WebSocket connections
- `openai` - OpenAI API client
- `sounddevice` - Audio input/output
- `numpy` - Audio processing
- `google-auth` - Calendar integration

## Development Notes

### Context Management Best Practices
1. Always use ContextManager methods to get formatted context
2. Don't make context decisions in ContextManager - it's a pure data provider
3. Summary generation should only be triggered by SpeechServices layer
4. Test context with both short and long conversations

### Realtime API Specifics
- WebSocket connection required for full functionality
- Session-level instructions only (no message-level system prompts)
- Temperature must be ≥0.6 for proper instruction following
- Supports both streaming and chunk-based modes

### Common Debugging Steps
1. Check `REALTIME_API_DEBUG` output for WebSocket messages
2. Verify session_summary.json contains proper User_Summary and Response_Summary
3. Ensure context length is appropriate (summary + 6 recent messages)
4. Confirm system prompt is not being sent as message for Realtime API

This system provides a robust, low-latency voice assistant with intelligent context management and seamless integration with home automation tools.