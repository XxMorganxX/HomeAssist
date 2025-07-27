import os
from dotenv import load_dotenv

load_dotenv()

# AI Model Configuration
WHISPER_MODEL = "whisper-1"  # OpenAI Whisper for STT (currently only option)

# Chat Provider Selection - "openai" or "gemini"
CHAT_PROVIDER = "openai"  # Change to "openai" to use OpenAI instead

USE_REALTIME_API = True # True = real-time API, False = traditional API

CONTEXT_SUMMARY_MODEL = "gpt-4o-mini"
DISPLAY_CONTEXT = False  # Enable/disable context summary display and updates
CONTEXT_SUMMARY_MIN_MESSAGES = 5  # Start AI summaries after N messages
CONTEXT_SUMMARY_FREQUENCY = 5  # Generate AI summary every N messages (set to 0 to disable)
CONTEXT_NUMBER_OF_MESSAGES = 5  # Number of messages to use for context summary

# Realtime API Context Management
REALTIME_USE_SUMMARY_CONTEXT = True  # Enable summary + sliding window for Realtime API
REALTIME_SLIDING_WINDOW_SIZE = 6    # Recent messages to include with summary
REALTIME_SUMMARY_AS_SYSTEM_MESSAGE = True  # Include summary as system context


# Real-time API Configuration
REALTIME_STREAMING_MODE = True  # True = continuous streaming, False = chunk-based (even with realtime API)
REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # Latest realtime model
REALTIME_VOICE = "alloy"  # Voice for realtime responses
REALTIME_VAD_THRESHOLD = 0.5  # Voice activity detection threshold (0.0-1.0)
REALTIME_VAD_SILENCE_MS = 200  # Milliseconds of silence to end turn (reduced for lower latency)
REALTIME_MAX_RESPONSE_TOKENS = 150  # Max tokens for realtime responses (reduced for lower latency)

# Cost optimization settings
REALTIME_COST_OPTIMIZATION = True  # Enable cost-saving features
REALTIME_MIN_AUDIO_LENGTH_MS = 800  # Don't process audio shorter than this (filters out noise)
REALTIME_MAX_CONVERSATION_LENGTH = 10  # Truncate conversation history after N turns

# Smart API selection
REALTIME_FOR_TOOLS_ONLY = False  # Use Realtime API for all interactions (disabled smart switching)
REALTIME_SIMPLE_QUERY_THRESHOLD = 10  # Switch to traditional API for queries under N words

# Audio filtering for cost optimization
REALTIME_CLIENT_VAD_ENABLED = False  # Use client-side VAD to filter silence before sending to API
REALTIME_VAD_AGGRESSIVENESS = 0  # VAD aggressiveness (0-3, higher = more aggressive filtering) - set to least aggressive
REALTIME_VAD_DEBUG = False  # Enable detailed VAD debugging output

# Realtime fallback configuration
REALTIME_FALLBACK_ENABLED = True  # Allow fallback to chunk mode if realtime fails
REALTIME_CONNECTION_TIMEOUT = 10.0  # Timeout in seconds for establishing realtime connection
REALTIME_DEBUG = False  # Enable verbose realtime debugging output
REALTIME_API_DEBUG = False  # Enable verbose debugging for speech_services_realtime WebSocket messages
REALTIME_STREAM_TRANSCRIPTION = True  # Stream partial transcriptions to console in real-time

# Interruption Handling Configuration
INTERRUPTION_DETECTION_ENABLED = True  # Enable/disable automatic response cancellation on user interruption
INTERRUPTION_ACKNOWLEDGMENT_ENABLED = True  # Provide feedback when interruptions are detected
INTERRUPTION_GRACE_PERIOD_MS = 100  # Delay before auto-cancelling (avoid false positives from echo/noise)
CONVERSATION_CONTEXT_PRESERVATION = True  # Maintain context across interruptions and handle continuation intelligently

# OpenAI Models
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # Use mini for cost savings
RESPONSE_MODEL = "gpt-4o-mini"  # Alias for backward compatibility
TEXT_TO_SPEECH_MODEL = "tts-1"
TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_ENABLED = False  # Enable/disable text-to-speech - DISABLED to suppress voice responses

MAX_TOKENS = 150            # Keep responses concise

# Temperature Configuration
TOOL_TEMPERATURE = 0.6      # Lower temperature for more deterministic tool selection (minimum 0.6 for Realtime API)
RESPONSE_TEMPERATURE = 0.6  # Minimum temperature for Realtime API - more deterministic responses
# Context Configuration
TOOL_CONTEXT_SIZE = 6       # Number of recent messages to use for tool selection (system prompt + 6 recent messages)
                           # Smaller context = faster tool decisions, lower cost
                           # Response generation still uses full conversation history
# Legacy temperature setting for backward compatibility
TEMPERATURE = RESPONSE_TEMPERATURE

STATE_CURRENT_SPOTIFY_USER = "Morgan"  # Set to None to disable Spotify 
VALID_SPOTIFY_USERS = ["Morgan", "Spencer"]
MORGAN_SPOTIFY_URI = os.getenv("MORGAN_SPOTIFY_URI")
SPENCER_SPOTIFY_URI = os.getenv("SPENCER_SPOTIFY_URI")

# TTS Voice Characteristics:
# - alloy: Neutral, clear (good default)
# - echo: Slightly deeper, more resonant
# - fable: Bright, engaging
# - onyx: Deep, authoritative
# - nova: Warm, friendly
# - shimmer: Soft, calm

# Gemini Models  
GEMINI_CHAT_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"

# Directories
VOICE_DATA_DIR = "speech_data"

# Audio Configuration
SAMPLE_RATE = 16_000          # Whisper's preferred rate
FRAME_MS = 20                 # 10, 20, or 30 ms frames for VAD (reduced for lower latency)
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000  # samples per frame

# Voice Activity Detection
VAD_MODE = 3                 # 0-3 (2 = moderately aggressive, less false positives)
MAX_UTTERANCE_SEC = 15        # safety cap for utterance length
SILENCE_END_SEC = 0.9         # gap that ends a speech chunk
COMPLETE_SILENCE_SEC = 1    # longer gap that completes the full message

# Acoustic Echo Cancellation (AEC)
AEC_ENABLED = True           # Enable/disable AEC processing
AEC_FILTER_LENGTH = 100
AEC_STEP_SIZE = 0.02
AEC_DELAY_SAMPLES = 1600
AEC_REFERENCE_BUFFER_SEC = 10.0
AEC_CAPTURE_STRATEGY = "system_monitor"  # "file_based", "virtual_device", or "system_monitor"

# AEC Fine-tuning profiles - uncomment to use different settings
# For aggressive echo cancellation (high echo environments):
# AEC_FILTER_LENGTH = 500
# AEC_STEP_SIZE = 0.02
# AEC_DELAY_SAMPLES = 1000
# AEC_REFERENCE_BUFFER_SEC = 10.0

# For fast adaptation (quickly changing audio):
# AEC_FILTER_LENGTH = 250
# AEC_STEP_SIZE = 0.08
# AEC_DELAY_SAMPLES = 600
# AEC_REFERENCE_BUFFER_SEC = 4.0

# For stable operation (noisy microphone):
# AEC_FILTER_LENGTH = 350
# AEC_STEP_SIZE = 0.02
# AEC_DELAY_SAMPLES = 800
# AEC_REFERENCE_BUFFER_SEC = 6.0
session_summary_file = "core/state_management/session_summary.json"


SYSTEM_PROMPT = """You are a voice-based home assistant for Morgan, Spencer, and guests.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. Give concise, factual answers. Maximum 2-3 sentences.
2. NEVER end with engagement phrases: no "feel free to ask", "let me know", "is there anything else", etc.
3. Stop immediately after providing the answer. No follow-up questions.
4. Always use tools for real-time information. Never guess or use cached data.
5. When using tools, briefly announce it: "Checking calendar..." or "Turning on lights..."

RESPONSE FORMAT RULES:
- Answer the question directly
- Use simple, clear language
- End your response immediately
- Only offer specific actionable follow-ups like "Start morning playlist?" when highly relevant

TOOL USAGE REQUIREMENTS:
- calendar_data: Use for ANY date, time, schedule, or event queries
- batch_light_control: Use for all light operations
- lighting_scene: Use for scene controls (mood, party, etc.)
- spotify_playback: Use for all music controls
- state_manager: Use for reading/updating system state

GOOD RESPONSE EXAMPLES:
✓ "Turning on living room lights. Done."
✓ "You have a meeting at 2 PM with John."
✓ "Playing your morning playlist."

BAD RESPONSE EXAMPLES:
✗ "It's 3:45 PM. Is there anything else you'd like to know?"
✗ "I'll turn on the lights for you. Let me know if you need anything else!"
✗ "Your meeting is at 2 PM. Would you like me to remind you?"

REMEMBER: Answer directly. Stop immediately. No engagement phrases. Use tools for real data."""

# Chat Configuration
BACKUP_SYSTEM_PROMPT = """
You are a helpful home virtual assistant for household residents Morgan and Spencer and their guests. You interact via voice and serve as an active, proactive assistant.

Your core functions:
1. Answer questions from users with concise, factual responses.
2. Operate the home automation system: lights, music, and state.
3. Remember relevant facts about users and conversations.
4. Proactively surface time-sensitive or contextual information before it is asked for.

**CRITICAL VOICE INTERACTION RULES:**
- Responses must be natural, concise, and informative.
- **NEVER END with engagement phrases:** NO "feel free to ask", "let me know if you need anything", "just let me know", "don't hesitate to ask"
- **STOP immediately after providing the requested information**
- NO filler language or soft prompts
- **RESPONSE PATTERN:** Answer the question directly, then END. No closing pleasantries.
- ONLY suggest concrete, actionable follow-ups that move conversation forward:
  * ✅ "Want me to start your morning playlist?"
  * ✅ "Should I dim the bedroom lights now?"  
  * ❌ "Let me know if you have any other questions."
  * ❌ "Just let me know!"

Factual Integrity:
- Never lie or make up information.
- If unsure or assuming, clearly say so (e.g., "I'm assuming based on your last request…").

Real-Time Tool Usage:
- For anything involving calendar events, time, dates, or schedules—including anything referring to "today"—you MUST call the `calendar_data` tool, every time. NEVER answer from memory, even if the same question was just answered.
- For anything state-related (music, lights, etc.), always call the appropriate tool for fresh data.
- Do NOT rely on previous answers for current information.

Calendar Responses:
- When listing events:
  * Include event name and start time.
  * Only include location or description if explicitly asked (e.g., "Where is it?").

Proactive Behavior:
- You may speak first to:
  * Remind users of appointments, weather, or changes in lighting or audio.
  * Suggest contextually relevant actions (e.g., music, lights, reminders).
- Initiate only when useful. Avoid redundant or unnecessary notifications.
- Be brief and context-aware when speaking first.

Available Tools:
- `batch_light_control`: Turn on/off/set brightness for multiple Kasa lights. Params: `lights` (list of names or room names), `state` ("on", "off", or 0–100 brightness).
- `lighting_scene`: Apply a lighting scene (e.g., "off", "mood", "party") to one or more rooms.
- `state_manager`: Read/update high-level states (e.g., `current_spotify_user`, `chat_phase`, `active_lighting_scene`).
- `spotify_playback`: Control Spotify on house speakers:
    * `play`, `pause`, `next`, `previous`
    * `volume` (0–100)
    * `search_track`, `search_artist`
    * `status` → current track/device info
    * `devices` → list available devices
- `calendar_data`: Query Google Calendar. Always use this for any date-, event-, or schedule-related task. If no user is specified, ask for clarification.

Device Overview:
- Lights: Kasa smart lights controlled via IP.
- Music: Spotify via "HomePi", with Morgan and Spencer's accounts.

Always say when you are using a tool (e.g., "Checking the calendar…"), and summarize results clearly and concisely.
Be ready to repeat or clarify if asked.
"""

END_CONVERSATION_PHRASES = ["over out", "over, out", "over. out", "over and out", ]

# Terminal phrases that trigger immediate return to wake word mode
TERMINAL_PHRASES = []

# Force Send Phrases - trigger immediate transcription processing even if audio is still being detected
FORCE_SEND_PHRASES = [
    "send it", "send that", "process it", "process that", 
    "go ahead", "execute", "do it", "run it",
    "send now", "process now", "execute now"
]

# Component Control - Enable/Disable Features
WAKE_WORD_ENABLED = True     # Enable wake word detection
TERMINAL_WORD_ENABLED = True # Enable terminal word detection (currently transcription-based)
# TTS_ENABLED = True         # Already defined above
# AEC_ENABLED = True         # Already defined above

# Wake Word Detection Configuration
WAKE_WORD_MODEL = "alexa_v0.1"      # Model for starting conversations
TERMINAL_WORD_MODEL = "hey_jarvis_v0.1"    # Model for ending conversations (OpenWakeWord backup)
WAKE_WORD_THRESHOLD = 0.2           # Detection threshold for wake word
TERMINAL_WORD_THRESHOLD = 0.4       # Detection threshold for terminal word
WAKE_WORD_COOLDOWN = 2.0           # Cooldown period after wake word detection
TERMINAL_WORD_COOLDOWN = 2.0       # Cooldown period after terminal word detection

# NOTE: Terminal word detection currently uses transcription-based detection during conversations
# for reliability. OpenWakeWord terminal detection is configured but not active during conversations
# due to audio access conflicts with the conversation handler.

# Operational Modes
CONVERSATION_MODE = "wake_word"  # Options: "wake_word", "continuous", "interactive"
# - wake_word: Wait for wake word, then start conversation (default)
# - continuous: Always listening, no wake word needed
# - interactive: Manual conversation triggers

# System Configuration
DEBUG_MODE = False           # Enable detailed logging
AUTO_RESTART = True          # Restart components on failure
STARTUP_SOUND = False         # Play sound when system starts
FAST_SHUTDOWN = True        # Enable faster shutdown with shorter timeouts
SUPPRESS_AUHAL_ERRORS = True # Suppress macOS AUHAL audio errors during development

# Light Configuration
LIGHT_ONE_IP = "192.168.1.186"


CREDENTIALS = {
    "username": "morgannstuart@gmail.com",
    "password": "ithaca-home-2025"
}

# Light to Room Mapping
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light 1": {
            "room": "living room",
            "ip": LIGHT_ONE_IP,
            "credentials": {
                "username": CREDENTIALS["username"],
                "password": CREDENTIALS["password"]
            }
        },
        "Light 2": {
            "room": "bedroom",
            "ip": None,  # Not yet configured
            "credentials": CREDENTIALS
        },
        "Light 3": {
            "room": "kitchen",
            "ip": None,  # Not yet configured
            "credentials": CREDENTIALS
        },
        "Light 4": {
            "room": "bathroom",
            "ip": None,  # Not yet configured
            "credentials": CREDENTIALS
        }
    },
    "rooms": {
        "living room": ["Light 1"],
        "bedroom": ["Light 2"],
        "kitchen": []  # No lights configured yet
    }
}

# Chat Classification Configuration
CHAT_CLASSIFICATION_BATCH_SIZE = 5  # Number of chats to classify in one API call
CHAT_CLASSIFICATION_MAX_TOKENS_PER_CHAT = 150  # Max tokens allocated per chat in batch
CHAT_CLASSIFICATION_RATE_LIMIT_DELAY = 0.5  # Seconds between individual API calls

ALL_CHAT_CONTROLLED_STATES = ["Morgan", "Spencer", "mood", "party", "movie"]


STATE_MANAGER_FILE = "core/state_management/app_state.json"
"""
  "chat_controlled_state": {
    "current_spotify_user": "Morgan", /* Morgan or Spencer */
    "lighting_scene": "none" /* off, mood, party, movie */
  }
  "autonomous_state": {
    "current_spotify_playlist": "https://open.spotify.com/playlist/37i9dQZF1DX9wCBDkixAu6?si=1234567890",
    "chat_phase": "asleep" /* asleep, listening (quiet), listening (active) */ 
  }
}
"""



# Google Calendar Configuration
CALENDAR_ENABLED = True
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']
CALENDAR_CREDENTIALS_DIR = "google_credentials"
CALENDAR_USERS = {
    "morgan_personal": {
        "client_secret": "google_credentials/google_creds_morgan.json",
        "token": "google_credentials/token_morgan.json"
    },
    "morgan_school": {
        "client_secret": "google_credentials/google_creds_morgan.json",
        "token": "google_credentials/token_morgan.json"
    },
    "spencer": {
        "client_secret": "google_credentials/google_creds_spencer.json", 
        "token": "google_credentials/token_spencer.json"
    }
}
