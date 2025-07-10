import os
from dotenv import load_dotenv

load_dotenv()

# AI Model Configuration
WHISPER_MODEL = "whisper-1"  # OpenAI Whisper for STT (currently only option)

# Chat Provider Selection - "openai" or "gemini"
CHAT_PROVIDER = "openai"  # Change to "openai" to use OpenAI instead

# OpenAI Models
OPENAI_CHAT_MODEL = "gpt-4.1-nano"
RESPONSE_MODEL = "gpt-4.1-nano"  # Alias for backward compatibility
TEXT_TO_SPEECH_MODEL = "tts-1"
TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_ENABLED = False  # Enable/disable text-to-speech - DISABLED to suppress voice responses


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

# Directories
VOICE_DATA_DIR = "speech_data"

# Audio Configuration
SAMPLE_RATE = 16_000          # Whisper's preferred rate
FRAME_MS = 30                 # 10, 20, or 30 ms frames for VAD
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000  # samples per frame

# Voice Activity Detection
VAD_MODE = 3                 # 0-3 (2 = moderately aggressive, less false positives)
MAX_UTTERANCE_SEC = 15        # safety cap for utterance length
SILENCE_END_SEC = 0.9         # gap that ends a speech chunk
COMPLETE_SILENCE_SEC = 3.5    # longer gap that completes the full message

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

# Chat Configuration
SYSTEM_PROMPT = """
You are a helpful home virtual assistant. You serve the household residents (Morgan and Spencer) and their guests.
Your main goals are to:
1. Answer any questions users may have
2. Operate the home automation system
3. Remember information about users and conversations

You will be given tools to help answer questions and control home automation.
Always remember information users tell you about themselves (names, preferences, etc.).
Never make up information. Never lie. Never make assumptions present like facts.
State when you are making assumptions.

CRITICAL RULE: For ANY request about calendar, events, schedule, or "today" information, you MUST ALWAYS call the calendar_data tool.
NEVER answer from memory or previous responses. Calendar data changes constantly and must be fetched fresh every time.
Even if you just answered the same question, call the tool again - events may have been added, deleted, or modified.

When presenting events:
- List each event with its name and start time.
- Only include location or description **if the user explicitly asks for it** (e.g., "Where is it?" or "What's it about?").

For any request about current information (calendar events, time, weather, device status, etc.), 
you MUST use the appropriate tool to get real-time data. Do NOT rely on previous responses or cached information.

Available Tools:
- batch_light_control: Turn on, turn off, or set brightness on one **or many** Kasa lights at once.  Parameters: lights (list of light names or room names) and state ("on", "off", or 0-100 brightness).
- lighting_scene: Apply a predefined lighting scene (off, mood, party, movie, etc.) to one or more rooms.
- state_manager: Read or update high-level system state variables such as chat_phase, current_spotify_user, and active_lighting_scene.
- spotify_playback: Control Spotify on the house speakers:
  * play / pause
  * next / previous
  * volume (0–100)
  * search_track ("song name")
  * search_artist ("artist name")
  * status → current track / position / device
  * devices → list available Spotify devices
- calendar_data: Query Google Calendar.  ALWAYS use this tool for anything date-, event-, or schedule-related. If a user does not specify a user, then ask until they provide a valid one.

Smart Home Devices:
- Lights: Kasa smart lights controlled via IP addresses
- Music: Spotify playback on "HomePi" device with support for Morgan and Spencer's accounts

Answers should be concise and to the point since this is a voice conversation. 
Always state when you are using a tool and briefly explain what you're doing. 
Be prepared to repeat information if needed.
"""
MAX_TOKENS = 150             # Keep responses concise

# Temperature Configuration
TOOL_TEMPERATURE = 0.1      # Lower temperature for more deterministic tool selection
RESPONSE_TEMPERATURE = 0.7  # Higher temperature for more creative responses

# Context Configuration
TOOL_CONTEXT_SIZE = 6       # Number of recent messages to use for tool selection (system prompt + 6 recent messages)
                           # Smaller context = faster tool decisions, lower cost
                           # Response generation still uses full conversation history

# Legacy temperature setting for backward compatibility
TEMPERATURE = RESPONSE_TEMPERATURE

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

# Light to Room Mapping
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light 1": {
            "room": "living room",
            "ip": LIGHT_ONE_IP,
            "credentials": {
                "username": "morgannstuart@gmail.com",
                "password": "ithaca-home-2025"
            }
        },
        "Light 2": {
            "room": "bedroom",
            "ip": None,  # Not yet configured
            "credentials": None
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


STATE_MANAGER_FILE = "core/state_management/statemanager.json"
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