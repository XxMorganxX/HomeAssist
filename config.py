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
TTS_ENABLED = False  # Enable/disable text-to-speech


STATE_CURRENT_SPOTIFY_USER = "Morgan" 
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
FRAME_MS = 30                 # 10, 20, or 30 ms frames for VAD
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000  # samples per frame

# Voice Activity Detection
VAD_MODE = 3                 # 0-3 (2 = moderately aggressive, less false positives)
MAX_UTTERANCE_SEC = 15        # safety cap for utterance length
SILENCE_END_SEC = 0.9         # gap that ends a speech chunk
COMPLETE_SILENCE_SEC = 3.5    # longer gap that completes the full message

# Acoustic Echo Cancellation (AEC)
AEC_ENABLED = True           # Enable/disable AEC processing
AEC_FILTER_LENGTH = 300      # NLMS filter length (200-500 typical)
AEC_STEP_SIZE = 0.05         # NLMS step size (0.01-0.1, smaller = more stable)
AEC_DELAY_SAMPLES = 800      # Estimated delay between speaker and mic (samples)
AEC_REFERENCE_BUFFER_SEC = 5.0  # How long to keep reference audio (seconds)
AEC_CAPTURE_STRATEGY = "file_based"  # "file_based", "virtual_device", or "system_monitor"

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

Available Tools:
- light_on: Turn on smart lights in any room. Requires room name and light name.
- light_off: Turn off smart lights in any room. Requires room name and light name.
- spotify_playback: Control Spotify music playback with these actions:
  * play/pause - Start or pause current playback
  * next/previous - Skip tracks forward or backward
  * volume - Set volume level (0-100)
  * search_track - Search for and play a specific song
  * search_artist - Play top tracks by an artist
  * status - Get current playback information
  * devices - List available Spotify devices

Smart Home Devices:
- Lights: Kasa smart lights controlled via IP addresses
- Music: Spotify playback on "HomePi" device with support for Morgan and Spencer's accounts

Answers should be concise and to the point since this is a voice conversation. 
Always state when you are using a tool and briefly explain what you're doing. 
Be prepared to repeat information if needed.
"""
MAX_TOKENS = 150             # Keep responses concise
TEMPERATURE = 0.7            # Response randomness (0-1)

END_CONVERSATION_PHRASES = ["over out", "over, out", "over. out", "over and out", ]

# Force Send Phrases - trigger immediate transcription processing even if audio is still being detected
FORCE_SEND_PHRASES = [
    "send it", "send that", "process it", "process that", 
    "go ahead", "execute", "do it", "run it",
    "send now", "process now", "execute now"
]

# Component Control - Enable/Disable Features
WAKE_WORD_ENABLED = True     # Enable wake word detection
# TTS_ENABLED = True         # Already defined above
# AEC_ENABLED = True         # Already defined above

# Operational Modes
CONVERSATION_MODE = "wake_word"  # Options: "wake_word", "continuous", "interactive"
# - wake_word: Wait for wake word, then start conversation (default)
# - continuous: Always listening, no wake word needed
# - interactive: Manual conversation triggers

# System Configuration
DEBUG_MODE = False           # Enable detailed logging
AUTO_RESTART = True          # Restart components on failure
STARTUP_SOUND = True         # Play sound when system starts

# Light Configuration
LIGHT_ONE_IP = "192.168.1.186"