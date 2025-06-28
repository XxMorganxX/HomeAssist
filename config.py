# OpenAI Models
WHISPER_MODEL = "whisper-1"
RESPONSE_MODEL = "gpt-3.5-turbo"
TEXT_TO_SPEECH_MODEL = "tts-1"

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
COMPLETE_SILENCE_SEC = 5.0    # longer gap that completes the full message

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

Tools:
- light_on: Should be used when the user wants to turn on a light.
- light_off: Should be used when the user wants to turn off a light.

Answers should be concise and to the point since this is a voice conversation. 
State when you are using a tool. Be prepared to repeat information if needed.
"""
MAX_TOKENS = 150             # Keep responses concise
TEMPERATURE = 0.7            # Response randomness (0-1)



# Light Configuration
LIGHT_ONE_IP = "192.168.1.186"