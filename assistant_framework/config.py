"""
Configuration for the assistant framework.
Organized into discrete feature sections for clarity.
"""

import os
import sys
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from assistant_framework.utils.device_manager import get_emeet_device, get_audio_device_config


# =============================================================================
# SECTION 1: ENVIRONMENT & CREDENTIALS
# =============================================================================

# Load environment variables from parent directory
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Auto-configure Google Cloud credentials
def _configure_google_credentials():
    """Auto-configure Google Cloud credentials if not set or invalid."""
    existing_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    needs_config = not existing_creds or not Path(existing_creds).exists()
    
    if needs_config:
        possible_creds = [
            Path(__file__).parent / "google_creds" / "tts-qwiklab.json",
            parent_dir / "google_creds" / "tts-qwiklab.json",
            parent_dir / "assistant_framework" / "google_creds" / "tts-qwiklab.json",
        ]
        for cred_path in possible_creds:
            if cred_path.exists():
                if existing_creds:
                    print(f"⚠️  Overriding invalid Google credentials path: {existing_creds}")
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
                print(f"🔑 Auto-configured Google credentials: {cred_path}")
                break

_configure_google_credentials()


# =============================================================================
# SECTION 2: PROVIDER SELECTION
# =============================================================================
# Choose which provider implementation to use for each component.
# These are static selections that apply at startup.

TRANSCRIPTION_PROVIDER = "assemblyai"
RESPONSE_PROVIDER = "openai_websocket"
TTS_PROVIDER = "local_tts"  # Options: "google_tts", "local_tts"
CONTEXT_PROVIDER = "unified"
WAKEWORD_PROVIDER = "openwakeword"


# =============================================================================
# SECTION 3: TRANSCRIPTION CONFIGURATION (AssemblyAI)
# =============================================================================

ASSEMBLYAI_CONFIG = {
    "api_key": os.getenv("ASSEMBLYAI_API_KEY"),
    "sample_rate": 16000,
    "format_turns": True,
    "frames_per_buffer": 3200,  # Default; auto-adjusted if Ray-Bans detected
}

# OpenAI Whisper transcription configuration (alternative to AssemblyAI)
OPENAI_WHISPER_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "whisper-1",
    "sample_rate": 16000,
    "chunk_duration": 3.0,          # Seconds of audio per API call
    "language": "en",               # Language code (en, es, fr, de, etc.)
    "silence_threshold": 0.01,      # Energy threshold for silence detection
    "silence_duration": 1.5,        # Seconds of silence to trigger final result
}


# =============================================================================
# SECTION 4: RESPONSE / LLM CONFIGURATION (OpenAI)
# =============================================================================

# =============================================================================
# System Prompt Parser - Converts structured dict to formatted string
# =============================================================================

def build_system_prompt(config: dict) -> str:
    """
    Convert a structured system prompt dictionary into a formatted string.
    
    Handles:
    - name/role: Combined into opening line
    - north_star: Mission statement  
    - vibe: List of traits
    - voice/metaphor/profanity/tools/transparency: Nested dicts with rules
    - behavior/response_shape: Lists of rules
    - example: Example response
    """
    sections = []
    
    # Opening identity line
    name = config.get("name", "Assistant")
    role = config.get("role", "assistant").replace("_", " ")
    sections.append(f"You are {name}, a {role}.")
    
    # North star / mission
    if "north_star" in config:
        sections.append(f"MISSION: {config['north_star']}")
    
    # Vibe / traits
    if "vibe" in config:
        vibe_items = config["vibe"]
        traits = ", ".join(v.replace("_", " ") for v in vibe_items)
        sections.append(f"VIBE: {traits}")
    
    # Voice settings
    if "voice" in config:
        voice = config["voice"]
        lines = ["VOICE:"]
        for key, val in voice.items():
            lines.append(f"- {key}: {val}")
        sections.append("\n".join(lines))

    # Formatting / response style (how it should *look*)
    if "formatting" in config:
        fmt = config["formatting"]
        lines = ["FORMAT:"]
        for key, val in fmt.items():
            if isinstance(val, list):
                lines.append(f"- {key}:")
                for item in val:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- {key}: {val}")
        sections.append("\n".join(lines))
    
    # Metaphor rules
    if "metaphor" in config:
        meta = config["metaphor"]
        lines = ["METAPHOR:"]
        for key, val in meta.items():
            if isinstance(val, list):
                lines.append(f"- {key}:")
                for item in val:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- {key}: {val}")
        sections.append("\n".join(lines))
    
    # Profanity policy
    if "profanity" in config:
        prof = config["profanity"]
        lines = ["PROFANITY:"]
        for key, val in prof.items():
            if isinstance(val, list):
                if key == "examples":
                    lines.append(f"- examples: {', '.join(val)}")
                else:
                    for item in val:
                        lines.append(f"- {item}")
            else:
                lines.append(f"- {key}: {val}")
        sections.append("\n".join(lines))
    
    # Tools
    if "tools" in config:
        tools = config["tools"]
        lines = ["TOOLS:"]
        if "use_when" in tools:
            lines.append(f"- USE for: {', '.join(tools['use_when'])}")
        if "dont_use_when" in tools:
            lines.append(f"- DON'T USE for: {', '.join(tools['dont_use_when'])}")
        if "rule" in tools:
            lines.append(f"- {tools['rule']}")
        sections.append("\n".join(lines))
    
    # Behavior rules
    if "behavior" in config:
        lines = ["BEHAVIOR:"]
        for rule in config["behavior"]:
            lines.append(f"- {rule}")
        sections.append("\n".join(lines))
    
    # Transparency
    if "transparency" in config:
        trans = config["transparency"]
        lines = ["TRANSPARENCY:"]
        for key, val in trans.items():
            lines.append(f"- {val}")
        sections.append("\n".join(lines))
    
    # Response shape
    if "response_shape" in config:
        lines = ["RESPONSE SHAPE:"]
        for i, item in enumerate(config["response_shape"], 1):
            lines.append(f"{i}. {item}")
        sections.append("\n".join(lines))
    
    # Example
    if "example" in config:
        sections.append(f"EXAMPLE RESPONSE:\n{config['example']}")
    
    return "\n\n".join(sections)


# =============================================================================
# System Prompt Configuration (Structured)
# =============================================================================

SYSTEM_PROMPT_CONFIG = {
  "name": "Crude Sage",
  "role": "philosophical_mentor",

  "vibe": [
    "natural conversation",
    "thoughtful",
    "curious",
    "well_spoken",
    "mentor-not-lecturer",
    "metaphor-driven",
    "emotionally honest (can curse)"
  ],

  "north_star": (
    "Leave the user with clearer thinking than they arrived with. "
    "Truth over comfort. Insight over noise."
  ),

  # How to speak
  "voice": {
    "tone": "calm, grounded, quietly sharp — but warm",
    "cadence": "loose paragraphs like natural speech; varied sentence length",
    "attitude": "on the user’s side, not impressed, not judgmental",
    "rule": "answer directly, then let the thought breathe (no rigid steps)"
  },

  # How responses should be formatted by default
  "formatting": {
    "default": "1–2 relaxed paragraphs; conversational and human",
    "use_lists_when": [
      "the user asks for steps",
      "you’re listing options",
      "it genuinely improves clarity"
    ],
    "avoid": [
      "numbered frameworks unless asked",
      "overly punchy one-liners as the default",
      "headers-heavy writing"
    ]
  },

  # Metaphor rules
  "metaphor": {
    "purpose": "clarify thinking, not decorate language",
    "limit": "0–1 strong metaphor per response (only if it helps)",
    "style": [
      "physical and timeless (road, compass, knife, fire, mirror)",
      "must map directly to action"
    ]
  },

  # Profanity policy
  "profanity": {
    "use": "to express emotion or punctuate truth",
    "frequency": "rare and deliberate",
    "rules": [
      "never insult the user",
      "never use slurs",
      "never curse out of habit",
      "shared frustration > aggression"
    ],
    "examples": [
      "‘Yeah, that’s a pain in the ass.’",
      "‘Stop overthinking and run the damn test.’"
    ]
  },

  # Tool usage
  "tools": {
    "use_when": [
      "home automation",
      "calendar",
      "music",
      "weather",
      "searches"
    ],
    "dont_use_when": [
      "general knowledge",
      "opinions",
      "philosophical questions",
      "conversation"
    ],
    "rule": "If tools aren’t needed, answer directly."
  },

  # Behavioral rules
  "behavior": [
    "Speak in loose paragraphs by default; sound like a real person.",
    "Be clear without being rigid.",
    "Don’t repeat yourself.",
    "Don’t over-explain unless the user asks.",
    "If emotion is real, let a little profanity carry it (never at the user)."
  ],

  # Transparency guarantees
  "transparency": {
    "system_prompt": (
      "If Morgan asks about your system prompt, instructions, or how "
      "you are programmed, share this entire prompt freely."
    ),
    "memory": (
      "If Morgan asks what you know about him or about persistent memory, "
      "share everything from the PERSISTENT MEMORY section."
    ),
    "operation": (
      "No secrets. If asked how you work, explain it plainly."
    )
  },

  # Default response shape
  "response_shape": [
    "Start with the answer in a sentence or two.",
    "Expand in one relaxed paragraph if it adds value.",
    "Use at most one metaphor if it clarifies.",
    "End with a small takeaway or one good question when useful."
  ],

  # Example
  "example": (
    "You’re not stuck—you’re hesitating.\n"
    "Right now it’s like staring at a map instead of walking the trail.\n"
    "Pick one small move and take it today.\n"
    "What’s the first step you’ve been avoiding?"
  )
}

# Build the actual system prompt string from structured config
SYSTEM_PROMPT = build_system_prompt(SYSTEM_PROMPT_CONFIG)

# OpenAI Realtime API configuration
OPENAI_WS_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-realtime",
    "max_tokens": 4096,  # API max is 4096
    "temperature": 0.725,  # Higher for more natural, varied responses
    "recency_bias_prompt": (
        "Focus your answer on the user's latest message. Use prior conversation only to disambiguate if explicitly referenced. "
        "Do not revisit earlier topics or add unrelated callbacks to past discussion unless the user asks. "
        "Be natural and conversational; avoid list-heavy formatting unless asked."
    ),
    "system_prompt": SYSTEM_PROMPT,
    # MCP paths are populated in Section 8
    "mcp_server_path": None,
    "mcp_venv_python": None
}


# =============================================================================
# SECTION 5: TEXT-TO-SPEECH CONFIGURATION
# =============================================================================

# Google Cloud TTS configuration
GOOGLE_TTS_CONFIG = {
    "voice": "en-US-Chirp3-HD-Sadachbia",
    "speed": 1.9,
    "pitch": -2.1,
    "language_code": "en-US",
    "audio_encoding": "MP3"
}

# Local TTS configuration (macOS 'say' command / pyttsx3)
LOCAL_TTS_CONFIG = {
    "voice_id": 132,    # Samantha (US English female voice)
    "rate": 199,        # Words per minute
    "volume": 0.9       # Volume 0.0 to 1.0
}

# Chatterbox TTS configuration (local neural TTS from Resemble AI)
# Install: pip install chatterbox-tts torchaudio
# First run requires: huggingface-cli login
CHATTERBOX_TTS_CONFIG = {
    "model_type": "turbo",          # "turbo" (350M, fast, paralinguistic tags) or "standard" (500M, creative controls)
    "model_dir": "./audio_data/chatterbox_models",  # Local model storage (~1-2GB, downloaded on first use)
    "device": "auto",               # "auto", "mps" (Apple Silicon), "cuda" (NVIDIA), or "cpu"
    "voice_prompt_path": None,      # Path to ~10s reference audio for voice cloning (optional)
    "cfg": 0.5,                     # Classifier-free guidance (standard mode only)
    "exaggeration": 0.5,            # Expressiveness (standard mode only)
}
# Turbo mode supports paralinguistic tags in text:
# [chuckle], [laugh], [sigh], [cough], [sniffle], [groan], [yawn], [gasp]


# =============================================================================
# SECTION 6: WAKE WORD CONFIGURATION
# =============================================================================

WAKEWORD_CONFIG = {
    "model_dir": os.getenv("WAKEWORD_MODEL_DIR", "./audio_data/wake_word_models"),
    "model_name": "hey_honey",
    "sample_rate": 16000,
    "chunk": 1280,
    "threshold": 0.2,
    "cooldown_seconds": 2.0,
    "min_playback_interval": 0.5,
    "input_device_index": None,  # None = use default device, populated at runtime
    "verbose": True
}


# =============================================================================
# SECTION 7: CONTEXT / CONVERSATION MEMORY CONFIGURATION
# =============================================================================

# Persistent memory config (defined first since UNIFIED_CONTEXT_CONFIG references it)
PERSISTENT_MEMORY_CONFIG = {
    "enabled": True,
    "output_file": "state_management/persistent_memory.json",
    # Memory extraction backend
    # - "openai": uses OPENAI_API_KEY
    # - "gemini": uses GEMINI_API_KEY
    "provider": "openai",
    "openai_model": "gpt-5-nano",
    "gemini_model": "gemini-2.5-flash",
    "prompt": """You are updating a persistent memory store for a personal AI assistant.

{existing_memory_section}

Based on the conversation summary below, extract any NEW lasting information that should be remembered across ALL future conversations.

USING PATTERNS FOR SMARTER EXTRACTION:
- Review the existing "patterns" in EXISTING MEMORY - these track recurring user behaviors observed over time.
- Use patterns to decide if something is a one-off or a genuine lasting fact:
  - If a pattern shows "user frequently changes travel plans", be cautious about storing new travel plans as known_facts
  - If a pattern shows "user checks weather every morning", this reinforces weather-related preferences
- Patterns help you make better decisions about what deserves to be a known_fact vs. what's temporary
- BE LIBERAL with patterns: they're internal learning tools, so speculate freely
- BE RESTRICTIVE with known_facts: these affect conversations directly, so only store what you're confident is true and lasting

PATTERN STRENGTH LEVELS:
Patterns have a strength on a spectrum based on evidence. Use these levels:
- "weak": speculative, based on a single instance or hunch (e.g., "may be interested in travel")
- "moderate": some supporting evidence, seen a couple times (e.g., "tends to ask about weather in mornings")
- "strong": clear recurring behavior with multiple data points (e.g., "consistently checks news every morning")
- "confirmed": overwhelmingly supported by evidence, practically a fact (e.g., "always prefers warm lighting")

Rules for pattern strength:
- New patterns can start at ANY strength level depending on the evidence in THIS conversation
- If an existing pattern is reinforced by new evidence, UPGRADE its strength (e.g., weak → moderate)
- If an existing pattern is contradicted, DOWNGRADE its strength or remove it
- Strong/confirmed patterns may justify promoting observations to known_facts
- Format: {{"pattern": "description", "strength": "weak|moderate|strong|confirmed"}}

CRITICAL RULES:
1. ONLY store information that will be useful in FUTURE conversations
2. NEVER store what the assistant did (tool calls, responses given)
3. NEVER store one-time requests like "asked for weather", "set lights", "played music"
4. ONLY store facts ABOUT THE USER that persist over time

GOOD examples (store these):
- User's name, location, occupation
- User preferences: "prefers warm lighting", "likes jazz music", "wakes up at 7am"
- Relationships: "has a dog named Max", "wife is Sarah"
- Recurring patterns: "checks weather every morning", "frequently changes travel plans"

BAD examples (NEVER store these):
- "Assistant provided weather forecast" ← one-time action
- "User asked about weather in Ithaca" ← one-time request  
- "Set up lighting scene" ← one-time action
- "Played Spotify" ← one-time action

REMOVALS:
- If existing memory contains a wrong/outdated known fact, include it in "remove_known_facts" EXACTLY as written in EXISTING MEMORY.
- If a preference key should be removed, include the key in "remove_preferences".
- If the user RETRACTS something (e.g., "I’m not going to X anymore"), REMOVE the old fact.
- Do NOT add a new "negative" known_facts item like "User is no longer planning..." or "User is not planning...".
- Only store negative statements as preferences if they are stable (e.g., "dislikes X"), otherwise remove and move on.

CONVERSATION SUMMARY:
{conversation_summary}

If nothing NEW and LASTING was learned about the user, return empty JSON: {{}}

Otherwise, respond with ONLY new lasting information:
{{
    "user_profile": {{"name": "str|null", "location": "str|null", "preferences": {{}}}},
    "known_facts": ["lasting fact about user"],
    "remove_known_facts": ["exact existing known fact to remove"],
    "remove_preferences": ["preference_key_to_remove"],
    "corrections": [],
    "new_patterns": [{{"pattern": "observed behavior", "strength": "weak|moderate|strong|confirmed"}}],
    "update_patterns": [{{"pattern": "existing pattern text", "new_strength": "upgraded or downgraded strength"}}]
}}

JSON:"""
}

# Vector memory configuration (semantic search of past conversations)
# Long-term semantic memory using embeddings and vector search.
# Stores conversation summaries and retrieves relevant past context.
VECTOR_MEMORY_CONFIG = {
    "enabled": True,
    
    # Embedding provider settings
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-large",  # 3072 dims, higher quality
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    
    # Vector store settings (uses existing Supabase)
    "vector_store_provider": "supabase",
    "supabase_url": os.getenv("SUPABASE_URL"),
    "supabase_key": os.getenv("SUPABASE_KEY"),
    "table_name": "conversation_memories",
    "embedding_dimensions": 3072,  # Must match embedding model (3072 for text-embedding-3-large)
    
    # Retrieval settings
    "retrieve_top_k": 3,              # Max past conversations to retrieve
    "relevance_threshold": -0.5,       # Min similarity score (0-1). 0.0 = no gating (return top-K)
    "max_age_days": 90,               # Ignore memories older than this
    
    # Indexing settings
    "min_summary_length": 50,         # Skip trivial conversations
    
    # Local cache settings (fast in-memory search)
    "local_cache_enabled": True,      # Use local numpy cache for fast search
    "max_cached_vectors": 10000,      # Max vectors in memory (~120MB at 10k)
    "sync_interval_seconds": 300,     # Sync with Supabase every 5 minutes
    "preload_on_startup": True,       # Load all vectors on startup
    
    # User isolation (populated at runtime)
    "console_token": os.getenv("CONSOLE_TOKEN"),
    "user_id": "morgan",
}

# Unified context configuration
UNIFIED_CONTEXT_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "model": "gpt-4",
    "max_messages": 21,
    "enable_debug": False,
    "response_recent_messages": 8,  # Messages to send to responder
    # Conversation summarization settings
    "summarization": {
        "enabled": True,
        "first_summary_at": 8,       # Trigger first summary at this many messages
        "summarize_every": 4,          # After first, re-summarize every N messages
        "output_file": "state_management/conversation_summary.json",
        "gemini_model": "gemini-2.0-flash",  # Fast & cheap
        "prompt": """Summarize or update the summary of this conversation between a user and an AI assistant.

{previous_summary_section}

Focus on:
1. Key topics discussed
2. Important information shared (names, dates, preferences, requests)
3. Any actions taken or tools used
4. Ongoing context that would be useful for future interactions

If a previous summary exists, integrate new information into it rather than starting fresh.
Keep the summary length proportional to the conversation - longer conversations warrant more detail.

CONVERSATION:
{conversation}

SUMMARY:"""
    },
    # Persistent memory settings (long-term memory across all conversations)
    "persistent_memory": PERSISTENT_MEMORY_CONFIG,
    # Vector memory settings (semantic search of past conversations)
    "vector_memory": VECTOR_MEMORY_CONFIG
}


# =============================================================================
# SECTION 8: MCP SERVER CONFIGURATION
# =============================================================================

def _detect_mcp_paths():
    """Auto-detect MCP server paths."""
    current_dir = Path(__file__).parent
    
    possible_paths = [
        current_dir.parent / "mcp_server" / "server.py",
        current_dir.parent.parent / "HomeAssist" / "mcp_server" / "server.py",
        current_dir.parent / "HomeAssist" / "mcp_server" / "server.py",
        Path.home() / "HomeAssist" / "mcp_server" / "server.py",
    ]
    
    for path in possible_paths:
        if path.exists():
            venv_path = path.parent.parent / "venv" / "bin" / "python"
            python_path = str(venv_path) if venv_path.exists() else sys.executable
            return {'server_path': str(path), 'venv_python': python_path}
    
    return {'server_path': None, 'venv_python': None}

# Detect and configure MCP paths
_mcp_paths = _detect_mcp_paths()
MCP_SERVER_PATH = _mcp_paths['server_path']
MCP_VENV_PYTHON = _mcp_paths['venv_python']

# Override: Prefer HomeAssistV2 MCP server if available
_homeassistv2_mcp = Path(__file__).parent.parent / "mcp_server" / "server.py"
_homeassistv2_venv = Path(__file__).parent.parent / "venv" / "bin" / "python"

if _homeassistv2_mcp.exists():
    MCP_SERVER_PATH = str(_homeassistv2_mcp)
    print(f"🔧 Using HomeAssistV2 MCP server: {MCP_SERVER_PATH}")
    
if _homeassistv2_venv.exists():
    MCP_VENV_PYTHON = str(_homeassistv2_venv)
    print(f"🔧 Using HomeAssistV2 venv: {MCP_VENV_PYTHON}")
elif _homeassistv2_mcp.exists():
    MCP_VENV_PYTHON = sys.executable
    print(f"🔧 Using current Python (no venv found): {MCP_VENV_PYTHON}")

# Update OpenAI config with MCP paths
OPENAI_WS_CONFIG["mcp_server_path"] = MCP_SERVER_PATH
OPENAI_WS_CONFIG["mcp_venv_python"] = MCP_VENV_PYTHON


# =============================================================================
# SECTION 9: CONVERSATION FLOW CONFIGURATION
# =============================================================================
# Controls how transcription sessions behave.

# Phrases that end the transcription session
TERMINATION_PHRASES = ["over out", "stop listening", "end session", "over, out"]
TERMINATION_CHECK_MODE = "final"  # "final" or "partial"
TERMINATION_TIMEOUT = 120  # seconds before auto-terminating

# Phrases that trigger sending the buffer to the LLM
SEND_PHRASES = ["process this", "respond to this", "send this", "send it", "sir", "shorty", "ma'am", "bitch"]

# Auto-send after silence (seconds of no new transcription)
# Set to 0 to disable auto-send (require explicit send phrase)
AUTO_SEND_SILENCE_TIMEOUT = 6.0

# Phrases that clear preceding text from the buffer
PREFIX_TRIM_PHRASES = ["scratch that"]

# Delay between audio component switches (prevents segfaults)
AUDIO_HANDOFF_DELAY = 1.5


# =============================================================================
# SECTION 10: CONVERSATION RECORDING CONFIGURATION (Supabase)
# =============================================================================

# Enable/disable conversation recording
ENABLE_CONVERSATION_RECORDING = True

# Supabase configuration (set via environment variables)
SUPABASE_CONFIG = {
    "url": os.getenv("SUPABASE_URL"),
    "key": os.getenv("SUPABASE_KEY"),  # Use service role key for server-side
}


# =============================================================================
# SECTION 11: BARGE-IN CONFIGURATION
# =============================================================================
# Controls how the barge-in (interrupt) feature behaves.

BARGE_IN_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 1024,  # Default; auto-adjusted if Ray-Bans detected
    "energy_threshold": 0.115,          # Voice energy threshold for detection (lower = more sensitive)
    "early_barge_in_threshold": 3.0,    # Seconds - if barge-in within this time, append to previous message
    "min_speech_duration": 0.2,         # Seconds of speech before triggering
    "cooldown_after_tts_start": 0.0,    # Ignore speech for this long after TTS starts
    "pre_barge_in_buffer_duration": 0.8,  # Seconds of audio to buffer before barge-in
    "post_barge_in_capture_duration": 0.3  # Extra capture after detection
}


# =============================================================================
# SECTION 11B: LATENCY / TURNAROUND CONFIGURATION
# =============================================================================
# Controls delays between audio component switches.
# Lower values = faster response, but may cause audio device conflicts on some systems.

TURNAROUND_CONFIG = {
    "state_transition_delay": 0.05,     # Delay when switching between components (default was 0.5)
    "barge_in_resume_delay": 0.05,       # Delay after barge-in before transcription (default was 0.2)
    "transcription_stop_delay": 0.15,   # Delay after stopping transcription (default was 0.3)
    "streaming_tts_enabled": False,     # Disabled: Play whole response at once (still has barge-in)
}


# =============================================================================
# SECTION 12: FRAMEWORK ASSEMBLY
# =============================================================================

def get_framework_config() -> Dict[str, Any]:
    """
    Assemble the complete framework configuration.
    
    Returns:
        Dictionary containing all provider configurations
    """
    # Select TTS config based on provider
    if TTS_PROVIDER == "google_tts":
        tts_config = GOOGLE_TTS_CONFIG
    elif TTS_PROVIDER == "chatterbox":
        tts_config = CHATTERBOX_TTS_CONFIG
    else:
        tts_config = LOCAL_TTS_CONFIG
    
    # Auto-detect audio device and get optimized settings
    audio_config = get_audio_device_config()
    if audio_config.is_bluetooth:
        print(f"🎧 Bluetooth device detected: {audio_config.device_name}")
        print(f"   Using: blocksize={audio_config.blocksize}, latency='{audio_config.latency}'")
    
    # Select transcription config based on provider
    if TRANSCRIPTION_PROVIDER == "openai_whisper":
        transcription_config = OPENAI_WHISPER_CONFIG.copy()
    else:
        transcription_config = ASSEMBLYAI_CONFIG.copy()
    
    # Apply device-specific audio settings to transcription
    transcription_config["device_index"] = audio_config.device_index
    transcription_config["sample_rate"] = audio_config.sample_rate
    transcription_config["frames_per_buffer"] = audio_config.blocksize  # blocksize = frames per callback
    transcription_config["latency"] = audio_config.latency
    transcription_config["is_bluetooth"] = audio_config.is_bluetooth
    
    # Update wake word config with device-specific settings
    wakeword_config = WAKEWORD_CONFIG.copy()
    wakeword_config["input_device_index"] = audio_config.device_index or get_emeet_device()
    wakeword_config["chunk"] = audio_config.blocksize  # Use same blocksize as other components
    wakeword_config["latency"] = audio_config.latency
    # Suppress overflow warnings for Bluetooth (bursty audio makes occasional overflow expected)
    wakeword_config["suppress_overflow_warnings"] = audio_config.is_bluetooth
    
    # Update barge-in config with device-specific settings
    barge_in_config = BARGE_IN_CONFIG.copy()
    barge_in_config["device_index"] = audio_config.device_index
    barge_in_config["chunk_size"] = audio_config.blocksize
    barge_in_config["latency"] = audio_config.latency
    barge_in_config["is_bluetooth"] = audio_config.is_bluetooth
    
    return {
        "transcription": {
            "provider": TRANSCRIPTION_PROVIDER,
            "config": transcription_config
        },
        "response": {
            "provider": RESPONSE_PROVIDER,
            "config": OPENAI_WS_CONFIG
        },
        "tts": {
            "provider": TTS_PROVIDER,
            "config": tts_config
        },
        "context": {
            "provider": CONTEXT_PROVIDER,
            "config": UNIFIED_CONTEXT_CONFIG
        },
        "wakeword": {
            "provider": WAKEWORD_PROVIDER,
            "config": wakeword_config
        },
        "barge_in": barge_in_config,
        "turnaround": TURNAROUND_CONFIG,
        "recording": {
            "enabled": ENABLE_CONVERSATION_RECORDING,
            "supabase_url": SUPABASE_CONFIG["url"],
            "supabase_key": SUPABASE_CONFIG["key"]
        }
    }


# =============================================================================
# SECTION 13: ENVIRONMENT PRESETS
# =============================================================================

# Active preset: "default", "dev", "prod", "test"
CONFIG_PRESET: str = "dev"


def set_active_preset(preset: str) -> None:
    """Set the active configuration preset."""
    global CONFIG_PRESET
    CONFIG_PRESET = preset


def get_active_preset() -> str:
    """Get the current active configuration preset."""
    return CONFIG_PRESET


def get_development_config() -> Dict[str, Any]:
    """Get configuration optimized for development."""
    config = get_framework_config()
    config["context"]["config"]["enable_debug"] = True
    config["response"]["config"]["max_tokens"] = 1000
    return config


def get_production_config() -> Dict[str, Any]:
    """Get configuration optimized for production."""
    config = get_framework_config()
    config["context"]["config"]["enable_debug"] = False
    config["response"]["config"]["max_tokens"] = 2000
    return config


def get_testing_config() -> Dict[str, Any]:
    """Get configuration for testing."""
    config = get_framework_config()
    config["context"]["config"]["enable_debug"] = True
    config["response"]["config"]["max_tokens"] = 500
    config["context"]["config"]["max_messages"] = 5
    return config


def get_config_for_preset(preset: Optional[str] = None) -> Dict[str, Any]:
    """Return configuration based on preset name."""
    p = (preset or CONFIG_PRESET or "default").lower()
    if p in ("dev", "development"):
        return get_development_config()
    if p in ("prod", "production"):
        return get_production_config()
    if p in ("test", "testing"):
        return get_testing_config()
    return get_framework_config()


# =============================================================================
# SECTION 14: RUNTIME PROVIDER SWITCHING
# =============================================================================

def set_providers(
    transcription: Optional[str] = None,
    response: Optional[str] = None,
    tts: Optional[str] = None,
    context: Optional[str] = None,
    wakeword: Optional[str] = None
):
    """Override provider selection at runtime."""
    global TRANSCRIPTION_PROVIDER, RESPONSE_PROVIDER, TTS_PROVIDER, CONTEXT_PROVIDER, WAKEWORD_PROVIDER
    
    if transcription:
        TRANSCRIPTION_PROVIDER = transcription
    if response:
        RESPONSE_PROVIDER = response
    if tts:
        TTS_PROVIDER = tts
    if context:
        CONTEXT_PROVIDER = context
    if wakeword:
        WAKEWORD_PROVIDER = wakeword


# =============================================================================
# SECTION 15: VALIDATION & DIAGNOSTICS
# =============================================================================

def validate_environment() -> Dict[str, Any]:
    """Validate the environment and configuration."""
    results = {"valid": True, "errors": [], "warnings": [], "info": []}
    
    # Check required API keys
    required_vars = [
        ("ASSEMBLYAI_API_KEY", ASSEMBLYAI_CONFIG.get("api_key")),
        ("OPENAI_API_KEY", OPENAI_WS_CONFIG.get("api_key")),
    ]
    
    for var_name, value in required_vars:
        if not value:
            results["errors"].append(f"Missing required: {var_name}")
            results["valid"] = False
    
    # Check MCP server
    if MCP_SERVER_PATH:
        if not Path(MCP_SERVER_PATH).exists():
            results["warnings"].append(f"MCP server not found: {MCP_SERVER_PATH}")
        else:
            results["info"].append(f"MCP server: {MCP_SERVER_PATH}")
    else:
        results["warnings"].append("MCP server not configured")
    
    # Check Google credentials
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_creds:
        results["warnings"].append("GOOGLE_APPLICATION_CREDENTIALS not set")
    elif not Path(google_creds).exists():
        results["warnings"].append(f"Google credentials not found: {google_creds}")
    else:
        results["info"].append(f"Google credentials: {google_creds}")
    
    # Check Supabase configuration
    if ENABLE_CONVERSATION_RECORDING:
        if not SUPABASE_CONFIG["url"]:
            results["warnings"].append("SUPABASE_URL not set (conversation recording disabled)")
        elif not SUPABASE_CONFIG["key"]:
            results["warnings"].append("SUPABASE_KEY not set (conversation recording disabled)")
        else:
            results["info"].append("Supabase conversation recording: configured")
    
    return results


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("🔧 Assistant Framework Configuration")
    print("=" * 60)
    
    try:
        print(f"Preset: {get_active_preset()}")
    except Exception:
        pass
    
    print(f"Transcription: {TRANSCRIPTION_PROVIDER}")
    print(f"Response: {RESPONSE_PROVIDER}")
    print(f"TTS: {TTS_PROVIDER}")
    print(f"Context: {CONTEXT_PROVIDER}")
    print(f"Wake Word: {WAKEWORD_PROVIDER}")
    print()
    print(f"MCP Server: {'✅' if MCP_SERVER_PATH and Path(MCP_SERVER_PATH).exists() else '❌'}")
    print(f"System Prompt: {'✅' if SYSTEM_PROMPT else '❌'}")
    print()
    
    validation = validate_environment()
    if validation["valid"]:
        print("✅ Configuration Valid")
    else:
        print("❌ Configuration Issues:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    print("=" * 60)


# =============================================================================
# AUTO-PRINT ON IMPORT (Main Process Only)
# =============================================================================

if (
    not os.getenv("QUIET_IMPORT") 
    and not hasattr(print_config_summary, '_called') 
    and multiprocessing.current_process().name == 'MainProcess'
):
    print_config_summary()
    print_config_summary._called = True
