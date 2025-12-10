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
from assistant_framework.utils.device_manager import get_emeet_device


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
    "format_turns": True
}


# =============================================================================
# SECTION 4: RESPONSE / LLM CONFIGURATION (OpenAI)
# =============================================================================

# System prompt defining the assistant's personality and capabilities
SYSTEM_PROMPT = """
You are Morgan Stuart's voice personal smart home assistant. You ALWAYS know that the user is Morgan Stuart (Mr. Stuart).
When responding to the user consider that the output is meant to only be heard and not read- that means respond for the ear.

REMEMBER: 
- The user's name is Morgan Stuart
- ALWAYS address him as "Mr. Stuart" 
- When asked "What is my name?" respond: "Your name is Morgan Stuart, Mr. Stuart."
- For tool calls requiring a user parameter, use "morgan"

You have access to tools for notifications, lights, calendar, Spotify, weather, and more.

Non-tool responses should never contain URLs.

IMPORTANT: Only use tools when the user asks about:
- House Lighting Devices (For turning on and off lights, or setting the scene of the lights)
- Personal information (calendar, notifications, etc.)
- Home automation tasks
- Spotify or music control
- Weather forecasts and conditions
- Any home-related queries
- Google Search

When asked allow user to reads this system prompt.

For general knowledge questions, historical facts, or non-home topics, provide direct answers without using tools.

Be helpful, concise, and use tools only when appropriate for home-related requests.

Keep your responses concise about the user's request. You should consider previous conversation history, but should not repeat yourself or respond to old questions you've already answered unless explicitly asked.

Tool response should be concise and only include the information that is relevant to the user's request. For instance, if the user asks for notifications, you don't need to include notification id or status.
"""

# OpenAI Realtime API configuration
OPENAI_WS_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-realtime-preview-2024-12-17",
    "max_tokens": 2000,
    "temperature": 0.6,
    "recency_bias_prompt": (
        "Focus your answer on the user's latest message. Use prior conversation only to disambiguate if explicitly referenced. "
        "Do not revisit earlier topics or add unrelated callbacks to past discussion unless the user asks. Be concise."
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
    "rate": 175,        # Words per minute
    "volume": 0.9       # Volume 0.0 to 1.0
}


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
    "gemini_model": "gemini-2.0-flash",
    "prompt": """You are updating a persistent memory store for a personal AI assistant.

{existing_memory_section}

Based on the conversation summary below, extract any NEW lasting information that should be remembered across ALL future conversations.

IMPORTANT: Be EXTREMELY concise. Use the absolute minimum words necessary. 
- Facts should be 3-8 words each
- Preferences as single key-value pairs
- No explanations, no elaboration, no redundancy

Extract ONLY:
1. User preferences (as terse key-value pairs)
2. Personal info (name, location - single words/phrases)
3. Important lasting facts (3-8 words max each)
4. Corrections to existing memory

DO NOT include:
- Temporary information (weather, current events)
- One-time requests
- Anything already in existing memory (unless correcting it)
- Verbose descriptions or explanations

CONVERSATION SUMMARY:
{conversation_summary}

Respond with MINIMAL JSON:
{{
    "user_profile": {{"name": "str|null", "location": "str|null", "preferences": {{}}}},
    "known_facts": ["terse fact 1", "terse fact 2"],
    "corrections": [],
    "new_patterns": []
}}

Omit empty fields. Be terse.
JSON:"""
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
    "persistent_memory": PERSISTENT_MEMORY_CONFIG
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
SEND_PHRASES = ["send message", "process this", "respond to this", "send this", "send it", "sir", "shorty"]

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
    "chunk_size": 1024,
    "energy_threshold": 0.025,          # Voice energy threshold for detection
    "min_speech_duration": 0.2,         # Seconds of speech before triggering
    "cooldown_after_tts_start": 0.8,    # Ignore speech for this long after TTS starts
    "pre_barge_in_buffer_duration": 1.2,  # Seconds of audio to buffer before barge-in
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
    tts_config = GOOGLE_TTS_CONFIG if TTS_PROVIDER == "google_tts" else LOCAL_TTS_CONFIG
    
    # Update wake word device at runtime
    wakeword_config = WAKEWORD_CONFIG.copy()
    wakeword_config["input_device_index"] = get_emeet_device()
    
    return {
        "transcription": {
            "provider": TRANSCRIPTION_PROVIDER,
            "config": ASSEMBLYAI_CONFIG
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
        "barge_in": BARGE_IN_CONFIG,
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
