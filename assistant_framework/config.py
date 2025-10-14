"""
Configuration for the assistant framework.
Supports static provider selection with provider-specific configurations.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from assistant_framework.utils.device_manager import get_emeet_device


# Load environment variables from parent directory if available
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Auto-configure Google Cloud credentials if not set or if the set path doesn't exist
existing_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
needs_config = (
    not existing_creds or 
    not Path(existing_creds).exists()
)

if needs_config:
    # Try common credential file locations
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


# =============================================================================
# PROVIDER SELECTION (Static, set at startup)
# =============================================================================

TRANSCRIPTION_PROVIDER = "assemblyai"
RESPONSE_PROVIDER = "openai_websocket"
TTS_PROVIDER = "google_tts"
CONTEXT_PROVIDER = "unified"
WAKEWORD_PROVIDER = "openwakeword"


# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Termination configuration for transcription sessions
TERMINATION_PHRASES = ["over out", "stop listening", "end session", "over, out"]
TERMINATION_CHECK_MODE = "final"  # "final" or "partial" - when to check for termination
TERMINATION_TIMEOUT = 120  # seconds before auto-terminating transcription
AUDIO_HANDOFF_DELAY = 0.5  # seconds to wait between audio component switches

# Send phrases that trigger sending transcription buffer to response component
SEND_PHRASES = ["send message", "process this", "respond to this", "send this", "send it", "sir"]

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

# Phrases that, if detected in the transcription buffer, will cause all text
# up to and including the phrase itself to be removed before sending to the LLM.
# Matching is case-insensitive and respects word boundaries.
# Example: ["actually", "scratch that", "no wait", "forget that"]
PREFIX_TRIM_PHRASES = ["scratch that"]


# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

# Automatically detect MCP server path
def _detect_mcp_paths():
    """Auto-detect MCP server paths."""
    current_dir = Path(__file__).parent
    
    # Look for HomeAssist directory
    possible_paths = [
        current_dir.parent.parent / "HomeAssist" / "mcp_server" / "server.py",
        current_dir.parent / "HomeAssist" / "mcp_server" / "server.py",
        Path.home() / "HomeAssist" / "mcp_server" / "server.py",
    ]
    
    for path in possible_paths:
        if path.exists():
            return {
                'server_path': str(path),
                'venv_python': str(path.parent.parent / "venv" / "bin" / "python")
            }
    
    return {
        'server_path': None,
        'venv_python': None
    }

_mcp_paths = _detect_mcp_paths()
MCP_SERVER_PATH = _mcp_paths['server_path']
MCP_VENV_PYTHON = _mcp_paths['venv_python']


# =============================================================================
# PROVIDER-SPECIFIC CONFIGURATIONS
# =============================================================================

# AssemblyAI Configuration
ASSEMBLYAI_CONFIG = {
    "api_key": os.getenv("ASSEMBLYAI_API_KEY"),
    "sample_rate": 16000,
    "format_turns": True
}

# OpenAI WebSocket Configuration  
OPENAI_WS_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o-realtime-preview-2024-12-17",
    "max_tokens": 2000,
    # Encourage concise, recent-focus answers
    "temperature": 0.6,
    "recency_bias_prompt": (
        "Focus your answer on the user's latest message. Use prior conversation only to disambiguate if explicitly referenced. "
        "Do not revisit earlier topics or add unrelated callbacks to past discussion unless the user asks. Be concise."
    ),
    "system_prompt": SYSTEM_PROMPT,
    "mcp_server_path": MCP_SERVER_PATH,
    "mcp_venv_python": MCP_VENV_PYTHON
}

# Google TTS Configuration
GOOGLE_TTS_CONFIG = {
    "voice": "en-US-Chirp3-HD-Sadachbia",
    "speed": 1.9,
    "pitch": -2.1,
    "language_code": "en-US",
    "audio_encoding": "MP3"
}

# Unified Context Configuration
UNIFIED_CONTEXT_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "model": "gpt-4",
    "max_messages": 21,
    "enable_debug": False,
    # Number of most recent messages to send to the responder (system prompt is provided separately)
    "response_recent_messages": 8
}


# =============================================================================
# FRAMEWORK CONFIGURATION
# =============================================================================

def get_framework_config() -> Dict[str, Any]:
    """
    Get the complete framework configuration.
    
    Returns:
        Dictionary containing all provider configurations
    """
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
            "config": GOOGLE_TTS_CONFIG
        },
        "context": {
            "provider": CONTEXT_PROVIDER,
            "config": UNIFIED_CONTEXT_CONFIG
        },
        "wakeword": {
            "provider": WAKEWORD_PROVIDER,
            "config": {
                "model_dir": "./audio_data/wake_word_models",
                "model_name": "alexa_v0.1",
                "sample_rate": 16000,
                "chunk": 1280,
                "threshold": 0.2,
                "cooldown_seconds": 2.0,
                "min_playback_interval": 0.5,
                "input_device_index": get_emeet_device(),  # ← set this to your mic index
                "verbose": False,
            }
        }
    }


# =============================================================================
# VALIDATION AND UTILITIES
# =============================================================================

def validate_environment() -> Dict[str, Any]:
    """
    Validate the environment and configuration.
    
    Returns:
        Dictionary containing validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # Check required environment variables
    required_env_vars = [
        ("ASSEMBLYAI_API_KEY", ASSEMBLYAI_CONFIG.get("api_key")),
        ("OPENAI_API_KEY", OPENAI_WS_CONFIG.get("api_key")),
    ]
    
    for var_name, value in required_env_vars:
        if not value:
            results["errors"].append(f"Missing required environment variable: {var_name}")
            results["valid"] = False
    
    # Check MCP server paths
    if MCP_SERVER_PATH:
        if not Path(MCP_SERVER_PATH).exists():
            results["warnings"].append(f"MCP server not found at: {MCP_SERVER_PATH}")
        else:
            results["info"].append(f"MCP server found at: {MCP_SERVER_PATH}")
    else:
        results["warnings"].append("MCP server path not configured - tools will not be available")
    
    # Check Google Cloud credentials
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_creds:
        results["warnings"].append("GOOGLE_APPLICATION_CREDENTIALS not set - may need service account key")
    elif not Path(google_creds).exists():
        results["warnings"].append(f"Google credentials file not found: {google_creds}")
    else:
        results["info"].append(f"Google credentials found: {google_creds}")
    
    return results


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("🔧 Assistant Framework Configuration")
    print("=" * 60)
    # Show active preset
    try:
        active = get_active_preset()
        print(f"Preset: {active}")
    except Exception:
        pass
    print(f"Transcription Provider: {TRANSCRIPTION_PROVIDER}")
    print(f"Response Provider: {RESPONSE_PROVIDER}")
    print(f"TTS Provider: {TTS_PROVIDER}")
    print(f"Context Provider: {CONTEXT_PROVIDER}")
    print()
    print(f"MCP Server: {'✅ Found' if MCP_SERVER_PATH and Path(MCP_SERVER_PATH).exists() else '❌ Not found'}")
    print(f"System Prompt: {'✅ Set' if SYSTEM_PROMPT else '❌ Not set'}")
    print()
    
    # Validation results
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
    
    if validation["info"]:
        print("\nℹ️  Info:")
        for info in validation["info"]:
            print(f"  - {info}")
    
    print("=" * 60)


# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

def get_development_config() -> Dict[str, Any]:
    """Get configuration optimized for development."""
    config = get_framework_config()
    
    # Enable debug mode
    config["context"]["config"]["enable_debug"] = True
    
    # Reduce max tokens for faster responses
    config["response"]["config"]["max_tokens"] = 1000
    
    return config


def get_production_config() -> Dict[str, Any]:
    """Get configuration optimized for production."""
    config = get_framework_config()
    
    # Disable debug mode
    config["context"]["config"]["enable_debug"] = False
    
    # Use full token limit
    config["response"]["config"]["max_tokens"] = 2000
    
    return config


def get_testing_config() -> Dict[str, Any]:
    """Get configuration for testing (with mock providers if needed)."""
    config = get_framework_config()
    
    # Enable debug mode for testing
    config["context"]["config"]["enable_debug"] = True
    
    # Reduce limits for faster testing
    config["response"]["config"]["max_tokens"] = 500
    config["context"]["config"]["max_messages"] = 5
    
    return config


# =============================================================================
# RUNTIME CONFIGURATION SWITCHING
# =============================================================================

def set_providers(transcription: Optional[str] = None,
                 response: Optional[str] = None,
                 tts: Optional[str] = None,
                 context: Optional[str] = None,
                 wakeword: Optional[str] = None):
    """
    Override provider selection at runtime.
    
    Args:
        transcription: Transcription provider name
        response: Response provider name
        tts: TTS provider name
        context: Context provider name
    """
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
# PRESET SELECTION (Static)
# =============================================================================

# Set the active configuration preset here. Options: "default", "dev", "prod", "test"
CONFIG_PRESET: str = "dev" #os.getenv("AF_CONFIG_PRESET", "prod")


def set_active_preset(preset: str) -> None:
    """Set the active configuration preset statically at runtime."""
    global CONFIG_PRESET
    CONFIG_PRESET = preset


def get_active_preset() -> str:
    """Get the current active configuration preset."""
    return CONFIG_PRESET


def get_config_for_preset(preset: Optional[str] = None) -> Dict[str, Any]:
    """Return a configuration based on a preset name. If None, uses static preset."""
    p = (preset or CONFIG_PRESET or "default").lower()
    if p in ("dev", "development"):
        return get_development_config()
    if p in ("prod", "production"):
        return get_production_config()
    if p in ("test", "testing"):
        return get_testing_config()
    return get_framework_config()


# Print config summary when imported (can be disabled by setting QUIET_IMPORT=True)
# Only print once per process to avoid duplicates
if not os.getenv("QUIET_IMPORT") and not hasattr(print_config_summary, '_called'):
    print_config_summary()
    print_config_summary._called = True