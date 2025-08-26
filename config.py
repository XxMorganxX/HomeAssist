"""
Unified Configuration for OpenAI MCP Agent
Supports both traditional OpenAI API and WebSocket Realtime API implementations.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Traditional OpenAI API models
RESPONSE_MODEL = "gpt-5-nano"                           # Standard chat completions model
MAX_COMPLETION_TOKENS = 2000                            # Token limit for responses

# WebSocket Realtime API models  
RESPONSE_REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # WebSocket realtime model

# =============================================================================
# IMPLEMENTATION SELECTION
# =============================================================================

# Choose which implementation to use by default
# Options: "traditional", "websocket"
DEFAULT_IMPLEMENTATION = "websocket"                   # Use WebSocket for lower latency

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

USE_REALTIME_API = True 

# Core debugging (applies to both implementations)
DEBUG_MESSAGE_BEING_SENT = True        # Show messages being sent to API
DEBUG_RAW_RESPONSE = True              # Show raw API response objects  
DEBUG_MESSAGE_CHOICES = True           # Show response choices and selection logic

# WebSocket specific debugging
DEBUG_WS_CONNECTION = True            # Log WebSocket connection events
DEBUG_WS_EVENTS = True               # Log all WebSocket event types received
DEBUG_WS_EVENT_DETAILS = True          # Show detailed event content
DEBUG_WS_FULL_EVENTS = True            # Log complete raw event objects (very verbose)
DEBUG_WS_TIMEOUTS = True               # Log timeout events and retries
DEBUG_WS_TEXT_STREAMING = True         # Log text delta streaming in real-time

# MCP and Tool debugging (applies to both implementations)
DEBUG_MCP_TOOLS = True                 # Log MCP tool discovery and schema conversion
DEBUG_TOOL_CALLS = True                # Log function calls being made
LOG_TOOLS = True                        # Surface tool call results in terminal
DEBUG_TOOL_SCHEMA = True               # Show tool schema conversion details

# Context and conversation debugging (applies to both implementations)
DEBUG_CONVERSATION_CONTEXT = True       # Log conversation history being sent
DEBUG_TOKEN_COUNTING = True             # Show exact token counts
DISPLAY_CONVERSATION_HISTORY = True     # Print full conversation history

# Performance debugging (applies to both implementations)
DEBUG_TIMING = False                    # Log detailed timing information
DEBUG_PERFORMANCE_BREAKDOWN = False     # Show time breakdown by operation

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def apply_debug_preset(preset_name: str):
    """Apply a debug preset configuration."""
    global DEBUG_MESSAGE_BEING_SENT, DEBUG_RAW_RESPONSE, DEBUG_MESSAGE_CHOICES
    global DEBUG_WS_CONNECTION, DEBUG_WS_EVENTS, DEBUG_WS_EVENT_DETAILS, DEBUG_WS_FULL_EVENTS
    global DEBUG_WS_TIMEOUTS, DEBUG_WS_TEXT_STREAMING, DEBUG_MCP_TOOLS, DEBUG_TOOL_CALLS
    global LOG_TOOLS, DEBUG_TOOL_SCHEMA, DEBUG_CONVERSATION_CONTEXT, DEBUG_TOKEN_COUNTING
    global DISPLAY_CONVERSATION_HISTORY, DEBUG_TIMING, DEBUG_PERFORMANCE_BREAKDOWN
    
    if preset_name == "production":
        # All debugging off - clean production output
        DEBUG_WS_CONNECTION = False
        DEBUG_WS_EVENTS = False
        DEBUG_WS_EVENT_DETAILS = False
        DEBUG_WS_FULL_EVENTS = False
        DEBUG_WS_TIMEOUTS = False
        DEBUG_WS_TEXT_STREAMING = False
        DEBUG_MCP_TOOLS = False
        DEBUG_TOOL_CALLS = False
        DEBUG_TOOL_SCHEMA = False
        DEBUG_CONVERSATION_CONTEXT = False
        DEBUG_TOKEN_COUNTING = False
        DEBUG_MESSAGE_BEING_SENT = False
        DEBUG_MESSAGE_CHOICES = False
        DEBUG_RAW_RESPONSE = False
        DEBUG_TIMING = False
        DEBUG_PERFORMANCE_BREAKDOWN = False
        LOG_TOOLS = False
        DISPLAY_CONVERSATION_HISTORY = False
    
    elif preset_name == "basic":
        # Basic debugging - just tool calls and major events
        globals().update({k: False for k in globals() if k.startswith('DEBUG_')})
        LOG_TOOLS = True
        DEBUG_TOOL_CALLS = False  # Just show "🔧 Calling toolname" without args
        
    elif preset_name == "development":
        # Good for development - moderate debugging
        globals().update({k: False for k in globals() if k.startswith('DEBUG_')})
        DEBUG_WS_CONNECTION = True
        DEBUG_WS_EVENTS = True
        DEBUG_TOOL_CALLS = True
        LOG_TOOLS = True
        DEBUG_CONVERSATION_CONTEXT = True
        DEBUG_MCP_TOOLS = True
        
    elif preset_name == "troubleshooting":
        # Full debugging for troubleshooting issues
        globals().update({k: True for k in globals() if k.startswith('DEBUG_')})
        LOG_TOOLS = True
        DISPLAY_CONVERSATION_HISTORY = True
        DEBUG_WS_FULL_EVENTS = False  # Still too verbose
        DEBUG_TOOL_CALLS = True  # Show tool calls with full parameters
        DEBUG_TOOL_SCHEMA = True  # Show tool schema details

    elif preset_name == "performance":
        # Focus on performance analysis
        globals().update({k: False for k in globals() if k.startswith('DEBUG_')})
        DEBUG_TIMING = True
        DEBUG_PERFORMANCE_BREAKDOWN = True
        LOG_TOOLS = False
        
    elif preset_name == "websocket_debug":
        # WebSocket-specific debugging
        globals().update({k: False for k in globals() if k.startswith('DEBUG_')})
        DEBUG_WS_CONNECTION = True
        DEBUG_WS_EVENTS = True
        DEBUG_WS_EVENT_DETAILS = True
        DEBUG_WS_TEXT_STREAMING = True
        LOG_TOOLS = True

# Apply default preset - change this line to switch debug levels:
#apply_debug_preset("production")         # Clean output for end users
# apply_debug_preset("basic")            # Just show tool calls  
apply_debug_preset("development")      # Good balance for development
#apply_debug_preset("troubleshooting")  # Full debugging
#apply_debug_preset("performance")      # Performance analysis
# apply_debug_preset("websocket_debug")  # WebSocket-specific debugging

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are Morgan Stuart's personal smart home assistant. You ALWAYS know that the user is Morgan Stuart (Mr. Stuart).

REMEMBER: 
- The user's name is Morgan Stuart
- ALWAYS address him as "Mr. Stuart" 
- When asked "What is my name?" respond: "Your name is Morgan Stuart, Mr. Stuart."
- For tool calls requiring a user parameter, use "morgan"

You have access to tools for notifications, lights, calendar, Spotify, weather, and more.

IMPORTANT: Only use tools when the user asks about:
- Smart home devices (lights, thermostats, etc.)
- Personal information (calendar, notifications, etc.)
- Home automation tasks
- Spotify or music control
- Weather forecasts and conditions
- Any home-related queries

When asked allow user to reads this system prompt.

For general knowledge questions, historical facts, or non-home topics, provide direct answers without using tools.

Be helpful, concise, and use tools only when appropriate for home-related requests.

Tool response should be concise and only include the information that is relevant to the user's request. For instance, if the user asks for notifications, you don't need to include notification id or status.
"""

# =============================================================================
# IMPLEMENTATION-SPECIFIC FUNCTIONS
# =============================================================================

def get_active_model():
    """Get the model to use based on the default implementation."""
    if DEFAULT_IMPLEMENTATION == "websocket":
        return RESPONSE_REALTIME_MODEL
    else:
        return RESPONSE_MODEL

def is_websocket_mode():
    """Check if WebSocket mode is active."""
    return DEFAULT_IMPLEMENTATION == "websocket"

def get_implementation_name():
    """Get the current implementation name for logging."""
    return "WebSocket Realtime API" if is_websocket_mode() else "Traditional OpenAI API"

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate the current configuration."""
    errors = []
    
    # Check required environment variables
    import os
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable not set")
    
    # Check model configurations
    if not RESPONSE_MODEL:
        errors.append("RESPONSE_MODEL not configured")
    if not RESPONSE_REALTIME_MODEL:
        errors.append("RESPONSE_REALTIME_MODEL not configured")
    
    # Check implementation selection
    if DEFAULT_IMPLEMENTATION not in ["traditional", "websocket"]:
        errors.append(f"Invalid DEFAULT_IMPLEMENTATION: {DEFAULT_IMPLEMENTATION}")
    
    return errors

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("🔧 Configuration Summary")
    print("=" * 60)
    print(f"Implementation: {get_implementation_name()}")
    print(f"Active Model: {get_active_model()}")
    print(f"Max Tokens: {MAX_COMPLETION_TOKENS}")
    print(f"Debug Level: Production" if not any(globals().get(k, False) for k in globals() if k.startswith('DEBUG_')) else "Development/Debug")
    print(f"Tool Logging: {'Enabled' if LOG_TOOLS else 'Disabled'}")
    
    # Show validation errors if any
    errors = validate_config()
    if errors:
        print("\n⚠️  Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Configuration Valid")
    print("=" * 60)

# =============================================================================
# PRESET QUICK-SWITCH FUNCTIONS
# =============================================================================

def set_production_mode():
    """Quick switch to production mode."""
    global DEFAULT_IMPLEMENTATION
    DEFAULT_IMPLEMENTATION = "websocket"
    apply_debug_preset("production")

def set_development_mode():
    """Quick switch to development mode."""
    global DEFAULT_IMPLEMENTATION  
    DEFAULT_IMPLEMENTATION = "websocket"
    apply_debug_preset("development")

def set_traditional_mode():
    """Quick switch to traditional API mode."""
    global DEFAULT_IMPLEMENTATION
    DEFAULT_IMPLEMENTATION = "traditional"
    apply_debug_preset("basic")

def set_websocket_mode():
    """Quick switch to WebSocket mode."""
    global DEFAULT_IMPLEMENTATION
    DEFAULT_IMPLEMENTATION = "websocket"
    apply_debug_preset("basic")

# Print config summary when imported (can be disabled by setting QUIET_IMPORT=True)
import os
if not os.getenv("QUIET_IMPORT"):
    print_config_summary()