"""
Configuration for enabling/disabling specific MCP tools.

This file allows you to control which tools are available to the voice assistant.
Simply set a tool to True to enable it or False to disable it.
"""

# Tool enable/disable configuration
ENABLED_TOOLS = {
    # Core tools - usually kept enabled
    "calendar_data": True,          # Calendar and scheduling queries
    "state_manager": True,          # System state management
    "get_notifications": True,      # Notification checking for users
    
    # Smart home controls
    "batch_light_control": True,    # Light on/off/brightness control
    "lighting_scene": True,         # Scene controls (mood, party, etc.)
    
    # Entertainment
    "spotify_playback": True,       # Music playback control
}

# Optional: Specify tools that should never be loaded
# (overrides ENABLED_TOOLS if there's a conflict)
DISABLED_TOOLS = [
    # Example: "dangerous_tool",
]

# Tool-specific configuration
TOOL_CONFIG = {
    "spotify_playback": {
        "default_user": "Morgan",  # Default Spotify user if not specified
        "allowed_users": ["Morgan", "Spencer"],
    },
    "get_notifications": {
        "auto_mark_read": False,  # Don't automatically mark as read
        "max_notifications": 10,  # Limit number of notifications returned
    },
    "batch_light_control": {
        "allowed_rooms": ["living_room", "bedroom", "kitchen", "bathroom"],
        "max_brightness": 100,
        "min_brightness": 10,
    }
}

def is_tool_enabled(tool_name: str) -> bool:
    """
    Check if a tool is enabled.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        True if tool is enabled, False otherwise
    """
    # Check if explicitly disabled
    if tool_name in DISABLED_TOOLS:
        return False
    
    # Check enabled status (default to False if not listed)
    return ENABLED_TOOLS.get(tool_name, False)

def get_tool_config(tool_name: str) -> dict:
    """
    Get configuration for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool-specific configuration dictionary
    """
    return TOOL_CONFIG.get(tool_name, {})