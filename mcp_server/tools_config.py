"""
Configuration for enabling/disabling specific MCP tools.

This file allows you to control which tools are available to the voice assistant.
Simply set a tool to True to enable it or False to disable it.
"""

# Tool enable/disable configuration
ENABLED_TOOLS = {
    # Legacy tools - kept for backward compatibility
    "calendar_data": False,          # Calendar and scheduling queries (legacy)
    "state_manager": False,          # System state management (legacy)
    "get_notifications": False,      # Notification checking for users (legacy)
    "batch_light_control": False,    # Light on/off/brightness control (legacy)
    "lighting_scene": False,         # Scene controls (mood, party, etc.) (legacy)
    "spotify_playback": False,       # Music playback control (legacy)
    
    # Improved tools - enhanced with rich parameter descriptions
    "improved_state_tool": True,         # Enhanced system state management
    "improved_get_notifications": True,     # Enhanced notification tool with metadata
    "improved_spotify_playback": True,      # Enhanced Spotify control with full API
    "improved_calendar_data": True,         # Enhanced calendar with comprehensive commands
    "improved_weather": True,               # Enhanced region-based weather tool
    "improved_google_search": True,         # New improved Google search tool
    "improved_kasa_lighting": True,         # New improved Kasa lighting tool (direct + scenes)
}

# Optional: Specify tools that should never be loaded
# (overrides ENABLED_TOOLS if there's a conflict)
DISABLED_TOOLS = [
    # Example: "dangerous_tool",
]

# Tool-specific configuration
TOOL_CONFIG = {
    "spotify_playback": {
        "default_user": "morgan",  # Default Spotify user if not specified
        "allowed_users": ["morgan"],
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