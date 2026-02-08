"""
Configuration for enabling/disabling MCP tools and their settings.
"""

import platform

# =============================================================================
# SECTION 1: TOOL ENABLEMENT
# =============================================================================
# Set True to enable, False to disable.

ENABLED_TOOLS = {
    # Legacy tools (disabled - replaced by MCP equivalents)
    "state_manager": False,
    "batch_light_control": False,
    "lighting_scene": False,
    
    # Active MCP tools
    "state_tool": True,
    "get_notifications": True,
    "briefing": True,  # Create and manage briefing announcements
    "spotify_playback": True,
    "calendar_data": True,
    "weather": True,
    "google_search": True,
    "kasa_lighting": True,
    "system_info": True,
    
    # macOS-only tools (conditionally enabled)
    "send_sms": platform.system() == "Darwin",  # Only enable on macOS
    "cursor_composer": platform.system() == "Darwin",  # Only enable on macOS
    "read_clipboard": platform.system() == "Darwin",  # Only enable on macOS
    "stickies": platform.system() == "Darwin",  # Only enable on macOS
}

# Tools that should NEVER be loaded (overrides ENABLED_TOOLS)
DISABLED_TOOLS = []


# =============================================================================
# SECTION 2: TOOL-SPECIFIC SETTINGS
# =============================================================================

TOOL_CONFIG = {
    "spotify_playback": {
        "default_user": "morgan",
        "allowed_users": ["morgan"],
    },
    "get_notifications": {
        # Notifications are marked as read when retrieved (sent to the LLM as context)
        "auto_mark_read": True,
        "max_notifications": 10,
    },
    "batch_light_control": {
        "allowed_rooms": ["living_room", "bedroom", "kitchen", "bathroom"],
        "max_brightness": 100,
        "min_brightness": 10,
    },
    "send_sms": {
        # Default phone number for notifications (can be overridden per-call)
        "default_phone_number": "+16319027854",
    },
}


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

def is_tool_enabled(tool_name: str) -> bool:
    """
    Check if a tool is enabled.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        True if enabled, False otherwise
    """
    if tool_name in DISABLED_TOOLS:
        return False
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
