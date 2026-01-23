"""
Configuration for MCP server tools.
Organized into discrete feature sections.
"""

import os
from pathlib import Path

# Project root (parent of mcp_server/)
_PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# SECTION 1: DEBUG & LOGGING
# =============================================================================

DEBUG_MODE = False  # Set to True for verbose calendar/tool output
LOG_TOOLS = os.environ.get("LOG_TOOLS", "true").lower() in ("true", "1", "yes")


# =============================================================================
# SECTION 2: TIME & LOCALE
# =============================================================================

# Default IANA time zone for calendar events
DEFAULT_TIME_ZONE = os.environ.get("DEFAULT_TIME_ZONE", "America/New_York")


# =============================================================================
# SECTION 3: SMART LIGHTING (Kasa)
# =============================================================================

# Individual light IP addresses
LIGHT_IPS = {
    "morgans_led": "192.168.1.49",
    "living_room_lamp": "192.168.1.165",
}

# Light-to-room mapping and room definitions
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light one": {
            "ip": LIGHT_IPS["morgans_led"],
            "room": "morgan_room",
        },
        "Light two": {
            "ip": LIGHT_IPS["living_room_lamp"],
            "room": "living_room",
        }
    },
    "rooms": {
        "morgan_room": ["Light one"],
        "living_room": ["Light two"],
        "all": ["Light one", "Light two"]
    }
}


# =============================================================================
# SECTION 4: SPOTIFY
# =============================================================================

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"

# User-specific Spotify configurations
SPOTIFY_USERS = {
    "Morgan": {
        "username": "morgan_spotify_username"
    },
    "Spencer": {
        "username": "spencer_spotify_username"
    }
}


# =============================================================================
# SECTION 5: GOOGLE CALENDAR
# =============================================================================

# Service Account credentials (recommended - never expires, no user interaction)
# The same service account can access multiple calendars if each is shared with it
_DEFAULT_SA_FILE = str(_PROJECT_ROOT / "creds" / "homeassist-google-service.json")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", _DEFAULT_SA_FILE)

# Calendar scopes
CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# User calendar configurations
# All calendars use the same service account - just share each calendar with:
#   calendar-homeassist@homeassist-465018.iam.gserviceaccount.com
#
# The "calendar_id" is the email address of the calendar owner (for main calendars)
# or the calendar ID from Settings > Integrate calendar (for secondary calendars)
#
# Optional fields:
#   - display_name: Human-friendly name shown to users
#   - aliases: List of alternative names the user might say (lowercase)
CALENDAR_USERS = {
    # Calendar 1: Morgan's personal Gmail calendar
    "morgan_personal": {
        "calendar_id": "morgannstuart@gmail.com",
        "display_name": "Personal",
        "aliases": ["personal", "main", "gmail", "my calendar"],
    },
    
    # Calendar 2: Morgan's Cornell calendar
    "morgan_school": {
        "calendar_id": "mns66@cornell.edu",
        "display_name": "School",
        "aliases": ["school", "cornell", "university", "class", "classes"],
    },
    
    # Calendar 3: Gen AI class calendar
    "Gen_AI": {
        "calendar_id": "c_a9971ca39f405c3c8c855332f1d5bc8721378fef7845b1e3fc7b62601e0911bc@group.calendar.google.com",
        "display_name": "Gen AI Class",
        "aliases": ["genai", "gen ai", "gen_ai", "ai", "ai class", "generative ai"],
    },
    
    # Calendar 4: HomeAssist calendar (secondary calendar under morgannstuart@gmail.com)
    "homeassist": {
        "calendar_id": "bd7409eb309d624908ee53c2adf02cfc3d087e50dd1139909df8d13e2b8bb8e4@group.calendar.google.com",
        "display_name": "HomeAssist",
        "aliases": ["assistant", "home assistant", "homeassist", "reminders"],
    },
}


def get_calendar_alias_map() -> dict:
    """Build a mapping of aliases -> calendar keys for natural language matching."""
    alias_map = {}
    for key, config in CALENDAR_USERS.items():
        # Map the key itself
        alias_map[key.lower()] = key
        # Map display name
        if config.get("display_name"):
            alias_map[config["display_name"].lower()] = key
        # Map all aliases
        for alias in config.get("aliases", []):
            alias_map[alias.lower()] = key
    return alias_map


CALENDAR_ALIAS_MAP = get_calendar_alias_map()


# =============================================================================
# SECTION 6: SMS/iMESSAGE (macOS only)
# =============================================================================

# Default phone number for SMS notifications (with country code)
SMS_DEFAULT_PHONE_NUMBER = os.environ.get("SMS_PHONE_NUMBER", "+16319027854")

# SMS configuration dict for the tool
SMS_CONFIG = {
    "default_phone_number": SMS_DEFAULT_PHONE_NUMBER,
}


# =============================================================================
# SECTION 7: STATE MANAGEMENT
# =============================================================================

# States that can be controlled via chat commands
ALL_CHAT_CONTROLLED_STATES = [
    "spotify_user",
    "lighting_scene",
    "volume_level",
    "do_not_disturb"
]
