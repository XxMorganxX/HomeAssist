"""
Configuration for MCP server tools.
Organized into discrete feature sections.
"""

import os
from pathlib import Path

# Project root (parent of mcp_server/)
_PROJECT_ROOT = Path(__file__).parent.parent

# Load personal settings (calendar IDs, phone number, device IPs, API keys) from
# the project's .env so they never need to be hardcoded here. Existing
# environment variables take precedence.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional; plain env vars still work
    pass


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

# Individual light IP addresses (LAN addresses of your Kasa devices; set in .env)
LIGHT_IPS = {
    "bedroom_led": os.environ.get("KASA_LIGHT_ONE_IP", ""),
    "living_room_lamp": os.environ.get("KASA_LIGHT_TWO_IP", ""),
}

# Light-to-room mapping and room definitions
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light one": {
            "ip": LIGHT_IPS["bedroom_led"],
            "room": "bedroom",
        },
        "Light two": {
            "ip": LIGHT_IPS["living_room_lamp"],
            "room": "living_room",
        }
    },
    "rooms": {
        "bedroom": ["Light one"],
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

# User-specific Spotify configurations (keyed by HOMEASSIST_USER_NAME, default "User")
SPOTIFY_USERS = {
    os.environ.get("HOMEASSIST_USER_NAME", "User"): {
        "username": "your_spotify_username"
    },
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
#   <service-account-name>@<gcp-project-id>.iam.gserviceaccount.com
#   (the "client_email" field of your service account JSON)
#
# The "calendar_id" is the email address of the calendar owner (for main calendars)
# or the calendar ID from Settings > Integrate calendar (for secondary calendars).
# Calendar IDs are read from the environment (see .env.example) so no personal
# addresses live in source control.
#
# Optional fields:
#   - display_name: Human-friendly name shown to users
#   - aliases: List of alternative names the user might say (lowercase)
CALENDAR_USERS = {
    # Calendar 1: personal calendar (default for writes)
    "personal": {
        "calendar_id": os.environ.get("CALENDAR_ID_PERSONAL", ""),
        "display_name": "Personal",
        "aliases": ["personal", "main", "primary", "my calendar"],
    },
    
    # Calendar 2: school calendar
    "school": {
        "calendar_id": os.environ.get("CALENDAR_ID_SCHOOL", ""),
        "display_name": "School",
        "aliases": ["school", "university", "college", "class", "classes"],
    },
    
    # Calendar 3: club / extracurricular calendar
    "club": {
        "calendar_id": os.environ.get("CALENDAR_ID_CLUB", ""),
        "display_name": "Club",
        "aliases": ["club", "clubs", "organization", "extracurricular"],
    },
    
    # Calendar 4: HomeAssist calendar (secondary calendar under the personal account)
    "homeassist": {
        "calendar_id": os.environ.get("CALENDAR_ID_HOMEASSIST", ""),
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
SMS_DEFAULT_PHONE_NUMBER = os.environ.get("SMS_PHONE_NUMBER", "")

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
