"""
Configuration for MCP server tools.
Organized into discrete feature sections.
"""

import os


# =============================================================================
# SECTION 1: DEBUG & LOGGING
# =============================================================================

DEBUG_MODE = True
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

CALENDAR_USERS = {
    "morgan_personal": {
        "client_secret": "google_creds/google_creds_morgan.json",
        "token": "google_creds/token_morgan.json"
    },
    "morgan_school": {
        "client_secret": "google_creds/google_creds_morgan.json",
        "token": "google_creds/token_morgan.json"
    },
    "spencer": {
        "client_secret": "google_creds/google_creds_spencer.json",
        "token": "google_creds/token_spencer.json"
    }
}


# =============================================================================
# SECTION 6: STATE MANAGEMENT
# =============================================================================

# States that can be controlled via chat commands
ALL_CHAT_CONTROLLED_STATES = [
    "spotify_user",
    "lighting_scene",
    "volume_level",
    "do_not_disturb"
]
