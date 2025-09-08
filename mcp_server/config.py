"""
Configuration file for MCP server tools.
Contains settings for lights, users, and other system components.
"""

import os

# Debug mode
DEBUG_MODE = True


# Light configuration
LIGHT_ONE_IP = "192.168.1.49"  # Morgan's LED
LIGHT_TWO_IP = "192.168.1.165"  # Living Room Lamp

# Light room mapping    
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light one": {
            "ip": LIGHT_ONE_IP,
            "room": "morgan_room",
        },
        "Light two": {
            "ip": LIGHT_TWO_IP,
            "room": "living_room",
        }
    },
    "rooms": {
        "morgan_room": ["Light one"],
        "living_room": ["Light two"],
        "all": ["Light one", "Light two"]
    }
}

# Spotify configuration
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"

# User configurations
SPOTIFY_USERS = {
    "Morgan": {
        "username": "morgan_spotify_username"
    },
    "Spencer": {
        "username": "spencer_spotify_username"
    }
}

# Chat controlled states (for state_tool)
ALL_CHAT_CONTROLLED_STATES = [
    "spotify_user",
    "lighting_scene",
    "volume_level",
    "do_not_disturb"
]

# Calendar users
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