"""
Configuration file for MCP server tools.
Contains settings for lights, users, and other system components.
"""

import os

# Debug mode
DEBUG_MODE = True

# Light configuration
LIGHT_ONE_IP = "192.168.1.100"  # Replace with actual IP
LIGHT_TWO_IP = "192.168.1.101"  # Replace with actual IP

# Light room mapping
LIGHT_ROOM_MAPPING = {
    "lights": {
        "Light 1": {
            "ip": LIGHT_ONE_IP,
            "room": "living_room",
            "credentials": {
                "username": None,  # Set if needed
                "password": None   # Set if needed
            }
        },
        "Light 2": {
            "ip": LIGHT_TWO_IP,
            "room": "bedroom",
            "credentials": {
                "username": None,
                "password": None
            }
        }
    },
    "rooms": {
        "living_room": ["Light 1"],
        "bedroom": ["Light 2"],
        "all": ["Light 1", "Light 2"]
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