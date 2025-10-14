#!/usr/bin/env python3
"""Check available Spotify devices."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import json

load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

project_root = Path(__file__).parent
cache_path = str(project_root / ".spotify_cache")

try:
    cache_handler = CacheFileHandler(cache_path=cache_path)
    
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-read-playback-state user-modify-playback-state playlist-read-private",
        cache_handler=cache_handler,
        open_browser=False
    )
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    print("=" * 60)
    print("Available Spotify Devices")
    print("=" * 60)
    
    devices_response = sp.devices()
    devices = devices_response['devices']
    
    if not devices:
        print("\n❌ NO DEVICES FOUND")
        print("\nTo use Spotify remote control:")
        print("1. Open Spotify on any device (phone, computer, web player)")
        print("2. Start playing any song")
        print("3. The device will then appear as available")
    else:
        print(f"\nFound {len(devices)} device(s):\n")
        for i, device in enumerate(devices, 1):
            active = "✅ ACTIVE" if device['is_active'] else "   "
            print(f"{i}. {active} {device['name']}")
            print(f"   Type: {device['type']}")
            print(f"   ID: {device['id']}")
            print(f"   Volume: {device['volume_percent']}%")
            print(f"   Restricted: {device.get('is_restricted', False)}")
            print(f"   Private Session: {device.get('is_private_session', False)}")
            print()
    
    # Also check current playback
    print("\nCurrent Playback Status:")
    current = sp.current_playback()
    if current:
        print(f"  Playing: {current['is_playing']}")
        print(f"  Device: {current['device']['name']}")
        if current.get('item'):
            print(f"  Track: {current['item']['name']}")
    else:
        print("  No active playback")
    
    print("=" * 60)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

