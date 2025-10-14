#!/usr/bin/env python3
"""
Debug what's happening with the Spotify cache file.
"""
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Debugging Spotify Cache")
print("=" * 70)

# First, check the raw cache file
cache_path = ".spotify_cache"
print(f"\n1. Checking cache file: {cache_path}")
print(f"   Exists: {os.path.exists(cache_path)}")

if os.path.exists(cache_path):
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    print("   Contents:")
    for key in cache_data:
        if key == "access_token":
            print(f"     - {key}: {cache_data[key][:20]}...")
        elif key == "refresh_token":
            print(f"     - {key}: {cache_data[key][:20]}...")
        else:
            print(f"     - {key}: {cache_data[key]}")

# Now try to use SpotifyOAuth to read it
print("\n2. Testing SpotifyOAuth cache reading:")

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

print(f"   CLIENT_ID present: {bool(client_id)}")
print(f"   CLIENT_SECRET present: {bool(client_secret)}")
print(f"   REDIRECT_URI: {redirect_uri}")

cache_handler = CacheFileHandler(cache_path=cache_path)

auth_manager = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-read-playback-state user-modify-playback-state user-read-private user-library-read",
    cache_handler=cache_handler,
    open_browser=False
)

print("\n3. Attempting to get cached token:")
token_info = auth_manager.get_cached_token()
if token_info:
    print("   ✅ Token retrieved successfully!")
    print(f"   Token expires at: {token_info.get('expires_at')}")
    print(f"   Is expired: {auth_manager.is_token_expired(token_info)}")
else:
    print("   ❌ Failed to get token from cache")
    print("   Trying validate_token method...")
    token_info = auth_manager.validate_token(cache_handler.get_cached_token())
    if token_info:
        print("   ✅ Token validated!")
    else:
        print("   ❌ Token validation failed")

print("\n4. Attempting to create Spotify client:")
try:
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    print(f"   ✅ Success! User: {user.get('display_name')} ({user.get('id')})")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 70)