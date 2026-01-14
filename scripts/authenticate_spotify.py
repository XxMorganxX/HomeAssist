#!/usr/bin/env python3
"""
Spotify Authentication Script

Run this ONCE to authenticate with Spotify and create the .spotify_cache file.
After running, the cache file contains refresh tokens that work indefinitely.

Usage:
    python authenticate_spotify.py

Requirements:
    - SPOTIFY_CLIENT_ID in .env
    - SPOTIFY_CLIENT_SECRET in .env
    - Browser access (one-time only)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv



# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    from spotipy.cache_handler import CacheFileHandler
except ImportError:
    print("❌ spotipy not installed. Run: pip install spotipy")
    sys.exit(1)

# Configuration
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"

# Cache file location (creds directory)
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH = PROJECT_ROOT / "creds" / ".spotify_cache"


def main():
    print("🎵 Spotify Authentication")
    print("=" * 40)
    print()
    
    # Check credentials
    if not CLIENT_ID:
        print("❌ SPOTIFY_CLIENT_ID not found in .env")
        print("   Get it from: https://developer.spotify.com/dashboard")
        sys.exit(1)
    
    if not CLIENT_SECRET:
        print("❌ SPOTIFY_CLIENT_SECRET not found in .env")
        print("   Get it from: https://developer.spotify.com/dashboard")
        sys.exit(1)
    
    print(f"✓ Client ID: {CLIENT_ID[:8]}...")
    print(f"✓ Redirect URI: {REDIRECT_URI}")
    print(f"✓ Cache path: {CACHE_PATH}")
    print()
    
    # Check if already authenticated
    if CACHE_PATH.exists():
        print("⚠️  Cache file already exists!")
        response = input("   Overwrite and re-authenticate? (y/N): ").strip().lower()
        if response != 'y':
            print("   Keeping existing authentication.")
            
            # Test existing auth
            print("\n🔍 Testing existing authentication...")
            try:
                cache_handler = CacheFileHandler(cache_path=str(CACHE_PATH))
                auth_manager = SpotifyOAuth(
                    client_id=CLIENT_ID,
                    client_secret=CLIENT_SECRET,
                    redirect_uri=REDIRECT_URI,
                    scope=SCOPE,
                    cache_handler=cache_handler,
                    open_browser=False
                )
                sp = spotipy.Spotify(auth_manager=auth_manager)
                user = sp.current_user()
                print(f"✅ Authenticated as: {user['display_name']} ({user['id']})")
            except Exception as e:
                print(f"❌ Existing auth failed: {e}")
                print("   Run again and choose 'y' to re-authenticate")
            return
    
    print("🌐 Opening browser for Spotify authorization...")
    print("   (If browser doesn't open, copy the URL from the terminal)")
    print()
    
    try:
        # Create auth manager - this will open browser
        cache_handler = CacheFileHandler(cache_path=str(CACHE_PATH))
        auth_manager = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            cache_handler=cache_handler,
            open_browser=True  # Opens browser for auth
        )
        
        # This triggers the OAuth flow
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Test the connection
        user = sp.current_user()
        
        print()
        print("=" * 40)
        print("✅ Authentication successful!")
        print(f"   User: {user['display_name']} ({user['id']})")
        print(f"   Cache saved to: {CACHE_PATH}")
        print()
        print("📋 Next steps:")
        print("   1. The .spotify_cache file is now created")
        print("   2. HomeAssist can now control your Spotify")
        print("   3. Tokens auto-refresh, no need to re-run this")
        print()
        print("🔒 For headless servers:")
        print(f"   Copy {CACHE_PATH} to your server's project root")
        
    except spotipy.SpotifyException as e:
        print(f"❌ Spotify API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
