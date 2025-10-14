#!/usr/bin/env python3
"""
One-time Spotify authentication script.
Run this to authenticate your Spotify account before using voice commands.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler

# Load environment variables
project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

print("=" * 70)
print("🎵 Spotify Authentication Setup")
print("=" * 70)

# Get credentials
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

if not client_id or not client_secret:
    print("❌ Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
    sys.exit(1)

print(f"✅ Client ID: {client_id[:10]}...")
print(f"✅ Client Secret: {client_secret[:10]}...")
print(f"✅ Redirect URI: {redirect_uri}")
print()

# Create OAuth manager with modern CacheFileHandler
scope = "user-read-playback-state user-modify-playback-state user-read-private user-library-read"
cache_path = ".spotify_cache"

# Use the new CacheFileHandler (recommended approach)
cache_handler = CacheFileHandler(cache_path=cache_path)

auth_manager = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    open_browser=False  # Don't auto-open browser to avoid port conflicts
)

print("📋 AUTHENTICATION STEPS:")
print("-" * 70)
print("1. Copy the URL below and paste it into your web browser")
print("2. Log in to Spotify and click 'Agree' to authorize the app")
print("3. You'll be redirected to a URL that starts with your redirect URI")
print("4. Copy the ENTIRE URL from your browser's address bar")
print("5. Paste it back here when prompted")
print("-" * 70)
print()

# Get authorization URL
auth_url = auth_manager.get_authorize_url()
print("🔗 AUTHORIZATION URL:")
print(auth_url)
print()

# Get the response URL from user
response_url = input("📥 Paste the redirect URL here: ").strip()

if not response_url:
    print("❌ No URL provided. Exiting.")
    sys.exit(1)

try:
    # Get the access token using the modern approach
    code = auth_manager.parse_response_code(response_url)
    print(f"✓ Extracted authorization code: {code[:20]}...")
    token_info = auth_manager.get_access_token(code, check_cache=False)
    
    print()
    print("=" * 70)
    print("✅ SUCCESS! Spotify authentication complete!")
    print("=" * 70)
    print(f"📁 Token saved to: {cache_path}")
    print()
    
    # Test the authentication
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user_info = sp.current_user()
    print(f"👤 Authenticated as: {user_info['display_name']} (@{user_info['id']})")
    print()
    print("🎵 You can now use Spotify voice commands in HomeAssist!")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print(f"❌ ERROR: {e}")
    print("=" * 70)
    print()
    print("Common issues:")
    print("  - Make sure you copied the ENTIRE URL from your browser")
    print("  - Verify the redirect URI in your .env matches your Spotify app settings")
    print("  - Check that your Spotify app credentials are correct")
    sys.exit(1)

