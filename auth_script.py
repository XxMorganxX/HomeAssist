#!/usr/bin/env python3
"""
Standalone Spotify OAuth helper that:
- Loads credentials from .env
- Opens your browser for consent
- Starts a tiny local HTTP server to capture the redirect
- Exchanges the code for tokens and caches them
- Verifies by calling /me and /devices

Usage:
  1) Ensure .env contains:
       SPOTIFY_CLIENT_ID=...
       SPOTIFY_CLIENT_SECRET=...
       SPOTIFY_REDIRECT_URI=http://localhost:8080/callback
  2) Run:  python auth_script.py
  3) Follow on-screen instructions

Notes:
- You can change the redirect port by editing SPOTIFY_REDIRECT_URI in .env
- Tokens are cached to .spotify_cache by default
"""
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import webbrowser
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler

# ----------------------------------------------------------------------------
# Load environment
# ----------------------------------------------------------------------------
ROOT = Path(__file__).parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8080/callback")
CACHE_PATH = os.getenv("SPOTIFY_CACHE_PATH", ".spotify_cache")
SHOW_DIALOG = (os.getenv("SPOTIFY_SHOW_DIALOG", "false").lower() in ("1", "true", "yes"))

SCOPES = [
    "user-read-playback-state",
    "user-modify-playback-state",
    "playlist-read-private",
]

print("=" * 80)
print("🎵 Spotify OAuth Helper")
print("=" * 80)
print(f".env loaded: {ENV_PATH.exists()}  → {ENV_PATH if ENV_PATH.exists() else '(not found)'}")
print(f"CLIENT_ID present: {bool(CLIENT_ID)}")
print(f"CLIENT_SECRET present: {bool(CLIENT_SECRET)}")
print(f"REDIRECT_URI: {REDIRECT_URI}")
print(f"CACHE_PATH: {CACHE_PATH}")
print(f"SHOW_DIALOG: {SHOW_DIALOG}")
print("Scopes:")
for s in SCOPES:
    print(f"  - {s}")

if not CLIENT_ID or not CLIENT_SECRET:
    print("\n❌ Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in .env")
    sys.exit(2)

# ----------------------------------------------------------------------------
# Tiny local server to capture the redirect
# ----------------------------------------------------------------------------
redirect_result = {"full_url": None}

class CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # silence default logging

    def do_GET(self):
        try:
            full_url = f"{REDIRECT_URI.split('/callback')[0]}{self.path}"
            redirect_result["full_url"] = full_url
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK - You can close this tab and return to the app.")
        except Exception:
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

# Parse host/port from REDIRECT_URI
parsed = urlparse(REDIRECT_URI)
host = parsed.hostname or "localhost"
port = parsed.port or 8080

server = HTTPServer((host, port), CallbackHandler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)

print("\n[1/5] Starting local redirect listener...")
try:
    server_thread.start()
    print(f"    Listening on http://{host}:{port} ...")
except OSError as e:
    print(f"    ❌ Failed to bind http://{host}:{port}: {e}")
    print("    Try changing SPOTIFY_REDIRECT_URI to a different port in .env and re-run.")
    sys.exit(3)

# ----------------------------------------------------------------------------
# Build OAuth and get authorize URL
# ----------------------------------------------------------------------------
print("[2/5] Building SpotifyOAuth...")
cache_handler = CacheFileHandler(cache_path=CACHE_PATH)
auth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=" ".join(SCOPES),
    cache_handler=cache_handler,
    open_browser=True,     # request to open default browser
    show_dialog=SHOW_DIALOG,
)

print("[3/5] Generating authorization URL...")
auth_url = auth.get_authorize_url()
print("    Authorization URL:")
print(f"    {auth_url}")

print("[4/5] Opening browser for consent...")
try:
    opened = webbrowser.open(auth_url, new=1, autoraise=True)
    print(f"    Browser open attempted: {opened}")
except Exception as e:
    print(f"    ⚠️ Browser open failed: {e}")
    print("    Please copy/paste the Authorization URL into your browser manually.")

# Wait for redirect capture
print("[5/5] Waiting for redirect with ?code=... (Ctrl+C to abort)")
start_wait = time.time()
while redirect_result["full_url"] is None:
    if time.time() - start_wait > 300:
        print("    ❌ Timeout waiting for redirect (5 minutes)")
        server.shutdown()
        sys.exit(4)
    time.sleep(0.2)

full_redirect = redirect_result["full_url"]
print(f"    ✅ Redirect captured: {full_redirect}")

# Exchange code for token
try:
    code = auth.parse_response_code(full_redirect)
    print(f"    Extracted code: {code[:24]}...")
    token_info = auth.get_access_token(code, check_cache=False)
except Exception as e:
    print(f"    ❌ Token exchange failed: {e}")
    server.shutdown()
    sys.exit(5)
finally:
    try:
        server.shutdown()
    except Exception:
        pass

# Verify API calls
print("\nVerifying access...")
try:
    sp = spotipy.Spotify(auth_manager=auth)
    me = sp.me()
    print(f"    👤 User: {me.get('display_name')} ({me.get('id')})")
    devices = sp.devices().get("devices", [])
    print(f"    🎛️ Devices: {[d.get('name') for d in devices] or 'None detected'}")
    print(f"    🔒 Tokens cached at: {CACHE_PATH}")
    print("\n✅ Done. You can now use Spotify features in the app.")
except Exception as e:
    print(f"    ❌ API verification failed: {e}")
    sys.exit(6)
