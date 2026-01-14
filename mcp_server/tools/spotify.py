"""
Spotify Playback Tool using BaseTool.

This tool provides comprehensive voice-controlled Spotify playback functionality
with enhanced parameter descriptions and detailed action specifications.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
from typing import Dict, Any, Optional, Literal, List
from mcp_server.base_tool import BaseTool
from dotenv import load_dotenv
from mcp_server.config import LOG_TOOLS

try:
    from mcp_server import config
except ImportError:
    # Fallback for MCP server context
    config = None

# Import user config for dynamic user resolution
try:
    from mcp_server.user_config import get_spotify_users, get_default_spotify_user
except ImportError:
    # Fallback if user_config not available
    def get_spotify_users():
        return ["user"]
    def get_default_spotify_user():
        return "user"

load_dotenv()


class SpotifyPlaybackTool(BaseTool):
    """Enhanced tool for comprehensive Spotify playback control with detailed command specifications."""
    
    name = "spotify_playback"
    description = "Control Spotify playback with comprehensive voice commands including play/pause, track navigation, volume control, and music search. Supports configured users and provides detailed playback status information. Use this when users want to control music playback, search for songs, or adjust audio settings."
    version = "1.0.2"
    
    def __init__(self):
        """Initialize the Spotify playback tool."""
        super().__init__()
        
        # Get configured users dynamically
        self._configured_users = get_spotify_users()
        self._default_user = get_default_spotify_user()
        
        # Spotify credentials for configured users (use lowercase keys)
        # Primary user uses default env vars, others use prefixed env vars
        self.SPOTIPY_CLIENT_ID = {}
        self.SPOTIPY_CLIENT_SECRET = {}
        
        for user in self._configured_users:
            user_lower = user.lower()
            if user_lower == self._default_user.lower():
                # Primary user uses default env vars
                self.SPOTIPY_CLIENT_ID[user_lower] = os.getenv("SPOTIFY_CLIENT_ID")
                self.SPOTIPY_CLIENT_SECRET[user_lower] = os.getenv("SPOTIFY_CLIENT_SECRET")
            else:
                # Other users use prefixed env vars
                prefix = user_lower.upper()
                self.SPOTIPY_CLIENT_ID[user_lower] = os.getenv(f"{prefix}_SPOTIFY_CLIENT_ID")
                self.SPOTIPY_CLIENT_SECRET[user_lower] = os.getenv(f"{prefix}_SPOTIFY_CLIENT_SECRET")
        
        # Debug: Log what was loaded from environment
        self.logger.info("🔍 Spotify Credentials Debug:")
        for user in self._configured_users:
            user_lower = user.lower()
            client_id = self.SPOTIPY_CLIENT_ID.get(user_lower)
            client_secret = self.SPOTIPY_CLIENT_SECRET.get(user_lower)
            self.logger.info(f"  {user}:")
            self.logger.info(f"    CLIENT_ID present: {bool(client_id)} (length: {len(client_id) if client_id else 0})")
            self.logger.info(f"    CLIENT_SECRET present: {bool(client_secret)} (length: {len(client_secret) if client_secret else 0})")
        
        # Use config values if available, otherwise fallbacks
        default_redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
        
        self.SPOTIPY_REDIRECT_URI = {}
        for user in self._configured_users:
            user_lower = user.lower()
            if config:
                uri_attr = f'{user_lower.upper()}_SPOTIFY_URI'
                self.SPOTIPY_REDIRECT_URI[user_lower] = getattr(config, uri_attr, default_redirect_uri)
            else:
                self.SPOTIPY_REDIRECT_URI[user_lower] = default_redirect_uri
        
        # Log redirect URIs for debugging
        self.logger.info("🔍 Spotify Redirect URIs:")
        for user, uri in self.SPOTIPY_REDIRECT_URI.items():
            self.logger.info(f"  {user}: {uri}")
        
        # Match the scope used by auth_script.py exactly
        self.SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"
        
        # Spotify client instances (lazy initialization)
        self.sp_clients = {}
        self.device_cache = {}
        self._devices_cache_time = {}
        self.CACHE_TIMEOUT = 30  # seconds
        
        # Available actions
        self.available_actions = ["play", "pause", "next", "previous", "volume", "search_track", "search_artist", "status", "devices", "shuffle", "repeat"]
    
    @property
    def available_users(self) -> List[str]:
        """Dynamically get available users from config."""
        return [u.lower() for u in self._configured_users]
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with comprehensive action descriptions.
        
        Returns:
            Detailed JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The Spotify action to perform. 'play' resumes or starts playback, 'pause' stops playback, 'next'/'previous' skip tracks, 'volume' adjusts playback volume (0-100), 'search_track' finds and plays a specific song, 'search_artist' finds and plays music by an artist, 'status' gets current playback information, 'devices' lists available playback devices, 'shuffle' toggles shuffle mode, 'repeat' cycles repeat modes (off/track/context).",
                    "enum": self.available_actions
                },
                "user": {
                    "type": "string",
                    "description": f"Which user's Spotify account to control (case-insensitive). Each user has separate playlists, preferences, and playback state. Default is {self._default_user} if not specified.",
                    "enum": self.available_users,
                    "default": self._default_user.lower()
                },
                "query": {
                    "type": "string",
                    "description": "Search query for track or artist when using search actions. For 'search_track': use song name and optionally artist (e.g., 'Bohemian Rhapsody Queen' or just 'Bohemian Rhapsody'). For 'search_artist': use artist name (e.g., 'The Beatles', 'Taylor Swift'). Be specific for better results."
                },
                "volume_level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Volume level percentage (0-100) when action is 'volume'. 0 is mute, 25 is quiet, 50 is moderate, 75 is loud, 100 is maximum. Consider current environment and time of day when setting volume."
                },
                "device_id": {
                    "type": "string",
                    "description": "Optional specific device ID to control. Use with 'devices' action first to see available devices. If not specified, uses the currently active device or the most recently used device."
                },
                "shuffle_state": {
                    "type": "boolean",
                    "description": "Shuffle state when action is 'shuffle'. True enables shuffle (random track order), False disables shuffle (album/playlist order). If not specified, toggles current state."
                },
                "repeat_mode": {
                    "type": "string",
                    "description": "Repeat mode when action is 'repeat'. 'off' plays through once, 'track' repeats current song, 'context' repeats current playlist/album. If not specified, cycles through modes.",
                    "enum": ["off", "track", "context"]
                },
                "search_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of search results to return when using search actions. Default is 5. Use lower values (1-3) for specific searches, higher values (5-20) for discovery.",
                    "default": 5
                }
            },
            "required": ["action"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Spotify playback control.
        
        Args:
            params: Tool parameters containing action and optional parameters
            
        Returns:
            Dictionary containing execution results and playback information
        """
        try:
            action = params.get("action")
            # Normalize user to lowercase for case-insensitive matching
            user = params.get("user", self._default_user).lower() if params.get("user") else self._default_user.lower()
            
            # Validate parameters
            if not action:
                return {
                    "success": False,
                    "error": "Missing required parameter: action",
                    "available_actions": self.available_actions,
                    "available_users": self.available_users
                }
            
            if LOG_TOOLS:
                # Use structured logging so it shows in the agent terminal
                print(f"🎵 SPOTIFY TOOL CALLED: action={action}, user={user}")
                self.logger.info("Executing Tool: Spotify -- %s", params)
            
            if action not in self.available_actions:
                return {
                    "success": False,
                    "error": f"Invalid action '{action}'",
                    "available_actions": self.available_actions
                }
            
            if user not in self.available_users:
                return {
                    "success": False,
                    "error": f"Invalid user '{user}'",
                    "available_users": self.available_users
                }
            
            # Validate action-specific parameters
            validation_error = self._validate_action_parameters(action, params)
            if validation_error:
                return validation_error
            
            # Initialize Spotify client for user
            sp_client = self._get_spotify_client(user)
            if not sp_client:
                # Check for specific missing components to provide better error messages
                import os as _os
                from pathlib import Path as _PathCheck
                _project_root = _PathCheck(__file__).parent.parent.parent
                _cache_file = _project_root / "creds" / ".spotify_cache"
                cache_exists = _cache_file.exists()
                has_credentials = bool(self.SPOTIPY_CLIENT_ID.get(user) and self.SPOTIPY_CLIENT_SECRET.get(user))

                if not cache_exists:
                    error_msg = (
                        "Spotify not authenticated. Please run 'python scripts/authenticate_spotify.py' "
                        "first to authorize Spotify access."
                    )
                elif not has_credentials:
                    error_msg = (
                        "Spotify credentials missing. Please check your .env file has "
                        "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (without spaces around '=')."
                    )
                else:
                    error_msg = (
                        "Spotify authentication failed. Please re-run the authentication script "
                        "to refresh your access tokens."
                    )

                return {
                    "success": False,
                    "error": error_msg,
                    "user": user,
                    "authentication_required": not cache_exists,
                    "credentials_missing": not has_credentials
                }
            
            # Execute the action
            result = self._execute_spotify_action(sp_client, action, params, user)
            
            # Add common metadata to all results
            result.update({
                "user": user,
                "action": action,
                "timestamp": self._get_current_timestamp()
            })
            
            # Log final output for visibility
            if LOG_TOOLS:
                try:
                    self.logger.info("Spotify tool output: %s", result)
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Spotify control: {e}")
            return {
                "success": False,
                "error": f"Spotify control failed: {str(e)}",
                "action": params.get("action", "unknown"),
                "user": params.get("user", "unknown")
            }
    
    def _validate_action_parameters(self, action: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate action-specific parameters."""
        if action in ["search_track", "search_artist"]:
            if not params.get("query"):
                return {
                    "success": False,
                    "error": f"Action '{action}' requires 'query' parameter",
                    "example": "For search_track: 'Bohemian Rhapsody', for search_artist: 'Queen'"
                }
        
        elif action == "volume":
            volume = params.get("volume_level")
            if volume is None:
                return {
                    "success": False,
                    "error": "Action 'volume' requires 'volume_level' parameter (0-100)"
                }
            if not isinstance(volume, int) or not (0 <= volume <= 100):
                return {
                    "success": False,
                    "error": "volume_level must be an integer between 0 and 100"
                }
        
        return None
    
    def _get_spotify_client(self, user: str):
        """Get or create Spotify client for user."""
        if user in self.sp_clients:
            return self.sp_clients[user]

        try:
            client_id = self.SPOTIPY_CLIENT_ID.get(user)
            client_secret = self.SPOTIPY_CLIENT_SECRET.get(user)
            redirect_uri = self.SPOTIPY_REDIRECT_URI.get(user)

            # Enhanced debugging
            self.logger.info(f"🔍 Attempting to create Spotify client for {user}:")
            self.logger.info(f"  client_id: {'✅ present' if client_id else '❌ MISSING'}")
            self.logger.info(f"  client_secret: {'✅ present' if client_secret else '❌ MISSING'}")
            self.logger.info(f"  redirect_uri: {'✅ present' if redirect_uri else '❌ MISSING'}")

            if not all([client_id, client_secret, redirect_uri]):
                missing_items = []
                if not client_id:
                    missing_items.append("CLIENT_ID (env: SPOTIFY_CLIENT_ID)")
                if not client_secret:
                    missing_items.append("CLIENT_SECRET (env: SPOTIFY_CLIENT_SECRET)")
                if not redirect_uri:
                    missing_items.append("REDIRECT_URI")
                self.logger.error(f"❌ Missing Spotify credentials for {user}: {', '.join(missing_items)}")
                self.logger.error("   Please check your .env file has SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (without spaces around =)")
                return None

            # Check if cache file exists - use absolute path to project root
            import os as _os
            import json as _json
            from pathlib import Path as _Path

            # Get the absolute path to the cache file in the creds directory
            project_root = _Path(__file__).parent.parent.parent
            cache_path = str(project_root / "creds" / ".spotify_cache")

            self.logger.info(f"🔍 Looking for cache at: {cache_path}")

            if not _os.path.exists(cache_path):
                self.logger.error(f"❌ Spotify authentication cache not found at: {cache_path}")
                self.logger.error("   Please run one of these authentication scripts first:")
                self.logger.error("   - python authenticate_spotify.py (manual browser auth)")
                self.logger.error("   - python auth_script.py (automatic browser auth)")
                self.logger.error("   This will create the required .spotify_cache file")
                return None

            # Create OAuth handler with modern CacheFileHandler
            # Use open_browser=False to avoid port conflicts with MCP server
            cache_handler = CacheFileHandler(cache_path=cache_path)

            self.logger.info(f"📁 Using cache file: {cache_path}")

            # Debug: Check what's actually in the cache file
            try:
                with open(cache_path, 'r') as f:
                    cache_content = _json.load(f)
                    self.logger.info(f"   Cache has token: {'access_token' in cache_content}")
                    self.logger.info(f"   Cache has refresh: {'refresh_token' in cache_content}")
            except Exception as e:
                self.logger.error(f"   Could not read cache file: {e}")

            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=self.SCOPE,
                cache_handler=cache_handler,
                open_browser=False  # Prevents starting local server on port 8889
            )

            # Check if token is valid and not expired
            token_info = auth_manager.get_cached_token()
            if not token_info:
                self.logger.error("❌ No valid token in cache file")
                self.logger.error("   The cache file exists but doesn't contain valid tokens")
                self.logger.error("   Please re-run the authentication script")
                return None

            if auth_manager.is_token_expired(token_info):
                self.logger.info("🔄 Token expired, attempting refresh...")
                try:
                    token_info = auth_manager.refresh_access_token(token_info['refresh_token'])
                    self.logger.info("✅ Token refreshed successfully")
                except Exception as refresh_error:
                    self.logger.error(f"❌ Failed to refresh token: {refresh_error}")
                    self.logger.error("   Please re-run the authentication script")
                    return None

            # Create Spotify client
            sp = spotipy.Spotify(auth_manager=auth_manager)
            self.sp_clients[user] = sp

            # Test the connection and log the authenticated user
            try:
                info = sp.current_user()
                display_name = (info or {}).get('display_name')
                user_id = (info or {}).get('id')
                self.logger.info(f"✅ Spotify client ready for {user}: {display_name} ({user_id})")
            except Exception as test_error:
                self.logger.error(f"❌ Failed to verify Spotify connection: {test_error}")
                self.logger.error("   This likely means authentication has failed")
                self.logger.error("   Please re-run the authentication script")
                return None

            return sp

        except Exception as e:
            self.logger.error(f"❌ Failed to create Spotify client for {user}: {e}")
            self.logger.error("   This may be due to missing dependencies or configuration issues")
            import traceback
            self.logger.error(f"   Full error: {traceback.format_exc()}")
            return None
    
    def _execute_spotify_action(self, sp_client, action: str, params: Dict[str, Any], user: str) -> Dict[str, Any]:
        """Execute a specific Spotify action."""
        try:
            if action == "play":
                return self._handle_play(sp_client, params)
            elif action == "pause":
                return self._handle_pause(sp_client)
            elif action == "next":
                return self._handle_next(sp_client)
            elif action == "previous":
                return self._handle_previous(sp_client)
            elif action == "volume":
                return self._handle_volume(sp_client, params["volume_level"])
            elif action == "search_track":
                return self._handle_search_track(sp_client, params["query"], params.get("search_limit", 5))
            elif action == "search_artist":
                return self._handle_search_artist(sp_client, params["query"], params.get("search_limit", 5))
            elif action == "status":
                return self._handle_status(sp_client)
            elif action == "devices":
                return self._handle_devices(sp_client)
            elif action == "shuffle":
                return self._handle_shuffle(sp_client, params.get("shuffle_state"))
            elif action == "repeat":
                return self._handle_repeat(sp_client, params.get("repeat_mode"))
            else:
                return {
                    "success": False,
                    "error": f"Action '{action}' not implemented"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute {action}: {str(e)}"
            }
    
    def _handle_play(self, sp_client, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle play action."""
        try:
            device_id = params.get("device_id")
            sp_client.start_playback(device_id=device_id)
            
            # Get current track info
            current = sp_client.current_playback()
            track_info = self._extract_track_info(current) if current else {}
            
            return {
                "success": True,
                "message": "Playback started",
                "current_track": track_info
            }
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str and "restriction" in error_str:
                return {
                    "success": False,
                    "error": "Cannot resume playback: Spotify restrictions apply. Try using Spotify Premium or a different device.",
                    "details": str(e)
                }
            elif "404" in error_str:
                return {
                    "success": False,
                    "error": "No active Spotify device found. Please open Spotify on a device first.",
                    "details": str(e)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to start playback: {str(e)}",
                    "details": str(e)
                }
    
    def _handle_pause(self, sp_client) -> Dict[str, Any]:
        """Handle pause action."""
        try:
            sp_client.pause_playback()
            return {
                "success": True,
                "message": "Playback paused"
            }
        except Exception as e:
            error_str = str(e).lower()
            # Check for common Spotify API errors
            if "403" in error_str and "restriction" in error_str:
                return {
                    "success": False,
                    "error": "Cannot pause: Spotify playback restrictions apply to this device or content. Try using a Spotify Premium account or a different playback device.",
                    "details": str(e)
                }
            elif "404" in error_str:
                return {
                    "success": False,
                    "error": "No active Spotify device found. Please start playing music on a Spotify device first.",
                    "details": str(e)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to pause playback: {str(e)}",
                    "details": str(e)
                }
    
    def _handle_next(self, sp_client) -> Dict[str, Any]:
        """Handle next track action."""
        sp_client.next_track()
        
        # Get new current track info
        import time
        time.sleep(0.5)  # Brief delay for track change
        current = sp_client.current_playback()
        track_info = self._extract_track_info(current) if current else {}
        
        return {
            "success": True,
            "message": "Skipped to next track",
            "current_track": track_info
        }
    
    def _handle_previous(self, sp_client) -> Dict[str, Any]:
        """Handle previous track action."""
        sp_client.previous_track()
        
        # Get new current track info
        import time
        time.sleep(0.5)  # Brief delay for track change
        current = sp_client.current_playback()
        track_info = self._extract_track_info(current) if current else {}
        
        return {
            "success": True,
            "message": "Skipped to previous track",
            "current_track": track_info
        }
    
    def _handle_volume(self, sp_client, volume_level: int) -> Dict[str, Any]:
        """Handle volume control."""
        sp_client.volume(volume_level)
        return {
            "success": True,
            "message": f"Volume set to {volume_level}%",
            "volume_level": volume_level
        }
    
    def _handle_search_track(self, sp_client, query: str, limit: int) -> Dict[str, Any]:
        """Handle track search and play."""
        results = sp_client.search(q=query, type='track', limit=limit)
        tracks = results['tracks']['items']
        
        if not tracks:
            return {
                "success": False,
                "message": f"No tracks found for query: '{query}'",
                "query": query
            }
        
        # Play the first result
        track = tracks[0]
        sp_client.start_playback(uris=[track['uri']])
        
        return {
            "success": True,
            "message": f"Playing track: {track['name']} by {', '.join(a['name'] for a in track['artists'])}",
            "track_played": self._extract_track_info_from_item(track),
            "total_results": len(tracks),
            "query": query
        }
    
    def _handle_search_artist(self, sp_client, query: str, limit: int) -> Dict[str, Any]:
        """Handle artist search and play."""
        results = sp_client.search(q=query, type='artist', limit=limit)
        artists = results['artists']['items']
        
        if not artists:
            return {
                "success": False,
                "message": f"No artists found for query: '{query}'",
                "query": query
            }
        
        # Get top tracks for the first artist
        artist = artists[0]
        top_tracks = sp_client.artist_top_tracks(artist['id'])['tracks']
        
        if top_tracks:
            # Play the top track
            track = top_tracks[0]
            sp_client.start_playback(uris=[track['uri']])
            
            return {
                "success": True,
                "message": f"Playing top track by {artist['name']}: {track['name']}",
                "artist": artist['name'],
                "track_played": self._extract_track_info_from_item(track),
                "query": query
            }
        else:
            return {
                "success": False,
                "message": f"No playable tracks found for artist: {artist['name']}",
                "artist": artist['name'],
                "query": query
            }
    
    def _handle_status(self, sp_client) -> Dict[str, Any]:
        """Handle playback status request."""
        current = sp_client.current_playback()
        
        if not current:
            return {
                "success": True,
                "message": "No active playback",
                "is_playing": False
            }
        
        track_info = self._extract_track_info(current)
        
        return {
            "success": True,
            "is_playing": current['is_playing'],
            "current_track": track_info,
            "device": current['device']['name'] if current.get('device') else "Unknown",
            "volume": current['device']['volume_percent'] if current.get('device') else None,
            "shuffle": current.get('shuffle_state', False),
            "repeat": current.get('repeat_state', 'off')
        }
    
    def _handle_devices(self, sp_client) -> Dict[str, Any]:
        """Handle device listing."""
        devices = sp_client.devices()['devices']
        
        device_list = []
        for device in devices:
            device_list.append({
                "id": device['id'],
                "name": device['name'],
                "type": device['type'],
                "is_active": device['is_active'],
                "volume": device['volume_percent']
            })
        
        return {
            "success": True,
            "devices": device_list,
            "total_devices": len(device_list)
        }
    
    def _handle_shuffle(self, sp_client, shuffle_state: Optional[bool]) -> Dict[str, Any]:
        """Handle shuffle toggle."""
        if shuffle_state is None:
            # Toggle current state
            current = sp_client.current_playback()
            current_shuffle = current.get('shuffle_state', False) if current else False
            new_shuffle = not current_shuffle
        else:
            new_shuffle = shuffle_state
        
        sp_client.shuffle(new_shuffle)
        
        return {
            "success": True,
            "message": f"Shuffle {'enabled' if new_shuffle else 'disabled'}",
            "shuffle_state": new_shuffle
        }
    
    def _handle_repeat(self, sp_client, repeat_mode: Optional[str]) -> Dict[str, Any]:
        """Handle repeat mode control."""
        if repeat_mode is None:
            # Cycle through repeat modes
            current = sp_client.current_playback()
            current_repeat = current.get('repeat_state', 'off') if current else 'off'
            
            # Cycle: off -> context -> track -> off
            cycle = {'off': 'context', 'context': 'track', 'track': 'off'}
            new_repeat = cycle.get(current_repeat, 'off')
        else:
            new_repeat = repeat_mode
        
        sp_client.repeat(new_repeat)
        
        return {
            "success": True,
            "message": f"Repeat mode set to {new_repeat}",
            "repeat_mode": new_repeat
        }
    
    def _extract_track_info(self, playback_data: Dict) -> Dict[str, Any]:
        """Extract track information from playback data."""
        if not playback_data or not playback_data.get('item'):
            return {}
        
        return self._extract_track_info_from_item(playback_data['item'])
    
    def _extract_track_info_from_item(self, track_item: Dict) -> Dict[str, Any]:
        """Extract track information from track item."""
        return {
            "name": track_item['name'],
            "artists": [artist['name'] for artist in track_item['artists']],
            "album": track_item['album']['name'],
            "duration_ms": track_item['duration_ms'],
            "uri": track_item['uri']
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except:
            return "unknown"