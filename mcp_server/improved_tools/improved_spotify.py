"""
Improved Spotify Playback Tool using ImprovedBaseTool.

This tool provides comprehensive voice-controlled Spotify playback functionality
with enhanced parameter descriptions and detailed action specifications.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, Any, Optional, Literal
from mcp_server.improved_base_tool import ImprovedBaseTool
from dotenv import load_dotenv
import sys
sys.path.insert(0, '../..')
from config import LOG_TOOLS

try:
    import config
except ImportError:
    # Fallback for MCP server context
    config = None

load_dotenv()


class ImprovedSpotifyPlaybackTool(ImprovedBaseTool):
    """Enhanced tool for comprehensive Spotify playback control with detailed command specifications."""
    
    name = "improved_spotify_playback"
    description = "Control Spotify playback with comprehensive voice commands including play/pause, track navigation, volume control, and music search. Supports multiple users (Morgan/Spencer) and provides detailed playback status information. Use this when users want to control music playback, search for songs, or adjust audio settings."
    version = "1.0.1"
    
    def __init__(self):
        """Initialize the improved Spotify playback tool."""
        super().__init__()
        
        # Spotify credentials for multiple users
        self.SPOTIPY_CLIENT_ID = {
            "Morgan": os.getenv("MORGAN_SPOTIFY_CLIENT_ID"),
            "Spencer": os.getenv("SPENCER_SPOTIFY_CLIENT_ID")
        }
        self.SPOTIPY_CLIENT_SECRET = {
            "Morgan": os.getenv("MORGAN_SPOTIFY_CLIENT_SECRET"),
            "Spencer": os.getenv("SPENCER_SPOTIFY_CLIENT_SECRET")
        }
        
        # Use config values if available, otherwise fallbacks
        if config:
            self.SPOTIPY_REDIRECT_URI = {
                "Morgan": getattr(config, 'MORGAN_SPOTIFY_URI', "http://localhost:8888/callback"),
                "Spencer": getattr(config, 'SPENCER_SPOTIFY_URI', "http://localhost:8888/callback")
            }
        else:
            self.SPOTIPY_REDIRECT_URI = {
                "Morgan": "http://localhost:8888/callback",
                "Spencer": "http://localhost:8888/callback"
            }
        
        self.SCOPE = "user-read-playback-state user-modify-playback-state user-read-private user-library-read"
        
        # Spotify client instances (lazy initialization)
        self.sp_clients = {}
        self.device_cache = {}
        self._devices_cache_time = {}
        self.CACHE_TIMEOUT = 30  # seconds
        
        # Available users and actions
        self.available_users = ["Morgan", "Spencer"]
        self.available_actions = ["play", "pause", "next", "previous", "volume", "search_track", "search_artist", "status", "devices", "shuffle", "repeat"]
    
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
                    "description": "Which user's Spotify account to control. Each user has separate playlists, preferences, and playback state. Morgan typically has work/focus playlists, Spencer has personal/entertainment music. Default is Morgan if not specified.",
                    "enum": self.available_users,
                    "default": "Morgan"
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
            user = params.get("user", "Morgan")
            
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
                return {
                    "success": False,
                    "error": f"Failed to initialize Spotify client for {user}. Check credentials and authentication.",
                    "user": user
                }
            
            # Execute the action
            result = self._execute_spotify_action(sp_client, action, params, user)
            
            # Add common metadata to all results
            result.update({
                "user": user,
                "action": action,
                "timestamp": self._get_current_timestamp()
            })
            
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
            
            if not all([client_id, client_secret, redirect_uri]):
                self.logger.error(f"Missing Spotify credentials for {user}")
                return None
            
            # Create OAuth handler
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=self.SCOPE,
                cache_path=f".spotify_cache_{user.lower()}"
            )
            
            # Create Spotify client
            sp = spotipy.Spotify(auth_manager=auth_manager)
            self.sp_clients[user] = sp
            
            # Test the connection
            sp.current_user()
            return sp
            
        except Exception as e:
            self.logger.error(f"Failed to create Spotify client for {user}: {e}")
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
    
    def _handle_pause(self, sp_client) -> Dict[str, Any]:
        """Handle pause action."""
        sp_client.pause_playback()
        return {
            "success": True,
            "message": "Playback paused"
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