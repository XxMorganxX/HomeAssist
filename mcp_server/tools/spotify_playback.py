"""
Spotify Playback MCP Tool
Provides voice-controlled Spotify playback functionality.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, Any, Optional
from mcp_server.base_tool import BaseTool, CoreServices
from typing import Literal
from dotenv import load_dotenv
import config

load_dotenv()
class SpotifyPlaybackTool(BaseTool):
    """Voice-controlled Spotify playback tool"""
    
    name = "spotify_playback"
    description = "Called when the user wants to control Spotify playback including play, pause, next, previous, volume control, and music search"
    version = "1.0.0"
    
    def __init__(self, core_services: Optional[CoreServices]):
        
        super().__init__(core_services)
        
        # Spotify credentials
        self.SPOTIPY_CLIENT_ID = {"Morgan": os.getenv("MORGAN_SPOTIFY_CLIENT_ID"), "Spencer": os.getenv("SPENCER_SPOTIFY_CLIENT_ID")}
        self.SPOTIPY_CLIENT_SECRET = {"Morgan": os.getenv("MORGAN_SPOTIFY_CLIENT_SECRET"), "Spencer": os.getenv("SPENCER_SPOTIFY_CLIENT_SECRET")} 
        self.SPOTIPY_REDIRECT_URI = {"Morgan": config.MORGAN_SPOTIFY_URI, "Spencer": config.SPENCER_SPOTIFY_URI}
        self.SCOPE = "user-read-playback-state user-modify-playback-state user-read-private"
        
        
        # Initialize Spotify client (lazy initialization)
        self.sp = None
        self.device_id = None
        self.current_device_name = None
        self._spotify_initialized = False
    
    def _initialize_spotify(self):
        """Initialize Spotify client and find device"""
        if self._spotify_initialized:
            return True
            
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.SPOTIPY_CLIENT_ID[config.STATE_CURRENT_SPOTIFY_USER],
                client_secret=self.SPOTIPY_CLIENT_SECRET[config.STATE_CURRENT_SPOTIFY_USER],
                redirect_uri=self.SPOTIPY_REDIRECT_URI[config.STATE_CURRENT_SPOTIFY_USER],
                scope=self.SCOPE,
                cache_path=None  # Disable token caching entirely
            ))
            
            print("✅ Spotify authentication successful!")
            
            # Find HomePi device
            devices = self._get_devices()
            if devices:
                self.device_id = self._find_homepi_device(devices)
                if self.device_id:
                    self.current_device_name = "HomePi"
                    self.log_info(f"Connected to Spotify device: {self.current_device_name}")
                else:
                    print("HomePi device not found, will use default device")
            else:
                print("No Spotify devices found")
            
            self._spotify_initialized = True
            return True
                
        except Exception as e:
            print(f"Failed to initialize Spotify: {e}")
            return False
    
    def _get_devices(self):
        """Get available Spotify devices"""
        try:
            devices = self.sp.devices()["devices"]
            return devices if devices else None
        except Exception as e:
            self.log_error(f"Error getting devices: {e}")
            return None
    
    def _find_homepi_device(self, devices):
        """Find the HomePi device"""
        for device in devices:
            if device['name'].lower() == 'homepi':
                return device['id']
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform",
                    "enum": ["play", "pause", "next", "previous", "volume", "search_track", "search_artist", "status", "devices"]
                },
                "query": {
                    "type": "string",
                    "description": "Search query for track or artist (required for search actions)"
                },
                "volume_level": {
                    "type": "integer",
                    "description": "Volume level from 0 to 100 (required for volume action)",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["action"],
            "description": self.description
        }
    
    def execute(self, params: Dict[str, Literal["play", "pause", "next", "previous", "volume", "search_track", "search_artist", "status", "devices"]]) -> Dict[str, Any]:
        """Execute the Spotify playback tool."""
        # Initialize Spotify on first use
        if not self._spotify_initialized:
            if not self._initialize_spotify():
                return {
                    "success": False,
                    "error": "Failed to initialize Spotify connection"
                }
        
        action = params.get("action")
        
        if not action:
            raise ValueError("action parameter is required")
        
        try:
            if action == "play":
                return self._play()
            elif action == "pause":
                return self._pause()
            elif action == "next":
                return self._next_track()
            elif action == "previous":
                return self._previous_track()
            elif action == "volume":
                volume_level = params.get("volume_level")
                if volume_level is None:
                    raise ValueError("volume_level parameter required for volume action")
                return self._set_volume(volume_level)
            elif action == "search_track":
                query = params.get("query")
                if not query:
                    raise ValueError("query parameter required for search_track action")
                return self._search_and_play(query, "track")
            elif action == "search_artist":
                query = params.get("query")
                if not query:
                    raise ValueError("query parameter required for search_artist action")
                return self._search_and_play(query, "artist")
            elif action == "status":
                return self._get_status()
            elif action == "devices":
                return self._get_devices_info()
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except spotipy.exceptions.SpotifyException as e:
            if "404" in str(e):
                return {
                    "success": False,
                    "error": "No active playback session found. Start playing music on Spotify first.",
                    "action": action
                }
            else:
                return {
                    "success": False,
                    "error": f"Spotify API error: {e}",
                    "action": action
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {e}",
                "action": action
            }
    
    def _play(self) -> Dict[str, Any]:
        """Start playback"""
        self.sp.start_playback(device_id=self.device_id)
        self.log_info("Playback started")
        return {
            "success": True,
            "message": "Playback started",
            "action": "play"
        }
    
    def _pause(self) -> Dict[str, Any]:
        """Pause playback"""
        self.sp.pause_playback(device_id=self.device_id)
        self.log_info("Playback paused")
        return {
            "success": True,
            "message": "Playback paused",
            "action": "pause"
        }
    
    def _next_track(self) -> Dict[str, Any]:
        """Skip to next track"""
        self.sp.next_track(device_id=self.device_id)
        self.log_info("Skipped to next track")
        return {
            "success": True,
            "message": "Skipped to next track",
            "action": "next"
        }
    
    def _previous_track(self) -> Dict[str, Any]:
        """Skip to previous track"""
        self.sp.previous_track(device_id=self.device_id)
        self.log_info("Skipped to previous track")
        return {
            "success": True,
            "message": "Skipped to previous track",
            "action": "previous"
        }
    
    def _set_volume(self, volume_level: int) -> Dict[str, Any]:
        """Set volume level"""
        self.sp.volume(volume_level, device_id=self.device_id)
        self.log_info(f"Volume set to {volume_level}")
        return {
            "success": True,
            "message": f"Volume set to {volume_level}",
            "action": "volume",
            "volume_level": volume_level
        }
    
    def _search_and_play(self, query: str, search_type: str) -> Dict[str, Any]:
        """Search for and play music"""
        try:
            if search_type == "track":
                results = self.sp.search(q=query, type='track', limit=1)
                tracks = results['tracks']['items']
                
                if not tracks:
                    return {
                        "success": False,
                        "error": f"No tracks found for '{query}'",
                        "action": "search_track",
                        "query": query
                    }
                
                selected_track = tracks[0]
                track_uri = selected_track['uri']
                artists = ", ".join([artist['name'] for artist in selected_track['artists']])
                
                # Play the selected track
                self.sp.start_playback(device_id=self.device_id, uris=[track_uri])
                self.log_info(f"Now playing: {selected_track['name']} by {artists}")
                
                return {
                    "success": True,
                    "message": f"Now playing: {selected_track['name']} by {artists}",
                    "action": "search_track",
                    "track_name": selected_track['name'],
                    "artists": artists,
                    "query": query
                }
                    
            elif search_type == "artist":
                results = self.sp.search(q=query, type='artist', limit=1)
                artists = results['artists']['items']
                
                if not artists:
                    return {
                        "success": False,
                        "error": f"No artists found for '{query}'",
                        "action": "search_artist",
                        "query": query
                    }
                
                selected_artist = artists[0]
                
                # Get top tracks for the artist
                top_tracks = self.sp.artist_top_tracks(selected_artist['id'])
                if top_tracks['tracks']:
                    track_uris = [track['uri'] for track in top_tracks['tracks'][:5]]
                    self.sp.start_playback(device_id=self.device_id, uris=track_uris)
                    self.log_info(f"Now playing top tracks by {selected_artist['name']}")
                    
                    return {
                        "success": True,
                        "message": f"Now playing top tracks by {selected_artist['name']}",
                        "action": "search_artist",
                        "artist_name": selected_artist['name'],
                        "query": query
                    }
                else:
                    return {
                        "success": False,
                        "error": "No top tracks found for this artist.",
                        "action": "search_artist",
                        "query": query
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Error searching: {e}",
                "action": f"search_{search_type}",
                "query": query
            }
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current playback status"""
        try:
            current = self.sp.current_playback()
            if current:
                track = current['item']
                artists = ", ".join([artist['name'] for artist in track['artists']])
                return {
                    "success": True,
                    "message": f"Currently playing: {track['name']} by {artists}",
                    "action": "status",
                    "track_name": track['name'],
                    "artists": artists,
                    "is_playing": current['is_playing']
                }
            else:
                return {
                    "success": True,
                    "message": "No track currently playing",
                    "action": "status",
                    "is_playing": False
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting status: {e}",
                "action": "status"
            }
    
    def _get_devices_info(self) -> Dict[str, Any]:
        """Get available devices information"""
        devices = self._get_devices()
        if devices:
            device_list = [{"name": d['name'], "type": d['type']} for d in devices]
            return {
                "success": True,
                "message": f"Found {len(devices)} Spotify devices",
                "action": "devices",
                "devices": device_list,
                "current_device": self.current_device_name
            }
        else:
            return {
                "success": False,
                "error": "No Spotify devices found",
                "action": "devices"
            }
