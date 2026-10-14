"""
Simple state manager for MCP tools.
Provides persistent state storage for tool configurations and settings.
Uses the shared app_state.json in state_management folder.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Get path to shared state file
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
_DEFAULT_STATE_FILE = _PROJECT_ROOT / "state_management" / "app_state.json"


def _prompt_initial_setup() -> dict:
    """
    Prompt user for essential setup information on first boot.
    Returns a dictionary with the initial state structure.
    """
    print("\n" + "=" * 60)
    print("🏠 HOMEASSIST FIRST-TIME SETUP")
    print("=" * 60)
    print("\nWelcome! Let's configure your assistant.\n")
    
    # =========================================================================
    # PRIMARY USER
    # =========================================================================
    while True:
        primary_user = input("👤 Enter your name (primary user): ").strip()
        if primary_user:
            break
        print("   Name cannot be empty. Please try again.")
    
    # =========================================================================
    # HOUSEHOLD MEMBERS (optional)
    # =========================================================================
    print(f"\n👥 Household Members (optional)")
    print("   Add other people who will use this assistant.")
    print("   Enter names separated by commas, or press Enter to skip.")
    household_input = input("   Household members: ").strip()
    household_members = []
    if household_input:
        household_members = [name.strip() for name in household_input.split(",") if name.strip()]
    
    # =========================================================================
    # INTEGRATIONS
    # =========================================================================
    integrations = {}
    
    # Spotify
    print(f"\n📻 Spotify Integration")
    spotify_enabled = input("   Enable Spotify? [Y/n]: ").strip().lower() != 'n'
    if spotify_enabled:
        spotify_user = input(f"   Spotify username [{primary_user}]: ").strip()
        if not spotify_user:
            spotify_user = primary_user
        integrations["spotify"] = {
            "enabled": True,
            "username": spotify_user
        }
    else:
        spotify_user = primary_user
        integrations["spotify"] = {"enabled": False}
    
    # Calendar
    print(f"\n📅 Google Calendar Integration")
    calendar_enabled = input("   Enable Google Calendar? [Y/n]: ").strip().lower() != 'n'
    if calendar_enabled:
        integrations["calendar"] = {
            "enabled": True,
            "default_calendar": "primary"
        }
    else:
        integrations["calendar"] = {"enabled": False}
    
    # Smart Home
    print(f"\n💡 Smart Home Integration")
    smart_home_enabled = input("   Enable smart home controls? [Y/n]: ").strip().lower() != 'n'
    if smart_home_enabled:
        integrations["smart_home"] = {"enabled": True}
    else:
        integrations["smart_home"] = {"enabled": False}
    
    # =========================================================================
    # DEFAULT PREFERENCES
    # =========================================================================
    print(f"\n⚙️  Default Preferences")
    
    # Lighting scene
    print("   Lighting scene options: mood, party, movie, all_on, all_off")
    lighting_scene = input("   Default lighting scene [all_on]: ").strip().lower()
    if lighting_scene not in ["mood", "party", "movie", "all_on", "all_off"]:
        lighting_scene = "all_on"
    
    # Volume
    volume_input = input("   Default volume (0-100) [50]: ").strip()
    try:
        volume_level = int(volume_input) if volume_input else 50
        volume_level = max(0, min(100, volume_level))
    except ValueError:
        volume_level = 50
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"✅ Setup complete for {primary_user}!")
    if household_members:
        print(f"   Household: {', '.join(household_members)}")
    enabled_integrations = [k for k, v in integrations.items() if v.get("enabled")]
    if enabled_integrations:
        print(f"   Integrations: {', '.join(enabled_integrations)}")
    print("=" * 60 + "\n")
    
    # Build initial state structure
    return {
        "user_state": {
            "primary_user": primary_user,
            "household_members": household_members,
            "created_at": datetime.now().isoformat(),
            "integrations": integrations
        },
        "chat_controlled_state": {
            "current_spotify_user": spotify_user,
            "lighting_scene": lighting_scene,
            "volume_level": str(volume_level),
            "do_not_disturb": "false"
        },
        "autonomous_state": {
            "notification_queue": {
                primary_user: {
                    "notifications": [],
                    "emails": []
                }
            }
        }
    }


class StateManager:
    """
    Manages system state and user preferences for MCP tools.
    Provides simple key-value storage with JSON persistence.
    Uses shared state file at state_management/app_state.json.
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file (default: state_management/app_state.json)
        """
        if state_file is None:
            self.state_file = _DEFAULT_STATE_FILE
        else:
            self.state_file = Path(state_file)
        
        self.state: Dict[str, Any] = {}
        self._last_updated: Dict[str, str] = {}
        
        # Load existing state
        self._load_state()
        
    def _load_state(self):
        """Load state from file if it exists, prompt for setup if not."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Handle both old format (direct state) and new format (wrapped)
                    if 'state' in data:
                        self.state = data.get('state', {})
                        self._last_updated = data.get('last_updated', {})
                    else:
                        # Direct format from state_management/statemanager.py
                        self.state = data
                logger.info(f"Loaded state from {self.state_file}")
            else:
                # Ensure directory exists
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if running interactively (can prompt for input)
                if sys.stdin.isatty():
                    self.state = _prompt_initial_setup()
                    print(f"📝 State file created at {self.state_file}")
                else:
                    # Non-interactive mode - use defaults
                    logger.warning("Non-interactive mode, using default state")
                    self.state = {
                        "user_state": {
                            "primary_user": "User",
                            "created_at": datetime.now().isoformat()
                        },
                        "chat_controlled_state": {
                            "current_spotify_user": "User",
                            "lighting_scene": "all_on",
                            "volume_level": "50",
                            "do_not_disturb": "false"
                        },
                        "autonomous_state": {
                            "notification_queue": {}
                        }
                    }
                
                self._save_state()
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            # Initialize with minimal defaults on error
            self.state = {
                "chat_controlled_state": {
                    "current_spotify_user": "User",
                    "lighting_scene": "all_on",
                    "volume_level": "50",
                    "do_not_disturb": "false"
                }
            }
    
    def _save_state(self):
        """Save current state to file (direct format for compatibility)."""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save state directly (compatible with state_management/statemanager.py)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a state value.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            State value or default
        """
        # Check if key is in chat_controlled_state
        if key in self.state.get("chat_controlled_state", {}):
            return self.state["chat_controlled_state"][key]
        
        # Otherwise check top-level state
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a state value.
        
        Args:
            key: State key to set
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            # Set in chat_controlled_state if it exists there
            if key in self.state.get("chat_controlled_state", {}):
                self.state["chat_controlled_state"][key] = value
            else:
                # Otherwise set at top level
                self.state[key] = value
            
            # Update timestamp
            self._last_updated[key] = datetime.now().isoformat()
            
            # Persist to disk
            self._save_state()
            
            return True
        except Exception as e:
            logger.error(f"Failed to set state {key}={value}: {e}")
            return False
    
    def get_last_updated(self, key: str) -> Optional[str]:
        """
        Get the last update timestamp for a key.
        
        Args:
            key: State key
            
        Returns:
            ISO format timestamp or None
        """
        return self._last_updated.get(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete a state value.
        
        Args:
            key: State key to delete
            
        Returns:
            True if successful
        """
        try:
            # Try to delete from chat_controlled_state
            if key in self.state.get("chat_controlled_state", {}):
                del self.state["chat_controlled_state"][key]
            elif key in self.state:
                del self.state[key]
            
            # Remove timestamp
            if key in self._last_updated:
                del self._last_updated[key]
            
            # Persist to disk
            self._save_state()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete state key {key}: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all state values.
        
        Returns:
            Dictionary of all state values
        """
        return self.state.copy()
    
    def clear(self) -> bool:
        """
        Clear all state.
        
        Returns:
            True if successful
        """
        try:
            self.state = {}
            self._last_updated = {}
            self._save_state()
            return True
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False


