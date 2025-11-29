"""
Simple state manager for MCP tools.
Provides persistent state storage for tool configurations and settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages system state and user preferences for MCP tools.
    Provides simple key-value storage with JSON persistence.
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file (default: ~/.mcp_state.json)
        """
        if state_file is None:
            state_file = os.path.expanduser("~/.mcp_state.json")
        
        self.state_file = Path(state_file)
        self.state: Dict[str, Any] = {}
        self._last_updated: Dict[str, str] = {}
        
        # Load existing state
        self._load_state()
        
    def _load_state(self):
        """Load state from file if it exists."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = data.get('state', {})
                    self._last_updated = data.get('last_updated', {})
                logger.info(f"Loaded state from {self.state_file}")
            else:
                # Initialize with default state structure
                self.state = {
                    "chat_controlled_state": {
                        "current_spotify_user": "Morgan",
                        "lighting_scene": "all_on",
                        "volume_level": "50",
                        "do_not_disturb": "false"
                    }
                }
                self._save_state()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            # Initialize with defaults on error
            self.state = {
                "chat_controlled_state": {
                    "current_spotify_user": "Morgan",
                    "lighting_scene": "all_on",
                    "volume_level": "50",
                    "do_not_disturb": "false"
                }
            }
    
    def _save_state(self):
        """Save current state to file."""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'state': self.state,
                'last_updated': self._last_updated
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
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


