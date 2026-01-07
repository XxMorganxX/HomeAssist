"""
User Configuration Module.

Provides centralized access to user-specific configuration.
All tools and components should use this module instead of hardcoding user names.

This module reads from state_management/app_state.json and provides:
- Primary user name
- Available users list
- Integration-specific user defaults (Spotify, Calendar, etc.)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Path to shared state file
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
_STATE_FILE = _PROJECT_ROOT / "state_management" / "app_state.json"


class UserConfig:
    """
    Centralized user configuration manager.
    
    Reads user configuration from app_state.json and provides
    consistent access across all tools and components.
    """
    
    _instance: Optional['UserConfig'] = None
    _state: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        """Singleton pattern to ensure consistent state across imports."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize user config by loading state."""
        if UserConfig._state is None:
            self._load_state()
    
    def _load_state(self) -> None:
        """Load state from app_state.json."""
        try:
            if _STATE_FILE.exists():
                with open(_STATE_FILE, 'r') as f:
                    UserConfig._state = json.load(f)
                logger.debug(f"Loaded user config from {_STATE_FILE}")
            else:
                logger.warning(f"State file not found: {_STATE_FILE}")
                UserConfig._state = {}
        except Exception as e:
            logger.error(f"Failed to load user config: {e}")
            UserConfig._state = {}
    
    def reload(self) -> None:
        """Force reload of state from file."""
        UserConfig._state = None
        self._load_state()
    
    @property
    def primary_user(self) -> str:
        """
        Get the primary user name.
        
        Returns:
            Primary user name from state, or "User" as fallback.
        """
        if not UserConfig._state:
            self._load_state()
        return UserConfig._state.get("user_state", {}).get("primary_user", "User")
    
    @property
    def primary_user_lower(self) -> str:
        """Get primary user name in lowercase for case-insensitive matching."""
        return self.primary_user.lower()
    
    def get_available_users(self) -> List[str]:
        """
        Get list of all available users.
        
        For now, returns primary user. Can be extended for multi-user households.
        
        Returns:
            List of user names.
        """
        users = [self.primary_user]
        
        # Check for additional household members
        if UserConfig._state:
            household = UserConfig._state.get("user_state", {}).get("household_members", [])
            users.extend(household)
        
        return users
    
    def get_integration_config(self, integration: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific integration.
        
        Args:
            integration: Integration name (e.g., "spotify", "calendar", "smart_home")
            
        Returns:
            Integration config dict or None if not configured.
        """
        if not UserConfig._state:
            self._load_state()
        
        integrations = UserConfig._state.get("user_state", {}).get("integrations", {})
        return integrations.get(integration)
    
    def is_integration_enabled(self, integration: str) -> bool:
        """
        Check if an integration is enabled.
        
        Args:
            integration: Integration name
            
        Returns:
            True if integration is enabled, False otherwise.
        """
        config = self.get_integration_config(integration)
        if config is None:
            return False
        return config.get("enabled", False)
    
    # =========================================================================
    # Spotify-specific helpers
    # =========================================================================
    
    def get_default_spotify_user(self) -> str:
        """
        Get the default Spotify user.
        
        Returns:
            Spotify username from state, or primary_user as fallback.
        """
        if not UserConfig._state:
            self._load_state()
        
        # First check chat_controlled_state for current spotify user
        current = UserConfig._state.get("chat_controlled_state", {}).get("current_spotify_user")
        if current:
            return current.lower()
        
        # Fall back to primary user
        return self.primary_user_lower
    
    def get_spotify_users(self) -> List[str]:
        """
        Get list of configured Spotify users.
        
        Returns:
            List of Spotify usernames (lowercase).
        """
        users = set()
        
        # Add current spotify user if set
        current = UserConfig._state.get("chat_controlled_state", {}).get("current_spotify_user")
        if current:
            users.add(current.lower())
        
        # Add primary user
        users.add(self.primary_user_lower)
        
        # Add any household members
        for member in self.get_available_users():
            users.add(member.lower())
        
        return list(users)
    
    # =========================================================================
    # Calendar-specific helpers
    # =========================================================================
    
    def get_default_calendar_user(self) -> str:
        """
        Get the default calendar user identifier.
        
        Returns:
            First calendar from CALENDAR_USERS config, or fallback to primary_personal.
        """
        try:
            from mcp_server.config import CALENDAR_USERS
            if CALENDAR_USERS:
                return list(CALENDAR_USERS.keys())[0]
        except ImportError:
            pass
        return f"{self.primary_user_lower}_personal"
    
    def get_calendar_users(self) -> List[str]:
        """
        Get list of configured calendar user identifiers.
        
        Reads from CALENDAR_USERS in mcp_server/config.py.
        
        Returns:
            List of calendar user IDs.
        """
        try:
            from mcp_server.config import CALENDAR_USERS
            if CALENDAR_USERS:
                return list(CALENDAR_USERS.keys())
        except ImportError:
            pass
        
        # Fallback to generated names
        base_user = self.primary_user_lower
        return [f"{base_user}_personal", f"{base_user}_school"]
    
    # =========================================================================
    # Notifications-specific helpers
    # =========================================================================
    
    def get_notification_users(self) -> List[str]:
        """
        Get list of users for notification system.
        
        Returns:
            List of user names (proper case).
        """
        return self.get_available_users()
    
    def get_default_notification_user(self) -> str:
        """
        Get the default user for notifications.
        
        Returns:
            Primary user name.
        """
        return self.primary_user


# Global singleton instance for easy import
_user_config = UserConfig()


# ============================================================================
# Convenience functions for direct import
# ============================================================================

def get_primary_user() -> str:
    """Get the primary user name."""
    return _user_config.primary_user


def get_primary_user_lower() -> str:
    """Get the primary user name in lowercase."""
    return _user_config.primary_user_lower


def get_available_users() -> List[str]:
    """Get list of all available users."""
    return _user_config.get_available_users()


def get_default_spotify_user() -> str:
    """Get the default Spotify user."""
    return _user_config.get_default_spotify_user()


def get_spotify_users() -> List[str]:
    """Get list of configured Spotify users."""
    return _user_config.get_spotify_users()


def get_default_calendar_user() -> str:
    """Get the default calendar user."""
    return _user_config.get_default_calendar_user()


def get_calendar_users() -> List[str]:
    """Get list of calendar users."""
    return _user_config.get_calendar_users()


def get_notification_users() -> List[str]:
    """Get list of notification users."""
    return _user_config.get_notification_users()


def get_default_notification_user() -> str:
    """Get the default notification user."""
    return _user_config.get_default_notification_user()


def reload_config() -> None:
    """Force reload of user configuration."""
    _user_config.reload()

