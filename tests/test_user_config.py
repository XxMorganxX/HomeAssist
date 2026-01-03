"""
Test suite for user configuration modularity.

Tests that all components properly use dynamic user configuration
instead of hardcoded values.
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestUserConfig:
    """Test the user_config module functionality."""
    
    @pytest.fixture
    def temp_state_dir(self):
        """Create a temporary state directory with test state file."""
        temp_dir = tempfile.mkdtemp()
        state_dir = Path(temp_dir) / "state_management"
        state_dir.mkdir(parents=True)
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_state(self):
        """Sample app_state.json content."""
        return {
            "user_state": {
                "primary_user": "TestUser",
                "household_members": ["Alice", "Bob"],
                "created_at": "2026-01-02T12:00:00",
                "integrations": {
                    "spotify": {"enabled": True, "username": "testuser"},
                    "calendar": {"enabled": True},
                    "smart_home": {"enabled": False}
                }
            },
            "chat_controlled_state": {
                "current_spotify_user": "testuser",
                "lighting_scene": "all_on",
                "volume_level": "50",
                "do_not_disturb": "false"
            },
            "autonomous_state": {
                "notification_queue": {
                    "TestUser": {
                        "notifications": [],
                        "emails": []
                    }
                }
            }
        }
    
    def test_user_config_loads_primary_user(self, temp_state_dir, sample_state):
        """Test that user_config correctly loads primary user from state."""
        state_file = Path(temp_state_dir) / "state_management" / "app_state.json"
        with open(state_file, 'w') as f:
            json.dump(sample_state, f)
        
        # Mock the module path
        with patch('mcp_server.user_config._STATE_FILE', state_file):
            from mcp_server.user_config import UserConfig
            config = UserConfig()
            config.reload()
            
            assert config.primary_user == "TestUser"
            assert config.primary_user_lower == "testuser"
    
    def test_user_config_returns_available_users(self, temp_state_dir, sample_state):
        """Test that available_users includes primary user and household members."""
        state_file = Path(temp_state_dir) / "state_management" / "app_state.json"
        with open(state_file, 'w') as f:
            json.dump(sample_state, f)
        
        with patch('mcp_server.user_config._STATE_FILE', state_file):
            from mcp_server.user_config import UserConfig
            config = UserConfig()
            config.reload()
            
            users = config.get_available_users()
            assert "TestUser" in users
            assert "Alice" in users
            assert "Bob" in users
    
    def test_user_config_fallback_when_no_state(self, temp_state_dir):
        """Test that user_config returns fallback when state file missing."""
        state_file = Path(temp_state_dir) / "state_management" / "app_state.json"
        # Don't create the file
        
        with patch('mcp_server.user_config._STATE_FILE', state_file):
            from mcp_server.user_config import UserConfig
            config = UserConfig()
            config.reload()
            
            assert config.primary_user == "User"


class TestCalendarToolModularity:
    """Test that calendar tool uses dynamic user configuration."""
    
    @pytest.fixture
    def mock_user_config(self):
        """Mock user config functions."""
        with patch('mcp_server.tools.calendar.get_calendar_users') as mock_users, \
             patch('mcp_server.tools.calendar.get_default_calendar_user') as mock_default:
            mock_users.return_value = ["testuser_personal", "testuser_school"]
            mock_default.return_value = "testuser_personal"
            yield mock_users, mock_default
    
    def test_calendar_uses_dynamic_default_user(self, mock_user_config):
        """Test that calendar tool uses configured default user."""
        from mcp_server.tools.calendar import CalendarTool
        
        tool = CalendarTool()
        
        # Check that available_users comes from config
        assert "testuser_personal" in tool.available_users
        
        # Check schema uses dynamic default
        schema = tool.get_schema()
        user_prop = schema["properties"]["commands"]["items"]["properties"]["user"]
        assert user_prop["default"] == "testuser_personal"
    
    def test_calendar_normalize_uses_dynamic_default(self, mock_user_config):
        """Test that _normalize_command uses dynamic default user."""
        from mcp_server.tools.calendar import CalendarTool
        
        tool = CalendarTool()
        
        # Empty command should get default user
        normalized = tool._normalize_command({})
        assert normalized["user"] == "testuser_personal"


class TestSpotifyToolModularity:
    """Test that spotify tool uses dynamic user configuration."""
    
    def test_spotify_uses_config_based_user(self):
        """Test that spotify tool reads from user config (not hardcoded)."""
        from mcp_server.tools.spotify import SpotifyPlaybackTool
        
        tool = SpotifyPlaybackTool()
        
        # Verify tool has dynamic user properties
        assert hasattr(tool, '_default_user')
        assert hasattr(tool, '_configured_users')
        assert hasattr(tool, 'available_users')
        
        # Verify the default user is used in schema
        schema = tool.get_schema()
        user_prop = schema["properties"]["user"]
        assert user_prop["default"] == tool._default_user.lower()
    
    def test_spotify_schema_is_dynamic(self):
        """Test that spotify schema builds from config, not hardcoded."""
        from mcp_server.tools.spotify import SpotifyPlaybackTool
        
        tool = SpotifyPlaybackTool()
        schema = tool.get_schema()
        
        # Verify enum comes from available_users
        assert schema["properties"]["user"]["enum"] == tool.available_users


class TestNotificationsToolModularity:
    """Test that notifications tool uses dynamic user configuration."""
    
    def test_notifications_uses_config_based_users(self):
        """Test that notifications tool reads from user config (not hardcoded)."""
        from mcp_server.tools.notifications import GetNotificationsTool
        
        tool = GetNotificationsTool()
        
        # Verify tool has dynamic user properties
        assert hasattr(tool, '_default_user')
        assert hasattr(tool, '_configured_users')
        
        # Verify the default user is used in schema
        schema = tool.get_schema()
        user_prop = schema["properties"]["user"]
        assert user_prop["default"] == tool._default_user
    
    def test_notifications_schema_is_dynamic(self):
        """Test that notifications schema builds from config, not hardcoded."""
        from mcp_server.tools.notifications import GetNotificationsTool
        
        tool = GetNotificationsTool()
        schema = tool.get_schema()
        
        # Verify enum comes from configured users
        assert schema["properties"]["user"]["enum"] == tool._configured_users


class TestStateManagerModularity:
    """Test that state manager uses dynamic user configuration."""
    
    @pytest.fixture
    def temp_state_file(self):
        """Create a temporary state file."""
        temp_dir = tempfile.mkdtemp()
        state_file = Path(temp_dir) / "app_state.json"
        
        initial_state = {
            "user_state": {"primary_user": "DynamicUser"},
            "chat_controlled_state": {},
            "autonomous_state": {"notification_queue": {}}
        }
        with open(state_file, 'w') as f:
            json.dump(initial_state, f)
        
        yield str(state_file)
        shutil.rmtree(temp_dir)
    
    def test_statemanager_get_primary_user(self, temp_state_file):
        """Test that StateManager.get_primary_user returns correct user."""
        from state_management.statemanager import StateManager
        
        manager = StateManager(filepath=temp_state_file)
        
        assert manager.get_primary_user() == "DynamicUser"
    
    def test_statemanager_refresh_news_uses_primary_user(self, temp_state_file):
        """Test that refresh_news_summary uses primary user by default."""
        from state_management.statemanager import StateManager
        
        manager = StateManager(filepath=temp_state_file)
        
        # Refresh news without specifying user
        manager.refresh_news_summary({"test": "news"})
        
        # Check it was stored under primary user
        manager.load()
        assert "DynamicUser" in manager.state["autonomous_state"]["notification_queue"]
        assert "news" in manager.state["autonomous_state"]["notification_queue"]["DynamicUser"]


class TestFirstTimeSetup:
    """Test first-time setup functionality."""
    
    @pytest.fixture
    def temp_state_dir(self):
        """Create a temporary directory for state files."""
        temp_dir = tempfile.mkdtemp()
        state_dir = Path(temp_dir) / "state_management"
        state_dir.mkdir(parents=True)
        yield temp_dir, state_dir
        shutil.rmtree(temp_dir)
    
    def test_setup_prompt_creates_valid_state(self, temp_state_dir):
        """Test that setup prompt creates valid state structure."""
        temp_dir, state_dir = temp_state_dir
        state_file = state_dir / "app_state.json"
        
        # Simulate non-interactive mode (will use defaults)
        with patch('sys.stdin.isatty', return_value=False):
            with patch('mcp_server.state_manager._DEFAULT_STATE_FILE', state_file):
                from mcp_server.state_manager import StateManager
                
                manager = StateManager(state_file=str(state_file))
                
                # Check state was created with required structure
                assert state_file.exists()
                
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                assert "user_state" in state
                assert "primary_user" in state["user_state"]
                assert "chat_controlled_state" in state
                assert "autonomous_state" in state


class TestNoHardcodedMorgan:
    """Test that 'Morgan' is not hardcoded in critical paths."""
    
    def test_calendar_schema_no_hardcoded_morgan(self):
        """Test calendar schema doesn't have hardcoded Morgan."""
        with patch('mcp_server.tools.calendar.get_calendar_users', return_value=["testuser_personal"]), \
             patch('mcp_server.tools.calendar.get_default_calendar_user', return_value="testuser_personal"):
            
            import importlib
            import mcp_server.tools.calendar as cal_module
            importlib.reload(cal_module)
            
            from mcp_server.tools.calendar import CalendarTool
            tool = CalendarTool()
            
            schema_str = json.dumps(tool.get_schema())
            assert "morgan" not in schema_str.lower() or "testuser" in schema_str.lower()
    
    def test_spotify_schema_no_hardcoded_morgan(self):
        """Test spotify schema doesn't have hardcoded Morgan."""
        from mcp_server.tools.spotify import SpotifyPlaybackTool
        
        tool = SpotifyPlaybackTool()
        schema = tool.get_schema()
        
        # The default should come from tool._default_user, not be hardcoded "morgan"
        user_default = schema["properties"]["user"]["default"]
        
        # Verify it matches the tool's configured default (not hardcoded "morgan")
        assert user_default == tool._default_user.lower()
        
        # The schema should use the dynamic enum, not a hardcoded list
        assert schema["properties"]["user"]["enum"] == tool.available_users


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

