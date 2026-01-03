"""
Pytest configuration and shared fixtures for HomeAssist tests.
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def temp_state_dir():
    """Create a temporary state management directory."""
    temp_dir = tempfile.mkdtemp()
    state_dir = Path(temp_dir) / "state_management"
    state_dir.mkdir(parents=True)
    yield state_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_user_state():
    """Return a sample user state configuration."""
    return {
        "user_state": {
            "primary_user": "TestUser",
            "household_members": ["Roommate"],
            "created_at": "2026-01-02T12:00:00",
            "integrations": {
                "spotify": {"enabled": True, "username": "testuser"},
                "calendar": {"enabled": True},
                "smart_home": {"enabled": True}
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


@pytest.fixture
def populated_state_file(temp_state_dir, sample_user_state):
    """Create a populated state file in the temp directory."""
    state_file = temp_state_dir / "app_state.json"
    with open(state_file, 'w') as f:
        json.dump(sample_user_state, f, indent=2)
    return state_file


@pytest.fixture
def mock_user_config_testuser():
    """Mock user config to return 'testuser' for all user functions."""
    with patch('mcp_server.user_config.get_primary_user', return_value="TestUser"), \
         patch('mcp_server.user_config.get_primary_user_lower', return_value="testuser"), \
         patch('mcp_server.user_config.get_available_users', return_value=["TestUser", "Roommate"]), \
         patch('mcp_server.user_config.get_default_spotify_user', return_value="testuser"), \
         patch('mcp_server.user_config.get_spotify_users', return_value=["testuser"]), \
         patch('mcp_server.user_config.get_default_calendar_user', return_value="testuser_personal"), \
         patch('mcp_server.user_config.get_calendar_users', return_value=["testuser_personal", "testuser_school"]), \
         patch('mcp_server.user_config.get_notification_users', return_value=["TestUser", "Roommate"]), \
         patch('mcp_server.user_config.get_default_notification_user', return_value="TestUser"):
        yield

