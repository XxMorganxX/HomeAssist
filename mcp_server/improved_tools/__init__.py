"""
Improved MCP Tools Package

This package contains enhanced versions of all MCP tools that use the ImprovedBaseTool
base class for better parameter descriptions and type safety.

All tools in this package provide:
- Rich parameter descriptions in function docstrings
- Better type annotations with Literal types
- Comprehensive documentation with usage examples
- Enhanced error handling and validation
"""

# Import all improved tools for easy access
from .improved_state_tool import ImprovedStateTool
from .improved_notifications import ImprovedGetNotificationsTool
from .improved_light_control import ImprovedBatchLightControlTool
from .improved_lighting_scene import ImprovedLightingSceneTool
from .improved_spotify import ImprovedSpotifyPlaybackTool
from .improved_calendar import ImprovedCalendarTool

__all__ = [
    'ImprovedStateTool',
    'ImprovedGetNotificationsTool', 
    'ImprovedBatchLightControlTool',
    'ImprovedLightingSceneTool',
    'ImprovedSpotifyPlaybackTool',
    'ImprovedCalendarTool'
]