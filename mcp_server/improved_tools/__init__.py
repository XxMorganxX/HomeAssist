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

# Import only existing improved tools for easy access
from .improved_state_tool import ImprovedStateTool
from .improved_notifications import ImprovedGetNotificationsTool
from .improved_kasa_lighting import ImprovedKasaLightingTool
from .improved_spotify import ImprovedSpotifyPlaybackTool
from .improved_calendar import ImprovedCalendarTool
from .improved_google_search import ImprovedGoogleSearchTool
from .improved_weather import ImprovedWeatherTool

__all__ = [
    'ImprovedStateTool',
    'ImprovedGetNotificationsTool',
    'ImprovedKasaLightingTool',
    'ImprovedSpotifyPlaybackTool',
    'ImprovedCalendarTool',
    'ImprovedGoogleSearchTool',
    'ImprovedWeatherTool',
]