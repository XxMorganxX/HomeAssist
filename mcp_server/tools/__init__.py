"""
MCP Tools Package

This package contains all MCP tools that use the BaseTool
base class for better parameter descriptions and type safety.

All tools in this package provide:
- Rich parameter descriptions in function docstrings
- Better type annotations with Literal types
- Comprehensive documentation with usage examples
- Enhanced error handling and validation
"""

import platform

# NOTE: Imports are intentionally lazy to avoid forcing all dependencies
# to be installed. The tool registry imports each tool file directly as needed.
# 
# If you need to import a specific tool, do:
#   from mcp_server.tools.state_tool import StateTool

__all__ = [
    'StateTool',
    'GetNotificationsTool',
    'KasaLightingTool',
    'SpotifyPlaybackTool',
    'CalendarTool',
    'GoogleSearchTool',
    'WeatherTool',
]

# Conditionally add macOS-only tools
if platform.system() == "Darwin":
    __all__.append('SendSMSTool')
