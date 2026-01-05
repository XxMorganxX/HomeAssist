"""
Weather Briefing

Scheduled process to analyze weather forecasts and create briefing 
announcements only when unusual weather conditions are detected.

Unusual conditions include:
- Precipitation (rain, snow, thunderstorms)
- Temperature extremes (unusually hot or cold for the season)
- Severe weather warnings

Location handling:
- Auto-detects from IP address if not configured
- Caches location for accuracy (can be manually set)
- Uses coordinates directly for precise weather data
"""

from .analyzer import (
    WeatherAnalyzer,
    GeoLocation,
    auto_detect_location,
    set_home_location,
)
from .main import main

__all__ = [
    "WeatherAnalyzer",
    "GeoLocation", 
    "auto_detect_location",
    "set_home_location",
    "main",
]

