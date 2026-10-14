"""
Weather Tool using BaseTool.

This module provides weather forecasts for the device's current location
with hourly and daily forecast support. Location is auto-detected via IP
geolocation and cached for future use.
"""

from typing import Any, Dict, Optional, Tuple
import json
import requests
from pathlib import Path

from mcp_server.base_tool import BaseTool
from math import ceil
from datetime import datetime
import re

try:
    from mcp_server.config import LOG_TOOLS
except ImportError:
    LOG_TOOLS = False

# Import weather client with fallback
try:
    from mcp_server.clients.weather_client import WeatherClient
except ImportError:
    WeatherClient = None

# Location cache file path (shared with weather briefing system)
LOCATION_CACHE_FILE = Path(__file__).parent.parent.parent / "scripts" / "scheduled" / "weather_briefing" / "location_cache.json"


class WeatherTool(BaseTool):
    """Weather information and forecasts MCP tool for device's location (hourly/daily)."""

    name = "weather"
    description = (
        "Get weather forecast for your current location. Uses hourly forecasts up to 36 hours "
        "(36 inclusive) or daily forecasts beyond that (rounded up to whole days, max 7). "
        "Location is automatically detected from the device. Units are imperial (°F, mph)."
    )
    version = "2.0.0"

    def __init__(self) -> None:
        """Initialize the weather tool."""
        super().__init__()
        self.weather_client = WeatherClient() if WeatherClient else None
        self._cached_location: Optional[Dict[str, Any]] = None

    def _get_location_from_ip(self) -> Optional[Dict[str, Any]]:
        """
        Get location from IP address using ipinfo.io with fallbacks.
        
        Returns:
            Dict with lat, lon, city, region, zip_code, or None on failure
        """
        # Primary: ipinfo.io
        try:
            response = requests.get("https://ipinfo.io/json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                loc = data.get("loc", "")
                if loc and "," in loc:
                    lat_str, lon_str = loc.split(",")
                    lat, lon = float(lat_str), float(lon_str)
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return {
                            "lat": lat,
                            "lon": lon,
                            "city": data.get("city", "") or "",
                            "region": data.get("region", "") or "",
                            "zip_code": data.get("postal", "") or "",
                            "source": "ipinfo.io",
                        }
        except Exception:
            pass
        
        # Fallback: ip-api.com
        try:
            response = requests.get(
                "http://ip-api.com/json/?fields=lat,lon,city,regionName,zip,status",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") != "fail":
                    lat, lon = data.get("lat"), data.get("lon")
                    if lat is not None and lon is not None:
                        return {
                            "lat": float(lat),
                            "lon": float(lon),
                            "city": data.get("city", "") or "",
                            "region": data.get("regionName", "") or "",
                            "zip_code": str(data.get("zip", "") or ""),
                            "source": "ip-api.com",
                        }
        except Exception:
            pass
        
        return None

    def _load_cached_location(self) -> Optional[Dict[str, Any]]:
        """Load location from cache file if it exists."""
        try:
            if LOCATION_CACHE_FILE.exists():
                with open(LOCATION_CACHE_FILE) as f:
                    data = json.load(f)
                    if "lat" in data and "lon" in data:
                        return data
        except Exception:
            pass
        return None

    def _save_location_to_cache(self, location: Dict[str, Any]) -> None:
        """Save location to cache file."""
        try:
            LOCATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOCATION_CACHE_FILE, "w") as f:
                json.dump(location, f, indent=2)
        except Exception:
            pass

    def _get_device_location(self) -> Optional[Dict[str, Any]]:
        """
        Get the device's current location, using cache or IP geolocation.
        
        Returns:
            Dict with lat, lon, city, region, zip_code, or None on failure
        """
        # Try memory cache first
        if self._cached_location:
            return self._cached_location
        
        # Try file cache
        cached = self._load_cached_location()
        if cached:
            self._cached_location = cached
            return cached
        
        # Detect from IP
        location = self._get_location_from_ip()
        if location:
            self._cached_location = location
            self._save_location_to_cache(location)
            return location
        
        return None

    def _get_location_display_name(self, location: Dict[str, Any]) -> str:
        """Get a human-readable location name."""
        city = location.get("city", "")
        region = location.get("region", "")
        if city and region:
            return f"{city}, {region}"
        elif city:
            return city
        elif location.get("zip_code"):
            return f"ZIP {location['zip_code']}"
        else:
            return f"({location['lat']:.2f}, {location['lon']:.2f})"

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema describing input parameters for this tool."""
        return {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "description": (
                        "Timeframe in hours (1-168). Uses hourly forecasts when <= 36 hours. "
                        "If > 36, rounds up to whole days and uses daily forecasts."
                    ),
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "description": (
                        "Timeframe in days (1-7). If days*24 <= 36, hourly forecasts are used; otherwise daily."
                    ),
                },
                "specific_date": {
                    "type": "string",
                    "description": (
                        "Get weather for a specific date. Format: 'YYYY-MM-DD' or relative terms like "
                        "'today', 'tomorrow', 'monday', 'tuesday', etc. Must be within the next 7 days."
                    ),
                },
            },
            "oneOf": [
                {"required": ["hours"]},
                {"required": ["days"]},
                {"required": ["specific_date"]},
            ],
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the weather tool with the provided parameters."""
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Weather -- %s", params)

            # Get device location
            location = self._get_device_location()
            if not location:
                return {
                    "success": False,
                    "error": "Could not detect device location. Check network connection.",
                }
            
            location_name = self._get_location_display_name(location)
            lat, lon = location["lat"], location["lon"]

            # Validate inputs
            validation_error = self._validate_parameters(params)
            if validation_error:
                return validation_error

            hours = params.get("hours")
            days = params.get("days")
            specific_date = params.get("specific_date")

            # Handle specific_date if provided
            if specific_date:
                date_days = self._parse_specific_date(specific_date)
                if date_days is None:
                    return {
                        "success": False,
                        "error": f"Invalid or out-of-range date: '{specific_date}'. Must be within next 7 days.",
                        "location": location_name,
                    }
                mode = "specific_date"
                normalized_hours, normalized_days = None, date_days
                daily = self.weather_client.fetch_open_meteo_daily(lat, lon, days=date_days, units="imperial")
                # Extract just the specific day's data
                if daily and isinstance(daily, dict):
                    day_idx = date_days - 1
                    specific_daily = {}
                    for key, values in daily.items():
                        if isinstance(values, list) and len(values) > day_idx:
                            specific_daily[key] = [values[day_idx]]
                        else:
                            specific_daily[key] = values
                    forecast_data = {"daily": specific_daily}
                else:
                    forecast_data = {"daily": daily}
            else:
                # Normalize timeframe and mode for hours/days
                mode, normalized_hours, normalized_days = self._normalize_timeframe(hours, days)
                
                # Fetch data per mode using imperial units
                if mode == "hourly":
                    hourly = self.weather_client.fetch_open_meteo_hourly(lat, lon, hours=normalized_hours, units="imperial")
                    forecast_data = {"hourly": hourly}
                else:
                    daily = self.weather_client.fetch_open_meteo_daily(lat, lon, days=normalized_days, units="imperial")
                    forecast_data = {"daily": daily}

            response: Dict[str, Any] = {
                "success": True,
                "location": location_name,
                "coordinates": {"lat": lat, "lon": lon},
                "zip_code": location.get("zip_code"),
                "mode": mode,
                "timeframe_hours": normalized_hours if mode == "hourly" else None,
                "timeframe_days": normalized_days if mode in ("daily", "specific_date") else None,
                "specific_date": specific_date if specific_date else None,
                "units": "imperial",
                "forecast": forecast_data,
                "timestamp": self._get_current_timestamp(),
            }

            return response

        except Exception as e:
            self.logger.error(f"Error executing weather tool: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _normalize_timeframe(self, hours: Optional[int], days: Optional[int]) -> Tuple[str, Optional[int], Optional[int]]:
        """
        Apply the 36-hour rule and max day cap.
        Returns: (mode, normalized_hours, normalized_days)
        mode in {"hourly", "daily"}
        """
        if hours is not None and days is not None:
            raise ValueError("Provide either hours or days, not both")

        if hours is not None:
            h = max(1, min(168, int(hours)))
            if h <= 36:
                return "hourly", h, None
            d = ceil(h / 24)
            d = max(1, min(7, d))
            return "daily", None, d

        if days is not None:
            d = max(1, min(7, int(days)))
            if d * 24 <= 36:
                h = d * 24
                return "hourly", h, None
            return "daily", None, d

        raise ValueError("Missing timeframe: provide hours or days")

    def _validate_parameters(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate parameters; return error dict or None if valid."""
        hours = params.get("hours")
        days = params.get("days")
        specific_date = params.get("specific_date")

        # Exactly one of hours, days, or specific_date must be provided
        provided_params = sum([hours is not None, days is not None, specific_date is not None])
        if provided_params != 1:
            return {
                "success": False,
                "error": "Provide exactly one timeframe: either 'hours' (1-168), 'days' (1-7), or 'specific_date' (within next 7 days)",
            }

        if hours is not None:
            try:
                h = int(hours)
            except Exception:
                return {"success": False, "error": "'hours' must be an integer"}
            if h < 1 or h > 168:
                return {"success": False, "error": "'hours' must be between 1 and 168"}

        if days is not None:
            try:
                d = int(days)
            except Exception:
                return {"success": False, "error": "'days' must be an integer"}
            if d < 1 or d > 7:
                return {"success": False, "error": "'days' must be between 1 and 7"}

        return None

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as an ISO 8601 string."""
        try:
            return datetime.now().isoformat()
        except Exception:
            return "unknown"
    
    def _parse_specific_date(self, date_str: str) -> Optional[int]:
        """
        Parse a specific date string and return the number of days from today.
        Returns None if the date is invalid or not within 7 days.
        """
        today = datetime.now().date()
        date_str_lower = date_str.lower().strip()
        
        if date_str_lower == 'today':
            return 1
        elif date_str_lower == 'tomorrow':
            return 2
        
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if date_str_lower in weekdays:
            target_weekday = weekdays.index(date_str_lower)
            current_weekday = today.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7
            if days_ahead > 7:
                return None
            return days_ahead + 1
        
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                delta = (target_date - today).days
                if 0 <= delta <= 7:
                    return delta + 1
                else:
                    return None
            except ValueError:
                return None
        
        return None
