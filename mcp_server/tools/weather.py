"""
Weather Tool using BaseTool.

This module provides weather forecasts and information for configured regions
with hourly and daily forecast support.
"""

from typing import Any, Dict, Optional, List

from mcp_server.base_tool import BaseTool
from math import ceil
from datetime import datetime, timedelta
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


class WeatherTool(BaseTool):
    """Weather information and forecasts MCP tool for regions (hourly/daily)."""

    name = "weather"
    description = (
        "Get weather for a specific region using hourly forecasts up to 36 hours "
        "(36 inclusive) or daily forecasts beyond that (rounded up to whole days, max 7). "
        "Units are imperial. Regions are preconfigured sets of ZIP codes."
    )
    version = "1.0.0"

    # Placeholder ZIP code mappings (edit these lists later)
    REGION_ZIP_MAP: Dict[str, List[str]] = {
        "ithaca": ["14850", "14853"],
        "new_york": ["10001", "10002"],
    }

    def __init__(self) -> None:
        """Initialize the weather tool."""
        super().__init__()
        self.weather_client = WeatherClient() if WeatherClient else None
        self.available_regions = list(self.REGION_ZIP_MAP.keys())

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema describing input parameters for this tool."""
        return {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        f"Target region for weather. Allowed: {self.available_regions}. "
                        "Each region returns data for its predefined ZIP codes only."
                    ),
                    "enum": self.available_regions,
                },
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "description": (
                        "Timeframe in hours (1-168). Use hourly forecasts when <= 36 hours (36 inclusive). "
                        "If > 36, this will be rounded up to whole days and daily forecasts will be used."
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
            "required": ["region"],
            "oneOf": [
                {"required": ["region", "hours"]},
                {"required": ["region", "days"]},
                {"required": ["region", "specific_date"]},
            ],
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the weather tool with the provided parameters."""
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Weather -- %s", params)

            # Validate inputs (also normalizes region to lowercase)
            validation_error = self._validate_parameters(params)
            if validation_error:
                return validation_error

            # Get normalized values after validation
            region = params.get("region")  # Now lowercase after validation
            hours = params.get("hours")
            days = params.get("days")
            specific_date = params.get("specific_date")

            # Determine region -> zip codes
            zip_codes = self.REGION_ZIP_MAP.get(region, [])

            # Handle specific_date if provided
            if specific_date:
                date_days = self._parse_specific_date(specific_date)
                if date_days is None:
                    return {
                        "success": False,
                        "error": f"Invalid or out-of-range date: '{specific_date}'. Must be within next 7 days.",
                        "region": region,
                    }
                # Use specific date forecast method
                mode = "specific_date"
                normalized_hours, normalized_days = None, date_days
                results = self.weather_client.get_specific_date_forecast_for_zipcodes(
                    zip_codes, day_index=date_days, units="imperial"
                )
            else:
                # Normalize timeframe and mode for hours/days
                mode, normalized_hours, normalized_days = self._normalize_timeframe(hours, days)
                
                # Fetch data per mode using imperial units
                if mode == "hourly":
                    results = self.weather_client.get_hourly_forecast_for_zipcodes(
                        zip_codes, hours=normalized_hours, units="imperial"
                    )
                else:
                    results = self.weather_client.get_daily_forecast_for_zipcodes(
                        zip_codes, days=normalized_days, units="imperial"
                    )

            # Determine success state
            success_count = sum(1 for v in results.values() if isinstance(v, dict) and not v.get("error"))
            failure_count = len(results) - success_count

            response: Dict[str, Any] = {
                "success": success_count > 0 and failure_count == 0,
                "region": region,
                "zip_codes": zip_codes,
                "mode": mode,
                "timeframe_hours": normalized_hours if mode == "hourly" else None,
                "timeframe_days": normalized_days if mode == "daily" else None,
                "specific_date": specific_date if specific_date else None,
                "units": "imperial",
                "results_by_zip": results,
                "successful_zips": success_count,
                "failed_zips": failure_count,
                "timestamp": self._get_current_timestamp(),
            }

            return response

        except Exception as e:
            self.logger.error(f"Error executing weather tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "region": params.get("region"),
            }

    # Internal helpers
    def _normalize_timeframe(self, hours: Optional[int], days: Optional[int]) -> (str, Optional[int], Optional[int]):
        """
        Apply the 36-hour rule and max day cap.
        Returns: (mode, normalized_hours, normalized_days)
        mode in {"hourly", "daily"}
        """
        if hours is not None and days is not None:
            # Prefer explicit error; validation prevents this already
            raise ValueError("Provide either hours or days, not both")

        if hours is not None:
            h = max(1, min(168, int(hours)))
            if h <= 36:  # 36 inclusive uses hourly
                return "hourly", h, None
            # >36: round up to days
            d = ceil(h / 24)
            d = max(1, min(7, d))
            return "daily", None, d

        if days is not None:
            d = max(1, min(7, int(days)))
            if d * 24 <= 36:
                # Use hourly (convert days to hours exactly)
                h = d * 24
                return "hourly", h, None
            return "daily", None, d

        # Should not reach here due to validation
        raise ValueError("Missing timeframe: provide hours or days")

    def _validate_parameters(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and normalize parameters; return error dict or None if valid."""
        region = params.get("region")
        hours = params.get("hours")
        days = params.get("days")
        specific_date = params.get("specific_date")

        # Normalize region to lowercase for case-insensitive matching
        if region:
            region = region.lower().strip()
            params["region"] = region  # Update params with normalized value

        # Region must be provided and valid; no fallback to other regions
        if not region or region not in self.available_regions:
            return {
                "success": False,
                "error": f"Invalid or missing region. Allowed regions: {self.available_regions}",
            }

        # Exactly one of hours, days, or specific_date must be provided
        provided_params = sum([hours is not None, days is not None, specific_date is not None])
        if provided_params != 1:
            return {
                "success": False,
                "error": "Provide exactly one timeframe: either 'hours' (1-168), 'days' (1-7), or 'specific_date' (within next 7 days)",
            }

        # Bounds checking (coarse; detailed normalization happens later)
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

        # Region ZIPs must exist (no fallback)
        zip_list = self.REGION_ZIP_MAP.get(region, [])
        if not zip_list:
            return {
                "success": False,
                "error": f"No ZIP codes configured for region '{region}'. Configure REGION_ZIP_MAP.",
            }

        return None

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as an ISO 8601 string."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except Exception:
            return "unknown"
    
    def _parse_specific_date(self, date_str: str) -> Optional[int]:
        """
        Parse a specific date string and return the number of days from today.
        Returns None if the date is invalid or not within 7 days.
        
        Supported formats:
        - 'today': 0 days from now
        - 'tomorrow': 1 day from now
        - 'monday', 'tuesday', etc.: Next occurrence of that day
        - 'YYYY-MM-DD': Specific date
        """
        today = datetime.now().date()
        date_str_lower = date_str.lower().strip()
        
        # Handle relative terms
        if date_str_lower == 'today':
            return 1  # For API, day 1 is today
        elif date_str_lower == 'tomorrow':
            return 2  # Day 2 is tomorrow
        
        # Handle weekday names (always future occurrences)
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if date_str_lower in weekdays:
            target_weekday = weekdays.index(date_str_lower)
            current_weekday = today.weekday()
            
            # Calculate days until target weekday (always in the future)
            days_ahead = (target_weekday - current_weekday) % 7
            if days_ahead == 0:  # If it's today's weekday, use next week
                days_ahead = 7
            
            # Check if within 7-day limit
            if days_ahead > 7:
                return None
            
            # Return 1-based index (days_ahead + 1 since API uses 1 for today)
            return days_ahead + 1
        
        # Handle YYYY-MM-DD format
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                delta = (target_date - today).days
                
                # Must be today or within next 7 days (0 = today, 1-7 = future days)
                if 0 <= delta <= 7:
                    return delta + 1  # API uses 1-based indexing (1 = today, 2 = tomorrow, etc.)
                else:
                    return None
            except ValueError:
                return None
        
        return None


