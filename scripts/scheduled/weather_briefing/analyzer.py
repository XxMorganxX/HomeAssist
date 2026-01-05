"""
Weather Briefing Analyzer

Analyzes weather forecasts to detect unusual conditions that warrant
briefing the user. Only creates alerts for actionable weather events.

Weather Codes (WMO Standard via Open-Meteo):
    0: Clear sky
    1-3: Mainly clear, partly cloudy, overcast
    45, 48: Fog
    51-57: Drizzle (including freezing)
    61-67: Rain (including freezing)
    71-77: Snow
    80-82: Rain showers
    85-86: Snow showers
    95-99: Thunderstorms (with/without hail)
"""

import json
import os
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Location cache file - stores verified location for reuse
LOCATION_CACHE_FILE = Path(__file__).parent / "location_cache.json"


@dataclass
class GeoLocation:
    """Geographic location with coordinates and metadata."""
    lat: float
    lon: float
    city: str = ""
    region: str = ""
    country: str = ""
    zip_code: str = ""
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "zip_code": self.zip_code,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoLocation":
        return cls(
            lat=data.get("lat", 0),
            lon=data.get("lon", 0),
            city=data.get("city", ""),
            region=data.get("region", ""),
            country=data.get("country", ""),
            zip_code=data.get("zip_code", ""),
            source=data.get("source", "cache"),
        )
    
    def display_name(self) -> str:
        if self.city and self.region:
            return f"{self.city}, {self.region}"
        elif self.city:
            return self.city
        elif self.zip_code:
            return f"ZIP {self.zip_code}"
        else:
            return f"({self.lat:.2f}, {self.lon:.2f})"


def _load_cached_location() -> Optional[GeoLocation]:
    """Load location from cache file if it exists."""
    try:
        if LOCATION_CACHE_FILE.exists():
            with open(LOCATION_CACHE_FILE) as f:
                data = json.load(f)
                loc = GeoLocation.from_dict(data)
                print(f"📍 Using cached location: {loc.display_name()}")
                return loc
    except Exception:
        pass
    return None


def _save_location_to_cache(location: GeoLocation) -> None:
    """Save location to cache file for future use."""
    try:
        with open(LOCATION_CACHE_FILE, "w") as f:
            json.dump(location.to_dict(), f, indent=2)
        print(f"   💾 Location cached for future use")
    except Exception as e:
        print(f"   ⚠️  Could not cache location: {e}")


def get_location_from_ip() -> Optional[GeoLocation]:
    """
    Get location from IP address using multiple geolocation services.
    
    Uses coordinates (lat/lon) as primary data since they're more reliable
    than ZIP codes from IP geolocation.
    
    Returns:
        GeoLocation object, or None on failure
    """
    # Services ordered by reliability for coordinates
    services = [
        {
            "name": "ipwho.is",
            "url": "https://ipwho.is/",
            "lat_key": "latitude",
            "lon_key": "longitude",
            "city_key": "city",
            "region_key": "region",
            "country_key": "country",
            "zip_key": "postal",
        },
        {
            "name": "ip-api.com",
            "url": "http://ip-api.com/json/?fields=lat,lon,city,regionName,country,zip,status",
            "lat_key": "lat",
            "lon_key": "lon",
            "city_key": "city",
            "region_key": "regionName",
            "country_key": "country",
            "zip_key": "zip",
        },
        {
            "name": "ipapi.co",
            "url": "https://ipapi.co/json/",
            "lat_key": "latitude",
            "lon_key": "longitude",
            "city_key": "city",
            "region_key": "region",
            "country_key": "country_name",
            "zip_key": "postal",
        },
    ]
    
    for service in services:
        try:
            response = requests.get(service["url"], timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check for error responses
                if data.get("status") == "fail" or data.get("success") is False:
                    continue
                
                lat = data.get(service["lat_key"])
                lon = data.get(service["lon_key"])
                
                # Coordinates are required
                if lat is None or lon is None:
                    continue
                
                # Validate coordinates are reasonable
                if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
                    continue
                
                return GeoLocation(
                    lat=float(lat),
                    lon=float(lon),
                    city=data.get(service["city_key"], "") or "",
                    region=data.get(service["region_key"], "") or "",
                    country=data.get(service["country_key"], "") or "",
                    zip_code=str(data.get(service["zip_key"], "") or ""),
                    source=service["name"],
                )
                
        except Exception:
            continue  # Try next service
    
    return None


def auto_detect_location(use_cache: bool = True) -> Optional[GeoLocation]:
    """
    Automatically detect location with caching support.
    
    Args:
        use_cache: If True, use cached location if available
        
    Returns:
        GeoLocation object, or None if detection fails
    """
    # Try cache first
    if use_cache:
        cached = _load_cached_location()
        if cached:
            return cached
    
    # Detect from IP
    print("🔍 Detecting location from IP address...")
    location = get_location_from_ip()
    
    if location:
        print(f"📍 Detected: {location.display_name()} (via {location.source})")
        print(f"   Coordinates: ({location.lat:.4f}, {location.lon:.4f})")
        
        # Cache for future use
        _save_location_to_cache(location)
        return location
    
    print("⚠️  Could not detect location from IP")
    return None


def auto_detect_zip_code() -> Optional[str]:
    """
    Automatically detect ZIP code from device's IP address.
    
    Note: ZIP codes from IP geolocation are often inaccurate.
    Consider using auto_detect_location() and coordinates instead.
    
    Returns:
        5-digit US ZIP code string, or None if detection fails
    """
    location = auto_detect_location()
    if location and location.zip_code:
        return location.zip_code
    return None


def set_home_location(lat: float, lon: float, city: str = "", region: str = "") -> GeoLocation:
    """
    Manually set and cache the home location.
    
    Use this to set an accurate location that will be used for all future
    weather briefings.
    
    Args:
        lat: Latitude
        lon: Longitude  
        city: City name (optional, for display)
        region: State/region (optional, for display)
        
    Returns:
        The saved GeoLocation
        
    Example:
        # Set location to Glen Head, NY
        set_home_location(40.8351, -73.6265, "Glen Head", "New York")
    """
    location = GeoLocation(
        lat=lat,
        lon=lon,
        city=city,
        region=region,
        source="manual",
    )
    _save_location_to_cache(location)
    print(f"✅ Home location set to: {location.display_name()}")
    print(f"   Coordinates: ({lat}, {lon})")
    return location


class AlertType(Enum):
    """Types of weather alerts we can generate."""
    RAIN = "rain"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    FREEZING = "freezing"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    HIGH_WIND = "high_wind"


@dataclass
class WeatherAlert:
    """A detected weather anomaly that warrants a briefing."""
    alert_type: AlertType
    date: str  # ISO date string
    day_name: str  # "Today", "Tomorrow", "Wednesday", etc.
    severity: str  # "moderate", "significant", "severe"
    message: str  # Human-readable description
    details: Dict[str, Any]  # Raw data for the alert


# Weather codes that indicate precipitation
RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}  # Drizzle and rain
SNOW_CODES = {71, 73, 75, 77, 85, 86}  # Snow
FREEZING_CODES = {56, 57, 66, 67}  # Freezing precipitation
THUNDERSTORM_CODES = {95, 96, 99}  # Thunderstorms
FOG_CODES = {45, 48}  # Fog

# Precipitation codes that warrant alerts
PRECIPITATION_CODES = RAIN_CODES | SNOW_CODES | FREEZING_CODES | THUNDERSTORM_CODES


class WeatherAnalyzer:
    """
    Analyzes weather data to detect unusual conditions.
    
    Configuration via environment variables:
        WEATHER_ZIP_CODE: User's ZIP code (default: 10001)
        WEATHER_TEMP_UNIT: "fahrenheit" or "celsius" (default: fahrenheit)
        
    Temperature thresholds (Fahrenheit, adjusted for season):
        Summer (Jun-Aug): Hot > 95°F, Cold < 60°F
        Winter (Dec-Feb): Hot > 55°F, Cold < 20°F
        Spring/Fall: Hot > 85°F, Cold < 35°F
    """
    
    # Seasonal temperature thresholds (Fahrenheit)
    TEMP_THRESHOLDS = {
        "winter": {"extreme_cold": 20, "cold": 32, "hot": 55, "extreme_heat": 65},
        "spring": {"extreme_cold": 28, "cold": 40, "hot": 80, "extreme_heat": 90},
        "summer": {"extreme_cold": 50, "cold": 60, "hot": 92, "extreme_heat": 100},
        "fall": {"extreme_cold": 28, "cold": 40, "hot": 80, "extreme_heat": 90},
    }
    
    # Precipitation thresholds (inches)
    PRECIP_THRESHOLDS = {
        "light": 0.1,
        "moderate": 0.25,
        "heavy": 0.5,
    }
    
    # Wind thresholds (mph)
    WIND_THRESHOLDS = {
        "breezy": 15,
        "windy": 25,
        "high_wind": 40,
    }
    
    def __init__(self, zip_code: Optional[str] = None, units: str = "imperial"):
        """
        Initialize the weather analyzer.
        
        Args:
            zip_code: US ZIP code for weather location
            units: "imperial" (Fahrenheit, mph) or "metric" (Celsius, km/h)
        """
        self.zip_code = zip_code or os.getenv("WEATHER_ZIP_CODE", "10001")
        self.units = units
        self._current_season = self._get_season()
    
    @staticmethod
    def _get_season() -> str:
        """Determine current season based on month."""
        month = datetime.now().month
        if month in (12, 1, 2):
            return "winter"
        elif month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        else:
            return "fall"
    
    @staticmethod
    def _get_day_name(date_str: str) -> str:
        """Convert ISO date to friendly day name."""
        date = datetime.fromisoformat(date_str)
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        if date.date() == today:
            return "Today"
        elif date.date() == tomorrow:
            return "Tomorrow"
        else:
            return date.strftime("%A")  # "Wednesday", etc.
    
    def _get_temp_thresholds(self) -> Dict[str, float]:
        """Get temperature thresholds for current season."""
        return self.TEMP_THRESHOLDS[self._current_season]
    
    def _classify_precipitation(self, weather_code: int, precip_amount: float, precip_prob: int) -> Optional[Tuple[AlertType, str]]:
        """
        Classify precipitation type and severity.
        
        Returns:
            Tuple of (AlertType, severity) or None if no alert needed
        """
        if weather_code not in PRECIPITATION_CODES:
            return None
        
        # Low probability = no alert
        if precip_prob < 50:
            return None
        
        # Determine type
        if weather_code in THUNDERSTORM_CODES:
            alert_type = AlertType.THUNDERSTORM
            severity = "severe" if weather_code in (96, 99) else "significant"  # Hail is severe
        elif weather_code in SNOW_CODES:
            alert_type = AlertType.SNOW
            if precip_amount >= self.PRECIP_THRESHOLDS["heavy"]:
                severity = "severe"
            elif precip_amount >= self.PRECIP_THRESHOLDS["moderate"]:
                severity = "significant"
            else:
                severity = "moderate"
        elif weather_code in FREEZING_CODES:
            alert_type = AlertType.FREEZING
            severity = "significant"  # Freezing precip is always significant
        elif weather_code in RAIN_CODES:
            alert_type = AlertType.RAIN
            if precip_amount >= self.PRECIP_THRESHOLDS["heavy"]:
                severity = "significant"
            elif precip_amount >= self.PRECIP_THRESHOLDS["moderate"]:
                severity = "moderate"
            else:
                # Light rain with low probability - skip
                if precip_prob < 70:
                    return None
                severity = "moderate"
        else:
            return None
        
        return (alert_type, severity)
    
    def _check_temperature_extreme(self, temp_high: float, temp_low: float) -> Optional[Tuple[AlertType, str]]:
        """
        Check if temperature is extreme for the season.
        
        Returns:
            Tuple of (AlertType, severity) or None if no alert needed
        """
        thresholds = self._get_temp_thresholds()
        
        # Check extreme heat
        if temp_high >= thresholds["extreme_heat"]:
            return (AlertType.EXTREME_HEAT, "severe")
        elif temp_high >= thresholds["hot"]:
            return (AlertType.EXTREME_HEAT, "moderate")
        
        # Check extreme cold
        if temp_low <= thresholds["extreme_cold"]:
            return (AlertType.EXTREME_COLD, "severe")
        elif temp_low <= thresholds["cold"]:
            return (AlertType.EXTREME_COLD, "moderate")
        
        return None
    
    def _check_wind(self, wind_speed: float, wind_gusts: float) -> Optional[Tuple[AlertType, str]]:
        """
        Check for high wind conditions.
        
        Returns:
            Tuple of (AlertType, severity) or None if no alert needed
        """
        max_wind = max(wind_speed, wind_gusts)
        
        if max_wind >= self.WIND_THRESHOLDS["high_wind"]:
            return (AlertType.HIGH_WIND, "significant")
        elif max_wind >= self.WIND_THRESHOLDS["windy"]:
            return (AlertType.HIGH_WIND, "moderate")
        
        return None
    
    def _build_alert_message(self, alert_type: AlertType, severity: str, details: Dict[str, Any]) -> str:
        """Build a human-readable alert message."""
        day_name = details.get("day_name", "")
        
        if alert_type == AlertType.RAIN:
            precip = details.get("precipitation_sum", 0)
            prob = details.get("precipitation_probability", 0)
            if severity == "significant":
                return f"{day_name}: Heavy rain expected ({prob}% chance, ~{precip:.1f}\" precipitation). Consider indoor plans."
            else:
                return f"{day_name}: Rain likely ({prob}% chance). You may want to bring an umbrella."
        
        elif alert_type == AlertType.SNOW:
            precip = details.get("precipitation_sum", 0)
            prob = details.get("precipitation_probability", 0)
            if severity == "severe":
                return f"{day_name}: Heavy snow expected ({prob}% chance). Roads may be hazardous."
            elif severity == "significant":
                return f"{day_name}: Snow expected ({prob}% chance). Plan for winter conditions."
            else:
                return f"{day_name}: Light snow possible ({prob}% chance)."
        
        elif alert_type == AlertType.THUNDERSTORM:
            if severity == "severe":
                return f"{day_name}: Severe thunderstorms with possible hail expected. Stay weather-aware."
            else:
                return f"{day_name}: Thunderstorms expected. Consider adjusting outdoor plans."
        
        elif alert_type == AlertType.FREEZING:
            return f"{day_name}: Freezing precipitation expected. Roads and walkways may be icy."
        
        elif alert_type == AlertType.EXTREME_HEAT:
            temp = details.get("temperature_high", 0)
            if severity == "severe":
                return f"{day_name}: Extreme heat ({temp:.0f}°F). Stay hydrated and limit outdoor activity."
            else:
                return f"{day_name}: Unusually hot ({temp:.0f}°F) for this time of year."
        
        elif alert_type == AlertType.EXTREME_COLD:
            temp = details.get("temperature_low", 0)
            if severity == "severe":
                return f"{day_name}: Dangerously cold ({temp:.0f}°F). Dress warmly and limit exposure."
            else:
                return f"{day_name}: Cold temperatures ({temp:.0f}°F). Bundle up if heading out."
        
        elif alert_type == AlertType.HIGH_WIND:
            wind = details.get("wind_speed_max", 0)
            gusts = details.get("wind_gusts_max", 0)
            if severity == "significant":
                return f"{day_name}: High winds expected (gusts up to {gusts:.0f} mph). Secure loose items."
            else:
                return f"{day_name}: Windy conditions ({wind:.0f} mph winds)."
        
        return f"{day_name}: Unusual weather expected."
    
    def _compute_week_context(self, daily: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute weekly averages and patterns to detect anomalies.
        
        Uses all available forecast days to establish what's "normal"
        for this week, making today's alerts relative to local context.
        
        Returns:
            Dict with avg_high, avg_low, typical_precip_prob, typical_wind, etc.
        """
        temp_highs = daily.get("temperature_2m_max", [])
        temp_lows = daily.get("temperature_2m_min", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        wind_gusts = daily.get("wind_gusts_10m_max", [])
        
        # Filter out None values
        valid_highs = [t for t in temp_highs if t is not None]
        valid_lows = [t for t in temp_lows if t is not None]
        valid_precip = [p for p in precip_probs if p is not None]
        valid_wind = [w for w in wind_gusts if w is not None]
        
        return {
            "avg_high": sum(valid_highs) / len(valid_highs) if valid_highs else None,
            "avg_low": sum(valid_lows) / len(valid_lows) if valid_lows else None,
            "max_high": max(valid_highs) if valid_highs else None,
            "min_low": min(valid_lows) if valid_lows else None,
            "avg_precip_prob": sum(valid_precip) / len(valid_precip) if valid_precip else 0,
            "avg_wind_gust": sum(valid_wind) / len(valid_wind) if valid_wind else 0,
            "days_with_precip": sum(1 for p in valid_precip if p >= 50),
        }
    
    def _is_temperature_anomaly_vs_week(
        self, 
        temp_high: float, 
        temp_low: float, 
        context: Dict[str, Any]
    ) -> Optional[Tuple[AlertType, str, str]]:
        """
        Check if today's temperature is anomalous compared to the week's forecast.
        
        This catches cases like "today is 15°F colder than the rest of the week"
        which may not hit seasonal thresholds but is still notable.
        
        Returns:
            Tuple of (AlertType, severity, reason) or None
        """
        avg_high = context.get("avg_high")
        avg_low = context.get("avg_low")
        
        if avg_high is None or avg_low is None:
            return None
        
        # Check if today is significantly colder than the week
        high_diff = avg_high - temp_high
        low_diff = avg_low - temp_low
        
        # Today is much colder than the week average
        if low_diff >= 15:  # 15°F colder than average
            return (AlertType.EXTREME_COLD, "significant", f"{low_diff:.0f}°F below this week's average")
        elif low_diff >= 10:
            return (AlertType.EXTREME_COLD, "moderate", f"{low_diff:.0f}°F below this week's average")
        
        # Today is much hotter than the week average
        if high_diff <= -15:  # 15°F hotter than average
            return (AlertType.EXTREME_HEAT, "significant", f"{abs(high_diff):.0f}°F above this week's average")
        elif high_diff <= -10:
            return (AlertType.EXTREME_HEAT, "moderate", f"{abs(high_diff):.0f}°F above this week's average")
        
        return None
    
    def analyze_forecast(self, forecast_data: Dict[str, Any], days_to_check: int = 1) -> List[WeatherAlert]:
        """
        Analyze weather forecast and return list of alerts for TODAY only.
        
        Uses the broader 7-day forecast to establish context and detect
        anomalies, but only generates alerts for the next 24 hours.
        
        Args:
            forecast_data: Raw forecast from WeatherClient.get_daily_forecast()
            days_to_check: Days to alert on (default 1 = today only)
            
        Returns:
            List of WeatherAlert objects for detected anomalies
        """
        alerts = []
        daily = forecast_data.get("daily", {})
        
        if not daily:
            print("⚠️  No daily forecast data available")
            return alerts
        
        dates = daily.get("time", [])
        weather_codes = daily.get("weathercode", [])
        temp_highs = daily.get("temperature_2m_max", [])
        temp_lows = daily.get("temperature_2m_min", [])
        precip_sums = daily.get("precipitation_sum", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        wind_speeds = daily.get("wind_speed_10m_max", [])
        wind_gusts = daily.get("wind_gusts_10m_max", [])
        
        # Compute weekly context for anomaly detection
        week_context = self._compute_week_context(daily)
        print(f"📊 Week context: avg high {week_context.get('avg_high', 0):.1f}°F, "
              f"avg low {week_context.get('avg_low', 0):.1f}°F, "
              f"{week_context.get('days_with_precip', 0)} days with precip")
        
        # Only check TODAY (next 24 hours) for alerts
        # But we use the full week's data for context
        for i in range(min(days_to_check, len(dates))):
            date = dates[i]
            day_name = self._get_day_name(date)
            
            # Get values for this day
            weather_code = weather_codes[i] if i < len(weather_codes) else 0
            temp_high = temp_highs[i] if i < len(temp_highs) else None
            temp_low = temp_lows[i] if i < len(temp_lows) else None
            precip_sum = precip_sums[i] if i < len(precip_sums) else 0
            precip_prob = precip_probs[i] if i < len(precip_probs) else 0
            wind_speed = wind_speeds[i] if i < len(wind_speeds) else 0
            wind_gust = wind_gusts[i] if i < len(wind_gusts) else 0
            
            day_details = {
                "day_name": day_name,
                "date": date,
                "weather_code": weather_code,
                "temperature_high": temp_high,
                "temperature_low": temp_low,
                "precipitation_sum": precip_sum,
                "precipitation_probability": precip_prob,
                "wind_speed_max": wind_speed,
                "wind_gusts_max": wind_gust,
                "week_context": week_context,
            }
            
            # Check for precipitation (rain, snow, storms are always notable)
            precip_result = self._classify_precipitation(weather_code, precip_sum, precip_prob)
            if precip_result:
                alert_type, severity = precip_result
                alerts.append(WeatherAlert(
                    alert_type=alert_type,
                    date=date,
                    day_name=day_name,
                    severity=severity,
                    message=self._build_alert_message(alert_type, severity, day_details),
                    details=day_details,
                ))
            
            # Check for temperature extremes (seasonal thresholds)
            if temp_high is not None and temp_low is not None:
                temp_result = self._check_temperature_extreme(temp_high, temp_low)
                if temp_result:
                    alert_type, severity = temp_result
                    alerts.append(WeatherAlert(
                        alert_type=alert_type,
                        date=date,
                        day_name=day_name,
                        severity=severity,
                        message=self._build_alert_message(alert_type, severity, day_details),
                        details=day_details,
                    ))
                else:
                    # Also check if today is anomalous vs the week
                    # (catches cases where it's not seasonally extreme but unusual for this week)
                    anomaly = self._is_temperature_anomaly_vs_week(temp_high, temp_low, week_context)
                    if anomaly:
                        alert_type, severity, reason = anomaly
                        # Customize message for week-relative anomaly
                        if alert_type == AlertType.EXTREME_COLD:
                            message = f"Today: Noticeably colder ({temp_low:.0f}°F low) - {reason}. Dress warmly."
                        else:
                            message = f"Today: Noticeably warmer ({temp_high:.0f}°F high) - {reason}."
                        
                        alerts.append(WeatherAlert(
                            alert_type=alert_type,
                            date=date,
                            day_name=day_name,
                            severity=severity,
                            message=message,
                            details=day_details,
                        ))
            
            # Check for high winds
            wind_result = self._check_wind(wind_speed, wind_gust)
            if wind_result:
                alert_type, severity = wind_result
                alerts.append(WeatherAlert(
                    alert_type=alert_type,
                    date=date,
                    day_name=day_name,
                    severity=severity,
                    message=self._build_alert_message(alert_type, severity, day_details),
                    details=day_details,
                ))
        
        return alerts
    
    def prioritize_alerts(self, alerts: List[WeatherAlert]) -> List[WeatherAlert]:
        """
        Sort alerts by severity and consolidate similar alerts.
        
        Returns alerts sorted with most severe/important first.
        """
        severity_order = {"severe": 0, "significant": 1, "moderate": 2}
        
        # Sort by severity, then by date
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (severity_order.get(a.severity, 3), a.date)
        )
        
        return sorted_alerts
    
    def format_alerts_for_briefing(self, alerts: List[WeatherAlert]) -> str:
        """
        Format multiple alerts into a single briefing message.
        
        Returns:
            Combined message suitable for TTS
        """
        if not alerts:
            return ""
        
        if len(alerts) == 1:
            return alerts[0].message
        
        # Group by severity
        severe = [a for a in alerts if a.severity == "severe"]
        significant = [a for a in alerts if a.severity == "significant"]
        moderate = [a for a in alerts if a.severity == "moderate"]
        
        parts = []
        
        if severe:
            parts.append("Weather alert: " + " ".join(a.message for a in severe))
        
        if significant:
            parts.append(" ".join(a.message for a in significant))
        
        if moderate and not severe:  # Skip moderate if we have severe alerts
            parts.append(" ".join(a.message for a in moderate[:2]))  # Limit moderate alerts
        
        return " ".join(parts)

