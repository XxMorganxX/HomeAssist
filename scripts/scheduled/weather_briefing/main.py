#!/usr/bin/env python3
"""
Weather Briefing - Main Entry Point

Fetches weather forecasts and creates briefing announcements only when
unusual weather conditions are detected in the next 2 days.

Usage:
    python main.py [--zip ZIP_CODE] [--days DAYS_TO_CHECK] [--dry-run]

Environment Variables:
    WEATHER_ZIP_CODE: Default ZIP code (default: 10001)
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service role key
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .analyzer import WeatherAnalyzer, WeatherAlert, auto_detect_location, set_home_location, GeoLocation
except ImportError:
    from analyzer import WeatherAnalyzer, WeatherAlert, auto_detect_location, set_home_location, GeoLocation

# Load environment variables
load_dotenv()


def print_banner():
    """Print startup banner."""
    print("=" * 60)
    print("🌤️  WEATHER BRIEFING")
    print("=" * 60)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


def get_weather_client():
    """Import and return WeatherClient from MCP server."""
    try:
        from mcp_server.clients.weather_client import WeatherClient
        return WeatherClient()
    except ImportError:
        print("❌ Failed to import WeatherClient from mcp_server")
        print("   Make sure you're running from the project root")
        return None


def fetch_weather_by_coords(lat: float, lon: float, days: int = 7) -> Optional[Dict[str, Any]]:
    """
    Fetch weather forecast data using coordinates directly.
    
    This is more accurate than ZIP code because we use the exact coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Number of days to fetch (max 7)
        
    Returns:
        Forecast data dict or None on error
    """
    client = get_weather_client()
    if not client:
        return None
    
    try:
        print(f"🌍 Fetching weather for ({lat:.4f}, {lon:.4f}) ({days} days)...")
        daily = client.fetch_open_meteo_daily(lat, lon, days=days, units="imperial")
        forecast = {
            "latitude": lat,
            "longitude": lon,
            "days": days,
            "daily": daily,
            "units": "imperial",
        }
        print(f"   ✅ Got forecast data")
        return forecast
    except Exception as e:
        print(f"   ❌ Error fetching weather: {e}")
        return None


def fetch_weather_data(zip_code: str, days: int = 7) -> Optional[Dict[str, Any]]:
    """
    Fetch weather forecast data for the specified ZIP code.
    
    Args:
        zip_code: US ZIP code
        days: Number of days to fetch (max 7)
        
    Returns:
        Forecast data dict or None on error
    """
    client = get_weather_client()
    if not client:
        return None
    
    try:
        print(f"📍 Fetching weather for ZIP {zip_code} ({days} days)...")
        forecast = client.get_daily_forecast(zip_code, days=days, units="imperial")
        print(f"   ✅ Got forecast data")
        return forecast
    except Exception as e:
        print(f"   ❌ Error fetching weather: {e}")
        return None


def create_briefing_from_alerts(
    alerts: List[WeatherAlert],
    user_id: str,
    analyzer: WeatherAnalyzer
) -> Optional[Dict[str, Any]]:
    """
    Create a briefing announcement from weather alerts.
    
    Args:
        alerts: List of WeatherAlert objects
        user_id: Target user ID
        analyzer: WeatherAnalyzer instance for formatting
        
    Returns:
        Briefing dict ready for Supabase, or None if no alerts
    """
    if not alerts:
        return None
    
    # Prioritize and format alerts
    sorted_alerts = analyzer.prioritize_alerts(alerts)
    message = analyzer.format_alerts_for_briefing(sorted_alerts)
    
    # Determine overall priority
    has_severe = any(a.severity == "severe" for a in alerts)
    has_significant = any(a.severity == "significant" for a in alerts)
    priority = "high" if has_severe else ("normal" if has_significant else "low")
    
    # Get alert types for metadata
    alert_types = list(set(a.alert_type.value for a in alerts))
    
    # Create briefing ID based on date and type
    today = datetime.now().strftime("%Y%m%d")
    briefing_id = f"weather_{today}_{uuid.uuid4().hex[:8]}"
    
    return {
        "id": briefing_id,
        "user_id": user_id,
        "content": {
            "message": message,
            "llm_instructions": "Present this weather update naturally and briefly. Focus on actionable advice.",
            "meta": {
                "source": "weather_briefing",
                "alert_types": alert_types,
                "alert_count": len(alerts),
                "severities": list(set(a.severity for a in alerts)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        "priority": priority,
        "status": "pending",
    }


def store_briefing_to_supabase(briefing: Dict[str, Any]) -> bool:
    """
    Store a briefing to the Supabase briefing_announcements table.
    
    Args:
        briefing: Briefing dict to store
        
    Returns:
        True if successful, False otherwise
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("⚠️  SUPABASE_URL or SUPABASE_KEY not set, skipping storage")
        return False
    
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
        
        # Prepare record for Supabase
        record = {
            "id": briefing["id"],
            "user_id": briefing["user_id"],
            "content": json.dumps(briefing["content"]),  # JSONB as string
            "priority": briefing["priority"],
            "status": briefing["status"],
        }
        
        # Upsert to avoid duplicates
        client.table("briefing_announcements").upsert(record).execute()
        print(f"   ✅ Stored briefing to Supabase: {briefing['id']}")
        return True
        
    except ImportError:
        print("⚠️  supabase package not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error storing to Supabase: {e}")
        return False


def save_alerts_locally(alerts: List[WeatherAlert], output_dir: Path) -> None:
    """Save alerts to local ephemeral data for debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "weather_alerts.json"
    
    alerts_data = [
        {
            "alert_type": a.alert_type.value,
            "date": a.date,
            "day_name": a.day_name,
            "severity": a.severity,
            "message": a.message,
            "details": a.details,
        }
        for a in alerts
    ]
    
    with open(output_file, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(alerts),
            "alerts": alerts_data,
        }, f, indent=2)
    
    print(f"   💾 Saved {len(alerts)} alerts to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Weather Briefing Generator")
    parser.add_argument("--zip", type=str, help="ZIP code (fallback if coords not available)")
    parser.add_argument("--lat", type=float, help="Latitude (more accurate than ZIP)")
    parser.add_argument("--lon", type=float, help="Longitude (more accurate than ZIP)")
    parser.add_argument("--days", type=int, default=1, help="Days ahead to alert on (default: 1 = today only)")
    parser.add_argument("--user", type=str, default="Morgan", help="User ID for briefing")
    parser.add_argument("--dry-run", action="store_true", help="Don't store to Supabase")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached location")
    parser.add_argument("--set-location", action="store_true", help="Set home location from --lat/--lon and exit")
    args = parser.parse_args()
    
    print_banner()
    
    # Handle --set-location command
    if args.set_location:
        if args.lat is None or args.lon is None:
            print("❌ --set-location requires both --lat and --lon")
            print("   Example: python main.py --set-location --lat 40.8351 --lon -73.6265")
            return 1
        set_home_location(args.lat, args.lon)
        return 0
    
    # Location resolution priority:
    # 1. Command line --lat/--lon (most accurate)
    # 2. Cached location (from previous run or manual set)
    # 3. Auto-detect from IP geolocation
    # 4. WEATHER_ZIP_CODE environment variable
    # 5. Command line --zip
    # 6. Default fallback (NYC)
    
    location: Optional[GeoLocation] = None
    zip_code: Optional[str] = None
    
    # Check for explicit coordinates
    if args.lat is not None and args.lon is not None:
        location = GeoLocation(lat=args.lat, lon=args.lon, source="command_line")
        print(f"📍 Using provided coordinates: ({args.lat}, {args.lon})")
    else:
        # Try auto-detection (includes cache check)
        location = auto_detect_location(use_cache=not args.no_cache)
    
    # Fallback to ZIP code if no coordinates
    if not location:
        zip_code = args.zip or os.getenv("WEATHER_ZIP_CODE")
        if zip_code:
            print(f"📍 Using ZIP code: {zip_code}")
        else:
            zip_code = "10001"  # NYC default
            print(f"⚠️  No location found, using default ZIP: {zip_code}")
    
    days_to_check = args.days
    user_id = args.user
    
    print(f"\nConfiguration:")
    if location:
        print(f"   Location: {location.display_name()}")
        print(f"   Coordinates: ({location.lat:.4f}, {location.lon:.4f})")
    else:
        print(f"   ZIP Code: {zip_code}")
    print(f"   Days to check: {days_to_check}")
    print(f"   User: {user_id}")
    print(f"   Dry run: {args.dry_run}")
    print()
    
    # Initialize analyzer
    analyzer = WeatherAnalyzer(zip_code=zip_code or "00000")
    print(f"🌡️  Current season: {analyzer._current_season}")
    print(f"   Temperature thresholds: {analyzer._get_temp_thresholds()}")
    print()
    
    # Fetch weather data - prefer coordinates over ZIP
    if location:
        forecast = fetch_weather_by_coords(location.lat, location.lon, days=7)
    else:
        forecast = fetch_weather_data(zip_code, days=7)
    
    if not forecast:
        print("❌ Failed to fetch weather data")
        return 1
    
    # Display full week forecast for context
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    print(f"\n📅 7-day forecast (for context):")
    for i, date in enumerate(dates):
        temp_high = daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else "?"
        temp_low = daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else "?"
        precip = daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else 0
        precip_prob = daily.get("precipitation_probability_max", [])[i] if i < len(daily.get("precipitation_probability_max", [])) else 0
        code = daily.get("weathercode", [])[i] if i < len(daily.get("weathercode", [])) else 0
        day_name = analyzer._get_day_name(date)
        marker = "→ " if i < days_to_check else "  "  # Mark days we'll alert on
        print(f"   {marker}{day_name} ({date}): {temp_low}°F - {temp_high}°F, precip: {precip}\" ({precip_prob}%), code: {code}")
    print(f"\n🎯 Alerting on: next {days_to_check} day(s) only (today)")
    
    # Analyze for unusual conditions
    print("🔍 Analyzing for unusual weather...")
    alerts = analyzer.analyze_forecast(forecast, days_to_check=days_to_check)
    
    if not alerts:
        print("   ✅ No unusual weather detected - no briefing needed")
        print("\n" + "=" * 60)
        print("✅ COMPLETE - No action required")
        print("=" * 60)
        return 0
    
    # Display alerts
    print(f"   ⚠️  Found {len(alerts)} weather alert(s):")
    for alert in alerts:
        severity_emoji = {"severe": "🔴", "significant": "🟠", "moderate": "🟡"}
        emoji = severity_emoji.get(alert.severity, "⚪")
        print(f"      {emoji} [{alert.severity.upper()}] {alert.message}")
    print()
    
    # Save locally for debugging
    ephemeral_dir = Path(__file__).parent / "ephemeral_data"
    save_alerts_locally(alerts, ephemeral_dir)
    
    # Create briefing
    print("📝 Creating briefing announcement...")
    briefing = create_briefing_from_alerts(alerts, user_id, analyzer)
    
    if not briefing:
        print("   ⚠️  Failed to create briefing")
        return 1
    
    print(f"   Briefing ID: {briefing['id']}")
    print(f"   Priority: {briefing['priority']}")
    print(f"   Message: {briefing['content']['message']}")
    print()
    
    # Store to Supabase
    if args.dry_run:
        print("🏃 Dry run - skipping Supabase storage")
        print(f"   Would store: {json.dumps(briefing, indent=2)}")
    else:
        print("💾 Storing to Supabase...")
        if store_briefing_to_supabase(briefing):
            print("   ✅ Briefing stored successfully")
        else:
            print("   ⚠️  Failed to store briefing (check logs)")
    
    print("\n" + "=" * 60)
    print("✅ WEATHER BRIEFING COMPLETE")
    print(f"   Alerts generated: {len(alerts)}")
    print(f"   Briefing created: {briefing['id']}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

