#!/usr/bin/env python3
"""
Weather Briefing - Main Entry Point

Fetches weather forecast for the device's current location and creates briefing 
announcements when unusual weather conditions are detected.

The device's location is automatically detected via IP geolocation and cached
for future use. Only the device's current location is used for weather briefings.

Usage:
    python main.py [--days DAYS_TO_CHECK] [--zip ZIP_CODE] [--dry-run]

Environment Variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service role key
"""

import argparse
import json
import os
import re
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


def _normalize_delivery_status(briefing: Dict[str, Any], status_field: str, legacy_terminal_status: str) -> str:
    """Treat legacy shared delivery fields as terminal delivery states."""
    current = briefing.get(status_field)
    if isinstance(current, str) and current:
        return current
    if briefing.get("status") == "delivered" or briefing.get("delivered_at"):
        if status_field == "voice_status":
            return legacy_terminal_status
        return "pending"
    return "pending"


def _has_pending_delivery(briefing: Dict[str, Any]) -> bool:
    """Return True while either Discord or voice still needs the briefing."""
    discord_status = _normalize_delivery_status(briefing, "discord_status", "sent")
    voice_status = _normalize_delivery_status(briefing, "voice_status", "read")
    return discord_status == "pending" or voice_status == "pending"


def _slugify_location_name(location_name: str) -> str:
    """Build a stable ASCII slug for weather briefing IDs."""
    slug = re.sub(r"[^a-z0-9]+", "_", location_name.lower()).strip("_")
    return slug or "location"


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
    
    # Create briefing ID with standard format: {source}_{date}_{uuid}
    today = datetime.now().strftime("%Y-%m-%d")
    briefing_id = f"weather_{today}_{uuid.uuid4()}"
    
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
        "opener_text": message,
        "priority": priority,
        "status": "pending",
    }


def create_location_briefing(
    location_name: str,
    alerts: List[WeatherAlert],
    user_id: str,
    analyzer: WeatherAnalyzer
) -> Optional[Dict[str, Any]]:
    """
    Create a briefing announcement for a single location.
    
    Args:
        location_name: Name of the location (e.g., "North Sea, New York")
        alerts: List of WeatherAlert objects for this location
        user_id: Target user ID
        analyzer: WeatherAnalyzer instance for formatting
        
    Returns:
        Briefing dict ready for Supabase, or None if no alerts
    """
    if not alerts:
        return None
    
    # Prioritize and format alerts
    sorted_alerts = analyzer.prioritize_alerts(alerts)
    
    # Build message - use original alert messages (they already have "Today:", "Tomorrow:", etc.)
    # Just strip the day prefix since this is a single-location briefing
    formatted_messages = []
    for alert in sorted_alerts:
        msg = alert.message
        # Remove "Today: " prefix for cleaner reading (it's implied for weather briefings)
        if msg.startswith("Today: "):
            msg = msg[7:]  # Remove "Today: "
        formatted_messages.append(msg)
    
    # Combine messages naturally
    message = " ".join(formatted_messages)
    
    # Determine priority
    has_severe = any(a.severity == "severe" for a in alerts)
    has_significant = any(a.severity == "significant" for a in alerts)
    priority = "high" if has_severe else ("normal" if has_significant else "low")
    
    # Get alert types for metadata
    alert_types = list(set(a.alert_type.value for a in alerts))
    
    # Create briefing ID: weather_{location}_{date}_{uuid}
    today = datetime.now().strftime("%Y-%m-%d")
    location_slug = _slugify_location_name(location_name)
    briefing_id = f"weather_{location_slug}_{today}_{uuid.uuid4()}"
    
    return {
        "id": briefing_id,
        "user_id": user_id,
        "content": {
            "message": message,
            "llm_instructions": "Present this weather update naturally and briefly. Focus on actionable advice.",
            "meta": {
                "source": "weather_briefing",
                "location": location_name,
                "alert_types": alert_types,
                "alert_count": len(alerts),
                "severities": list(set(a.severity for a in alerts)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        "opener_text": message,
        "priority": priority,
        "status": "pending",
    }


def cancel_existing_weather_briefings(client, user_id: str, location_slug: Optional[str] = None) -> int:
    """
    Cancel any existing pending weather briefings for a user.
    
    Called before inserting a new weather briefing to prevent duplicates.
    
    Args:
        client: Supabase client
        user_id: User ID to cancel briefings for
        location_slug: If provided, only cancel briefings for this location
                      (format: weather_{location_slug}_{date}_{uuid})
        
    Returns:
        Number of briefings cancelled
    """
    try:
        # Find pending weather briefings for this user
        # Format: weather_{location}_{date}_{uuid} or legacy weather_{date}_{uuid}
        if location_slug:
            # Only cancel briefings for this specific location
            pattern = f"weather_{location_slug}_%"
        else:
            # Cancel all weather briefings (legacy behavior)
            pattern = "weather_%"
        
        response = (
            client.table("briefing_announcements")
            .select("id, status, delivered_at, discord_status, voice_status")
            .eq("user_id", user_id)
            .eq("status", "pending")
            .like("id", pattern)
            .execute()
        )
        
        if not response.data:
            return 0
        
        # Cancel each pending weather briefing
        cancelled = 0
        for briefing in response.data:
            if not _has_pending_delivery(briefing):
                continue
            try:
                client.table("briefing_announcements").update({
                    "status": "cancelled"
                }).eq("id", briefing["id"]).execute()
                cancelled += 1
                print(f"   🚫 Cancelled old weather briefing: {briefing['id']}")
            except Exception as e:
                print(f"   ⚠️  Failed to cancel {briefing['id']}: {e}")
        
        return cancelled
        
    except Exception as e:
        print(f"   ⚠️  Error checking for existing briefings: {e}")
        return 0


def store_briefing_to_supabase(briefing: Dict[str, Any]) -> bool:
    """
    Store a briefing to the Supabase briefing_announcements table.
    
    Automatically cancels any existing pending weather briefings for the same
    location before inserting the new one.
    
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
        
        # Extract location slug from briefing ID if present
        # Format: weather_{location}_{date}_{uuid}
        user_id = briefing["user_id"]
        briefing_id = briefing["id"]
        location_slug = None
        
        # Parse location from ID: weather_ithaca_2026-01-06_uuid
        parts = briefing_id.split("_")
        if len(parts) >= 4 and parts[0] == "weather":
            # Check if second part looks like a location (not a date)
            if not parts[1].startswith("20"):  # Not a year
                location_slug = parts[1]
        
        # Cancel existing briefings for this location only
        cancelled = cancel_existing_weather_briefings(client, user_id, location_slug)
        if cancelled > 0:
            print(f"   📋 Cancelled {cancelled} existing briefing(s) for {location_slug or 'weather'}")
        
        # Prepare record for Supabase
        record = {
            "id": briefing["id"],
            "user_id": briefing["user_id"],
            "content": json.dumps(briefing["content"]),  # JSONB as string
            "opener_text": briefing.get("opener_text"),
            "priority": briefing["priority"],
            "status": briefing["status"],
        }
        
        # Insert the new briefing
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
    parser.add_argument("--days", type=int, default=1, help="Days ahead to alert on (default: 1 = today only)")
    parser.add_argument("--zip", type=str, help="Optional US ZIP code override (skips auto-detected location)")
    parser.add_argument("--user", type=str, default="Morgan", help="User ID for briefing")
    parser.add_argument("--dry-run", action="store_true", help="Don't store to Supabase")
    parser.add_argument("--refresh-location", action="store_true", help="Force refresh of cached location")
    args = parser.parse_args()
    
    print_banner()
    
    days_to_check = args.days
    user_id = args.user

    zip_override = (args.zip or "").strip()
    location_name = ""
    forecast = None

    if zip_override:
        # Explicit ZIP from caller (for CI/workflow compatibility).
        location_name = f"ZIP {zip_override}"
        print(f"\nConfiguration:")
        print(f"   Location: {location_name} (override)")
        print(f"   ZIP Code: {zip_override}")
        print(f"   Days to check: {days_to_check}")
        print(f"   User: {user_id}")
        print(f"   Dry run: {args.dry_run}")
        print()

        analyzer = WeatherAnalyzer(zip_code=zip_override)
        print(f"🌡️  Current season: {analyzer._current_season}")
        print(f"   Temperature thresholds: {analyzer._get_temp_thresholds()}")
        print()

        print(f"\n{'='*60}")
        print(f"📍 {location_name}")
        print("="*60)
        forecast = fetch_weather_data(zip_override, days=7)
    else:
        # Auto-detect device location (uses cache unless --refresh-location)
        print("\n📍 Detecting device location...")
        location = auto_detect_location(use_cache=not args.refresh_location)
        
        if not location:
            print("❌ Failed to detect device location. Cannot generate weather briefing.")
            print("   Try running with --refresh-location to retry IP geolocation.")
            return 1
        
        location_name = location.display_name()
        
        print(f"\nConfiguration:")
        print(f"   Location: {location_name}")
        print(f"   Coordinates: ({location.lat:.4f}, {location.lon:.4f})")
        if location.zip_code:
            print(f"   ZIP Code: {location.zip_code}")
        print(f"   Days to check: {days_to_check}")
        print(f"   User: {user_id}")
        print(f"   Dry run: {args.dry_run}")
        print()
        
        analyzer = WeatherAnalyzer(zip_code=location.zip_code or "00000")
        print(f"🌡️  Current season: {analyzer._current_season}")
        print(f"   Temperature thresholds: {analyzer._get_temp_thresholds()}")
        print()
        
        # Fetch weather using coordinates (more accurate than ZIP code)
        print(f"\n{'='*60}")
        print(f"📍 {location_name}")
        print("="*60)
        forecast = fetch_weather_by_coords(location.lat, location.lon, days=7)

    if not forecast:
        print(f"   ❌ Failed to fetch weather data for {location_name or 'requested location'}")
        return 1
    
    # Display forecast
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    print(f"\n📅 7-day forecast:")
    for i, date in enumerate(dates):
        temp_high = daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else "?"
        temp_low = daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else "?"
        precip = daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else 0
        precip_prob = daily.get("precipitation_probability_max", [])[i] if i < len(daily.get("precipitation_probability_max", [])) else 0
        code = daily.get("weathercode", [])[i] if i < len(daily.get("weathercode", [])) else 0
        day_name = analyzer._get_day_name(date)
        marker = "→ " if i < days_to_check else "  "
        print(f"   {marker}{day_name} ({date}): {temp_low}°F - {temp_high}°F, precip: {precip}\" ({precip_prob}%), code: {code}")
    
    # Analyze for unusual conditions
    print(f"\n🔍 Analyzing weather for unusual conditions...")
    alerts = analyzer.analyze_forecast(forecast, days_to_check=days_to_check)
    
    # Summarize alerts
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print("="*60)
    
    if not alerts:
        print("   ✅ No unusual weather detected - no briefing needed")
        print("\n" + "=" * 60)
        print("✅ COMPLETE - No action required")
        print("=" * 60)
        return 0
    
    print(f"   ⚠️  Found {len(alerts)} weather alert(s):")
    for alert in alerts:
        severity_emoji = {"severe": "🔴", "significant": "🟠", "moderate": "🟡"}
        emoji = severity_emoji.get(alert.severity, "⚪")
        print(f"      {emoji} [{alert.severity.upper()}] {alert.message}")
    print()
    
    # Save alerts locally for debugging
    ephemeral_dir = Path(__file__).parent / "ephemeral_data"
    save_alerts_locally(alerts, ephemeral_dir)
    
    # Create briefing
    print(f"\n📝 Creating weather briefing...")
    briefing = create_location_briefing(location_name, alerts, user_id, analyzer)
    
    if not briefing:
        print(f"   ⚠️  Failed to create briefing")
        return 1
    
    print(f"   Briefing ID: {briefing['id']}")
    print(f"   Priority: {briefing['priority']}")
    print(f"   Message: {briefing['content']['message']}")
    
    # Store to Supabase
    if args.dry_run:
        print(f"   🏃 Dry run - would store: {briefing['id']}")
    else:
        print(f"   💾 Storing to Supabase...")
        if store_briefing_to_supabase(briefing):
            print(f"   ✅ Briefing stored successfully")
        else:
            print(f"   ⚠️  Failed to store briefing (check logs)")
    
    print("\n" + "=" * 60)
    print("✅ WEATHER BRIEFING COMPLETE")
    print(f"   Location: {location_name}")
    print(f"   Alerts found: {len(alerts)}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
