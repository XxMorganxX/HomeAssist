#!/usr/bin/env python3
"""
Calendar Reminder Analyzer - Main Entry Point

Scheduled process that analyzes upcoming calendar events and determines
optimal reminder timing using AI-powered analysis.

This script:
1. Fetches upcoming calendar events for configured users
2. Analyzes each event to determine appropriate reminder timing
3. Stores reminder suggestions for the assistant to use
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Import after path setup
from analyzer import ReminderAnalyzer
from briefing_creator import BriefingCreator

# Configuration
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "reminder_state.json"
EPHEMERAL_DIR = SCRIPT_DIR / "ephemeral_data"
OUTPUT_FILE = EPHEMERAL_DIR / "reminder_suggestions.json"

# How far ahead to look for events (days)
LOOKAHEAD_DAYS = int(os.getenv("REMINDER_LOOKAHEAD_DAYS", "7"))


def print_banner():
    """Print script banner."""
    print("=" * 60)
    print("📅 CALENDAR REMINDER ANALYZER")
    print("=" * 60)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Lookahead: {LOOKAHEAD_DAYS} days")
    print("=" * 60)


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def load_state() -> dict:
    """Load the analyzer state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading state file: {e}")
    
    return {
        "last_run": None,
        "events_analyzed": 0,
        "users_processed": [],
    }


def save_state(state: dict):
    """Save analyzer state to disk."""
    try:
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"💾 State saved to {STATE_FILE.name}")
    except IOError as e:
        print(f"❌ Failed to save state: {e}")


def get_calendar_users() -> list:
    """Get list of calendar users to process from config."""
    try:
        from mcp_server.config import CALENDAR_USERS
        return list(CALENDAR_USERS.keys())
    except ImportError:
        print("⚠️  Could not import CALENDAR_USERS from config")
        # Fallback to env var or default
        users_env = os.getenv("REMINDER_CALENDAR_USERS", "morgan_personal")
        return [u.strip() for u in users_env.split(",")]


def fetch_events_for_user(user: str, days_ahead: int = 7) -> list:
    """
    Fetch upcoming calendar events for a specific user.
    
    Args:
        user: Calendar user identifier
        days_ahead: Number of days to look ahead
        
    Returns:
        List of formatted calendar events
    """
    try:
        from mcp_server.clients.calendar_client import CalendarComponent
        
        print(f"📆 Fetching events for {user}...")
        
        calendar = CalendarComponent(user=user)
        
        # Calculate time range
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(days=days_ahead)
        
        # Get events
        events = calendar.get_events(
            num_events=50,
            time_min=now.isoformat(),
            time_max=end_time.isoformat(),
        )
        
        # Format events
        formatted_events = [calendar.format_event(event) for event in events]
        
        print(f"   ✅ Found {len(formatted_events)} events")
        return formatted_events
        
    except ValueError as e:
        print(f"   ⚠️  Invalid user '{user}': {e}")
        return []
    except Exception as e:
        print(f"   ❌ Error fetching events for {user}: {e}")
        return []


def save_suggestions(suggestions: dict):
    """Save reminder suggestions to ephemeral output file."""
    try:
        EPHEMERAL_DIR.mkdir(exist_ok=True)
        
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookahead_days": LOOKAHEAD_DAYS,
            "suggestions": suggestions,
        }
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved suggestions to {OUTPUT_FILE.name}")
        return True
        
    except IOError as e:
        print(f"❌ Failed to save suggestions: {e}")
        return False


def store_to_notification_queue(suggestions: dict):
    """
    Store high-priority reminder suggestions to the notification queue.
    
    Only stores events that need reminders within the next 24 hours.
    """
    try:
        from state_management.statemanager import StateManager
        
        state_manager = StateManager()
        notifications_added = 0
        
        for user, analyses in suggestions.items():
            # Map calendar user to assistant user
            assistant_user = "Morgan" if "morgan" in user.lower() else "Spencer"
            
            for analysis in analyses:
                # Check if event is within next 24 hours and high priority
                priority = analysis.get("priority", "medium")
                if priority != "high":
                    continue
                
                event_title = analysis.get("event_title", "Event")
                event_date = analysis.get("event_date", "")
                event_time = analysis.get("event_time", "")
                reminders = analysis.get("reminders_minutes_before", [])
                
                # Create notification content
                reminder_desc = ", ".join(
                    f"{m // 60}h" if m >= 60 else f"{m}m" for m in reminders[:2]
                )
                
                notification = {
                    "intended_recipient": assistant_user,
                    "notification_content": (
                        f"📅 Upcoming: {event_title}\n"
                        f"When: {event_date} at {event_time}\n"
                        f"Suggested reminders: {reminder_desc} before"
                    ),
                    "relevant_when": "next_24_hours",
                    "source": "calendar_reminder_analyzer",
                }
                
                # Note: This would need the notification queue to support calendar reminders
                # For now, we just log the intention
                print(f"   📣 Would notify {assistant_user}: {event_title}")
                notifications_added += 1
        
        if notifications_added:
            print(f"✅ Prepared {notifications_added} high-priority notifications")
        
    except ImportError:
        print("⚠️  StateManager not available, skipping notification queue")
    except Exception as e:
        print(f"⚠️  Error adding to notification queue: {e}")


def main():
    """Main entry point for the reminder analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze calendar events and suggest optimal reminder timing"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=LOOKAHEAD_DAYS,
        help=f"Days to look ahead (default: {LOOKAHEAD_DAYS})",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Specific user to analyze (default: all configured users)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but don't save results",
    )
    parser.add_argument(
        "--skip-supabase",
        action="store_true",
        help="Skip pushing briefings to Supabase",
    )
    parser.add_argument(
        "--skip-openers",
        action="store_true",
        help="Skip AI generation of opener_text for briefings",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Load state
    state = load_state()
    if state.get("last_run"):
        print(f"Last run: {state['last_run']}")
    
    # Initialize analyzer
    print_separator("INITIALIZING")
    analyzer = ReminderAnalyzer()
    
    # Determine users to process
    if args.user:
        users = [args.user]
    else:
        users = get_calendar_users()
    
    print(f"Users to process: {', '.join(users)}")
    
    # Process each user
    all_suggestions = {}
    total_events = 0
    
    for user in users:
        print_separator(f"USER: {user}")
        
        # Fetch events
        events = fetch_events_for_user(user, days_ahead=args.days)
        
        if not events:
            print(f"   ℹ️  No upcoming events for {user}")
            continue
        
        # Analyze events
        analyses = analyzer.analyze_events(events, user)
        
        if analyses:
            all_suggestions[user] = analyses
            total_events += len(analyses)
            
            # Print summary
            print(analyzer.format_reminder_summary(analyses))
    
    # Save results
    print_separator("SAVING RESULTS")
    
    briefings_created = 0
    
    if args.dry_run:
        print("🔍 Dry run - not saving results")
    else:
        if all_suggestions:
            save_suggestions(all_suggestions)
            
            # Create and store briefing announcements
            if not args.skip_supabase:
                print_separator("CREATING BRIEFINGS")
                briefing_creator = BriefingCreator(generate_openers=not args.skip_openers)
                
                # Create briefings from suggestions
                briefings = briefing_creator.create_briefings_from_suggestions(all_suggestions)
                briefings_created = len(briefings)
                
                if briefings:
                    print(briefing_creator.format_briefing_summary(briefings))
                    
                    # Store to Supabase briefing_announcements table
                    if briefing_creator.is_available():
                        briefing_creator.store_briefings(briefings)
                        
                        # Cleanup old delivered reminders
                        briefing_creator.cleanup_delivered_reminders()
                    else:
                        print("⚠️  Supabase not available, briefings not stored")
                else:
                    print("📭 No future briefings to create (all reminders in the past)")
            else:
                print("⏭️  Skipping Supabase briefing storage")
    
    # Update state
    state["events_analyzed"] = total_events
    state["users_processed"] = users
    state["briefings_created"] = briefings_created
    
    if not args.dry_run:
        save_state(state)
    
    # Final summary
    print_separator("COMPLETE")
    print(f"✅ Analyzed {total_events} events for {len(users)} user(s)")
    if briefings_created:
        print(f"📣 Created {briefings_created} reminder briefings")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

