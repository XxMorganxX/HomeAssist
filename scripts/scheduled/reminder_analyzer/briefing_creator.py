"""
Briefing Creator for Calendar Reminders

Converts reminder suggestions into scheduled briefing announcements
and stores them in Supabase for the assistant to deliver.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Default timezone for parsing local times
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")

# Default start hour for all-day events (9 AM)
ALL_DAY_START_HOUR = int(os.getenv("ALL_DAY_EVENT_START_HOUR", "9"))


class ReminderDatetimeCalculator:
    """
    Calculates actual reminder datetimes from event data.
    
    Handles:
    - Events with specific times (e.g., "13:00:00")
    - All-day events (e.g., "All Day")
    - Timezone conversion
    """
    
    def __init__(self, timezone_str: str = DEFAULT_TIMEZONE):
        """Initialize with a specific timezone."""
        self.tz = ZoneInfo(timezone_str)
    
    def parse_event_datetime(
        self,
        event_date: str,
        event_time: str,
    ) -> datetime:
        """
        Parse event date and time into a timezone-aware datetime.
        
        Args:
            event_date: Date string in YYYY-MM-DD format
            event_time: Time string in HH:MM:SS format, or "All Day"
            
        Returns:
            Timezone-aware datetime for the event start
        """
        # Parse the date
        try:
            date_obj = datetime.strptime(event_date, "%Y-%m-%d").date()
        except ValueError:
            # Fallback: try other formats
            date_obj = datetime.now().date()
        
        # Parse the time
        if event_time.lower() in ("all day", "all-day", "allday", ""):
            # All-day events: use configured start hour
            time_obj = datetime.min.time().replace(hour=ALL_DAY_START_HOUR)
        else:
            try:
                # Try HH:MM:SS format
                time_obj = datetime.strptime(event_time, "%H:%M:%S").time()
            except ValueError:
                try:
                    # Try HH:MM format
                    time_obj = datetime.strptime(event_time, "%H:%M").time()
                except ValueError:
                    # Fallback to 9 AM
                    time_obj = datetime.min.time().replace(hour=9)
        
        # Combine date and time with timezone
        naive_dt = datetime.combine(date_obj, time_obj)
        return naive_dt.replace(tzinfo=self.tz)
    
    def calculate_reminder_datetime(
        self,
        event_datetime: datetime,
        minutes_before: int,
    ) -> datetime:
        """
        Calculate the datetime when a reminder should fire.
        
        Args:
            event_datetime: When the event starts
            minutes_before: How many minutes before to remind
            
        Returns:
            Datetime when the reminder should be delivered
        """
        return event_datetime - timedelta(minutes=minutes_before)
    
    def calculate_all_reminder_times(
        self,
        event_date: str,
        event_time: str,
        reminders_minutes_before: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Calculate all reminder times for an event.
        
        Args:
            event_date: Event date string
            event_time: Event time string
            reminders_minutes_before: List of minutes before event
            
        Returns:
            List of dicts with reminder_at datetime and minutes_before
        """
        event_dt = self.parse_event_datetime(event_date, event_time)
        
        reminders = []
        for minutes in reminders_minutes_before:
            reminder_dt = self.calculate_reminder_datetime(event_dt, minutes)
            reminders.append({
                "reminder_at": reminder_dt,
                "reminder_at_iso": reminder_dt.isoformat(),
                "minutes_before": minutes,
                "event_datetime": event_dt,
                "event_datetime_iso": event_dt.isoformat(),
            })
        
        return reminders
    
    def format_time_until(self, minutes: int) -> str:
        """Format minutes into human-readable time description."""
        if minutes >= 1440:
            days = minutes // 1440
            return f"{days} day{'s' if days > 1 else ''}"
        elif minutes >= 60:
            hours = minutes // 60
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes > 1 else ''}"


class BriefingCreator:
    """
    Creates briefing announcements from calendar reminder suggestions
    and stores them in Supabase.
    """
    
    # Supabase table for briefing announcements
    TABLE_NAME = "briefing_announcements"
    
    # System prompt for generating opener text
    OPENER_SYSTEM_PROMPT = """Generate a friendly 1-2 sentence spoken reminder for a voice assistant. Be warm, natural, and helpful. Don't be robotic.

Examples:
- Input: "Doctor Appointment in 1 hour" → Output: "Just a heads up - your doctor appointment is in about an hour. Need anything before you head out?"
- Input: "Team Meeting in 15 minutes" → Output: "Quick reminder, your team meeting starts in 15 minutes!"
- Input: "Vegas Trip tomorrow" → Output: "Hey! Just wanted to remind you about your Vegas trip tomorrow. Have you started packing yet?"
"""
    
    def __init__(self, timezone_str: str = DEFAULT_TIMEZONE, generate_openers: bool = True):
        """Initialize the briefing creator.
        
        Args:
            timezone_str: Timezone for datetime calculations
            generate_openers: Whether to generate AI opener_text for briefings
        """
        self.calculator = ReminderDatetimeCalculator(timezone_str)
        self._client = None
        self._initialized = False
        self._generate_openers = generate_openers
        self._ai_model = None
        self._init_supabase()
        if generate_openers:
            self._init_ai()
    
    def _init_supabase(self):
        """Initialize Supabase client."""
        try:
            from supabase import create_client
            
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                print("⚠️  BriefingCreator: SUPABASE_URL or SUPABASE_KEY not set")
                return
            
            self._client = create_client(url, key)
            self._initialized = True
            print("✅ BriefingCreator Supabase client initialized")
            
        except ImportError:
            print("⚠️  BriefingCreator: supabase package not installed")
        except Exception as e:
            print(f"❌ BriefingCreator: Failed to initialize Supabase - {e}")
    
    def _init_ai(self):
        """Initialize AI model for opener generation."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("⚠️  BriefingCreator: GEMINI_API_KEY not set, openers will not be generated")
                return
            
            genai.configure(api_key=api_key)
            # Use gemini-2.0-flash-lite for opener generation (faster, doesn't use thinking tokens)
            opener_model = os.getenv("OPENER_GEMINI_MODEL", "gemini-2.0-flash-lite")
            self._ai_model = genai.GenerativeModel(
                opener_model,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 200,
                },
            )
            print(f"✅ BriefingCreator AI initialized ({opener_model})")
            
        except ImportError:
            print("⚠️  BriefingCreator: google-generativeai not installed, openers will not be generated")
        except Exception as e:
            print(f"⚠️  BriefingCreator: Failed to initialize AI - {e}")
    
    def generate_opener(self, message: str, max_retries: int = 2) -> Optional[str]:
        """
        Generate a natural conversation opener for a reminder message.
        
        Args:
            message: The reminder message to convert to an opener
            max_retries: Number of retries if response seems incomplete
            
        Returns:
            Generated opener text, or None if generation fails
        """
        if not self._ai_model:
            return None
        
        # Build full prompt with instructions
        prompt = f"""{self.OPENER_SYSTEM_PROMPT}

Generate a friendly spoken reminder for: "{message}"

Output ONLY the reminder text (1-2 complete sentences that end properly), nothing else:"""
        
        for attempt in range(max_retries + 1):
            try:
                response = self._ai_model.generate_content(prompt)
                
                # Extract text from response - try .text first (most reliable)
                opener = ""
                try:
                    opener = response.text
                except Exception:
                    # Fallback: iterate through candidates
                    try:
                        candidates = getattr(response, "candidates", []) or []
                        for c in candidates:
                            cont = getattr(c, "content", None)
                            parts = getattr(cont, "parts", []) if cont else []
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    opener += t
                    except Exception:
                        opener = ""
                
                opener = (opener or "").strip()
                
                # Validate opener is complete (ends with sentence-ending punctuation)
                if opener and self._is_opener_complete(opener):
                    return opener
                elif opener and attempt < max_retries:
                    # Retry if truncated
                    continue
                elif opener:
                    # Last attempt - try to fix truncated opener
                    return self._fix_truncated_opener(opener, message)
                    
            except Exception as e:
                if attempt == max_retries:
                    print(f"⚠️  Failed to generate opener after {max_retries + 1} attempts: {e}")
                continue
        
        return None
    
    def _is_opener_complete(self, opener: str) -> bool:
        """Check if an opener appears to be a complete sentence."""
        if not opener:
            return False
        
        # Must be at least 30 characters for a meaningful opener
        if len(opener) < 30:
            return False
        
        # Should end with sentence-ending punctuation
        ending_chars = ".!?"
        stripped = opener.rstrip()
        if stripped and stripped[-1] in ending_chars:
            return True
        
        # Check if it ends with a closing quote followed by punctuation
        if len(stripped) >= 2 and stripped[-1] in "\"'" and stripped[-2] in ending_chars:
            return True
        
        return False
    
    def _fix_truncated_opener(self, opener: str, original_message: str) -> str:
        """Attempt to fix a truncated opener by completing it sensibly."""
        # If it ends mid-sentence, try to complete it
        opener = opener.rstrip()
        
        # Remove trailing incomplete fragments
        # Common truncation patterns: ends with articles, prepositions, etc.
        incomplete_endings = [" a", " an", " the", " at", " in", " on", " to", " for", " your", " is", " are"]
        for ending in incomplete_endings:
            if opener.lower().endswith(ending):
                opener = opener[:-len(ending)].rstrip()
                break
        
        # If still doesn't end properly, add a sensible ending
        if opener and opener[-1] not in ".!?":
            # Just add a period if we had to truncate
            if opener[-1] in ",:;":
                opener = opener[:-1]
            opener = opener.rstrip() + "."
        
        return opener
    
    def is_available(self) -> bool:
        """Check if Supabase storage is available."""
        return self._initialized and self._client is not None
    
    def create_briefings_from_suggestions(
        self,
        suggestions: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Convert reminder suggestions into briefing announcements.
        
        Args:
            suggestions: Dict mapping calendar_user -> list of analyzed events
            
        Returns:
            List of briefing dicts ready for storage
        """
        briefings = []
        now = datetime.now(timezone.utc)
        
        for calendar_user, events in suggestions.items():
            # Map calendar user to assistant user
            assistant_user = self._map_to_assistant_user(calendar_user)
            
            for event in events:
                event_briefings = self._create_event_briefings(
                    event=event,
                    calendar_user=calendar_user,
                    assistant_user=assistant_user,
                    now=now,
                )
                briefings.extend(event_briefings)
        
        # Sort by reminder time
        briefings.sort(key=lambda b: b.get("deliver_at", ""))
        
        return briefings
    
    def _map_to_assistant_user(self, calendar_user: str) -> str:
        """Map calendar user identifier to assistant user name."""
        calendar_user_lower = calendar_user.lower()
        
        if "morgan" in calendar_user_lower:
            return "Morgan"
        elif "spencer" in calendar_user_lower:
            return "Spencer"
        else:
            # Default to primary user from env or Morgan
            return os.getenv("DEFAULT_ASSISTANT_USER", "Morgan")
    
    def _create_event_briefings(
        self,
        event: Dict[str, Any],
        calendar_user: str,
        assistant_user: str,
        now: datetime,
    ) -> List[Dict[str, Any]]:
        """Create briefing announcements for a single event."""
        event_id = event.get("event_id", "")
        event_title = event.get("event_title", "Event")
        event_date = event.get("event_date", "")
        event_time = event.get("event_time", "")
        priority = event.get("priority", "medium")
        reminders_minutes = event.get("reminders_minutes_before", [])
        
        # Calculate reminder datetimes
        reminder_times = self.calculator.calculate_all_reminder_times(
            event_date, event_time, reminders_minutes
        )
        
        briefings = []
        for reminder in reminder_times:
            reminder_at = reminder["reminder_at"]
            
            # Skip reminders in the past
            if reminder_at.astimezone(timezone.utc) < now:
                continue
            
            # Generate unique ID for this specific reminder
            reminder_id = f"cal_reminder_{event_id}_{reminder['minutes_before']}m"
            
            # Format the briefing message
            time_until = self.calculator.format_time_until(reminder["minutes_before"])
            
            if event_time.lower() in ("all day", "all-day", "allday", ""):
                time_desc = f"on {event_date}"
            else:
                # Format time more naturally (e.g., "1:00 PM" instead of "13:00:00")
                try:
                    t = datetime.strptime(event_time, "%H:%M:%S")
                    formatted_time = t.strftime("%I:%M %p").lstrip("0")
                except ValueError:
                    formatted_time = event_time
                time_desc = f"at {formatted_time} on {event_date}"
            
            message = f"You have '{event_title}' coming up {time_desc}. This is your {time_until} reminder."
            
            # active_from is the full datetime when the reminder should become active
            active_from_datetime = reminder["reminder_at_iso"]
            
            # Build content JSONB matching briefing_announcements schema
            content = {
                "message": message,
                "llm_instructions": f"Remind the user about their upcoming event: {event_title}. Be helpful and mention any preparation they might need.",
                "active_from": active_from_datetime,
                "meta": {
                    "source": "calendar_reminder_analyzer",
                    "event_id": event_id,
                    "event_title": event_title,
                    "event_date": event_date,
                    "event_time": event_time,
                    "event_datetime_iso": reminder["event_datetime_iso"],
                    "reminder_at_iso": reminder["reminder_at_iso"],
                    "minutes_before": reminder["minutes_before"],
                    "calendar_user": calendar_user,
                },
            }
            
            # Map priority to briefing_announcements format
            briefing_priority = "high" if priority == "high" else "normal"
            
            # Generate opener_text using AI
            opener_text = None
            if self._generate_openers and self._ai_model:
                opener_text = self.generate_opener(message)
            
            briefing = {
                "id": reminder_id,
                "user_id": assistant_user,
                "content": content,
                "priority": briefing_priority,
                "status": "pending",
            }
            
            # Add opener_text if generated
            if opener_text:
                briefing["opener_text"] = opener_text
            
            briefings.append(briefing)
        
        return briefings
    
    def store_briefings(
        self,
        briefings: List[Dict[str, Any]],
    ) -> bool:
        """
        Store briefing announcements to Supabase briefing_announcements table.
        
        Args:
            briefings: List of briefing dicts matching briefing_announcements schema
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            print("⚠️  Supabase not available, skipping briefing storage")
            return False
        
        if not briefings:
            print("📭 No briefings to store")
            return True
        
        try:
            # Deduplicate by ID (same event may appear for multiple calendar users)
            seen_ids = set()
            records = []
            for b in briefings:
                record_id = b.get("id")
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    records.append(b)
            
            if len(records) < len(briefings):
                print(f"   ℹ️  Deduplicated {len(briefings) - len(records)} duplicate briefings")
            
            # Upsert to Supabase (will update existing reminders with same ID)
            self._client.table(self.TABLE_NAME).upsert(records).execute()
            
            print(f"📅 Stored {len(records)} calendar reminder briefings to Supabase (briefing_announcements)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store briefings: {e}")
            return False
    
    def get_pending_reminders(
        self,
        user: str,
    ) -> List[Dict[str, Any]]:
        """
        Get pending calendar reminder briefings for a user.
        
        Args:
            user: User ID to fetch reminders for
            
        Returns:
            List of pending reminder briefings from briefing_announcements table
        """
        if not self.is_available():
            return []
        
        try:
            # Fetch pending calendar reminders for user
            response = (
                self._client.table(self.TABLE_NAME)
                .select("*")
                .eq("user_id", user)
                .eq("status", "pending")
                .like("id", "cal_reminder_%")
                .order("created_at")
                .execute()
            )
            
            return response.data or []
            
        except Exception as e:
            print(f"❌ Failed to fetch pending reminders: {e}")
            return []
    
    def cleanup_delivered_reminders(self, older_than_days: int = 7) -> int:
        """
        Delete delivered calendar reminder briefings older than specified days.
        
        Args:
            older_than_days: Delete delivered reminders older than this
            
        Returns:
            Number of deleted records
        """
        if not self.is_available():
            return 0
        
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
            
            # Delete old delivered calendar reminders
            response = (
                self._client.table(self.TABLE_NAME)
                .delete()
                .like("id", "cal_reminder_%")
                .eq("status", "delivered")
                .lt("delivered_at", cutoff)
                .execute()
            )
            
            # Count isn't directly available, estimate from response
            count = len(response.data) if response.data else 0
            if count > 0:
                print(f"🧹 Cleaned up {count} old delivered calendar reminders")
            
            return count
            
        except Exception as e:
            print(f"❌ Failed to cleanup delivered reminders: {e}")
            return 0
    
    def format_briefing_summary(self, briefings: List[Dict[str, Any]]) -> str:
        """Format briefings into a human-readable summary."""
        if not briefings:
            return "No briefings created."
        
        # Count how many have openers
        with_openers = sum(1 for b in briefings if b.get("opener_text"))
        
        lines = [
            f"📣 Created {len(briefings)} Reminder Briefings ({with_openers} with openers)",
            "=" * 50,
        ]
        
        for b in briefings[:10]:  # Limit display
            content = b.get("content", {})
            meta = content.get("meta", {})
            
            event_title = meta.get("event_title", "Event")
            active_from = content.get("active_from", "")
            reminder_at = meta.get("reminder_at_iso", "")
            user = b.get("user_id", "")
            opener_text = b.get("opener_text", "")
            
            # Parse reminder_at for display
            try:
                dt = datetime.fromisoformat(reminder_at.replace("Z", "+00:00"))
                dt_local = dt.astimezone(self.calculator.tz)
                time_str = dt_local.strftime("%b %d at %I:%M %p")
            except (ValueError, AttributeError):
                time_str = active_from
            
            lines.append(f"\n📅 {event_title}")
            lines.append(f"   👤 {user} | 🕐 Active: {time_str}")
            if opener_text:
                lines.append(f"   🗣️ \"{opener_text[:80]}{'...' if len(opener_text) > 80 else ''}\"")
            else:
                lines.append(f"   ⚠️ No opener generated")
        
        if len(briefings) > 10:
            lines.append(f"\n... and {len(briefings) - 10} more")
        
        return "\n".join(lines)

