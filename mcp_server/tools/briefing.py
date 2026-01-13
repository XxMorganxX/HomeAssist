"""
Briefing Management Tool using BaseTool.

This tool allows the assistant to create, list, and dismiss briefing announcements
that are spoken to users when they wake up the assistant.

Briefings are stored in Supabase (briefing_announcements table) and are delivered
when the user activates the assistant with a briefing-enabled wake word.
"""

import json
import os
import uuid
from typing import Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS

# Import user config for dynamic user resolution
try:
    from mcp_server.user_config import get_notification_users, get_default_notification_user
except ImportError:
    def get_notification_users():
        return ["Morgan", "Spencer"]
    def get_default_notification_user():
        return "Morgan"

# Import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

# Default timezone
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")


class BriefingTool(BaseTool):
    """Tool to create, list, and manage briefing announcements for the user."""
    
    name = "briefing"
    description = """Create and manage briefing announcements that are spoken to users when they wake up the assistant.

Use this tool when the user wants to:
- Create a reminder or briefing to tell them something later
- Schedule a notification to be announced at a specific time
- Check what pending briefings they have
- Dismiss or cancel a pending briefing

IMPORTANT: Briefings are different from calendar events. Briefings are announcements 
spoken by the assistant, while calendar events are appointments on the user's calendar.

REMINDER TIMING - Two ways to specify when to remind:
1. ABSOLUTE TIME: Use 'remind_at' with a specific datetime
   - "Remind me at 12:30pm" → remind_at='2026-01-13T12:30:00'
2. RELATIVE TO EVENT: Use 'remind_before_minutes' with the event time
   - "Remind me 30 minutes before" → event_time='2026-01-13T13:00:00', remind_before_minutes=30

You MUST provide either 'remind_at' OR both 'event_time' and 'remind_before_minutes'.

Actions:
- 'create': Create a new briefing announcement (requires timing)
- 'list': List pending briefings for a user
- 'dismiss': Cancel/remove a pending briefing

Examples:
- "Set a reminder called 'Call Merrick' at 1pm and remind me at 12:30" 
  → action='create', message='Call Merrick', event_time='2026-01-13T13:00:00', remind_at='2026-01-13T12:30:00'

- "Set a reminder for 1pm tomorrow and remind me 30 minutes before"
  → action='create', message='...', event_time='2026-01-13T13:00:00', remind_before_minutes=30

- "What briefings do I have pending?" → action='list'
- "Cancel that reminder" → action='dismiss', briefing_id='...'"""
    version = "1.1.0"
    
    TABLE_NAME = "briefing_announcements"
    
    def __init__(self):
        """Initialize the briefing tool."""
        super().__init__()
        
        # Get configured users dynamically
        self._configured_users = get_notification_users()
        self._default_user = get_default_notification_user()
        
        # Timezone for parsing
        self._tz = ZoneInfo(DEFAULT_TIMEZONE)
        
        # Initialize Supabase client
        self._supabase_client: Optional[Client] = None
        self._supabase_available = False
        self._init_supabase()
    
    def _init_supabase(self):
        """Initialize Supabase client for briefing management."""
        if not SUPABASE_AVAILABLE:
            self.logger.debug("Supabase package not installed")
            return
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            self.logger.debug("SUPABASE_URL or SUPABASE_KEY not set")
            return
        
        try:
            self._supabase_client = create_client(url, key)
            self._supabase_available = True
            self.logger.debug("Supabase client initialized for briefings")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Supabase client: {e}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool with detailed parameter descriptions."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform. 'create' to add a new briefing, 'list' to see pending briefings, 'dismiss' to cancel a briefing.",
                    "enum": ["create", "list", "dismiss"],
                    "default": "create"
                },
                "user": {
                    "type": "string",
                    "description": f"Which user the briefing is for. Available users: {', '.join(self._configured_users)}",
                    "enum": self._configured_users,
                    "default": self._default_user
                },
                "message": {
                    "type": "string",
                    "description": "The message/content of the briefing. Required for 'create' action. This is what will be announced to the user. Be clear and concise (e.g., 'Call Merrick', 'Respond to email')."
                },
                "event_time": {
                    "type": "string",
                    "description": "When the actual event/task is scheduled. Use ISO 8601 format (e.g., '2026-01-13T13:00:00' for 1pm). Required when using 'remind_before_minutes'. This is the time of the event itself, NOT when you want to be reminded."
                },
                "remind_at": {
                    "type": "string",
                    "description": "ABSOLUTE reminder time - exactly when the briefing should be announced. Use ISO 8601 format (e.g., '2026-01-13T12:30:00' for 12:30pm). Use this when the user says 'remind me AT [time]'."
                },
                "remind_before_minutes": {
                    "type": "integer",
                    "description": "RELATIVE reminder time - how many minutes before 'event_time' to announce the briefing. Use this when the user says 'remind me X minutes before'. Requires 'event_time' to be set.",
                    "minimum": 1,
                    "maximum": 10080
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level of the briefing. 'high' for urgent items, 'normal' for regular reminders, 'low' for FYI items.",
                    "enum": ["high", "normal", "low"],
                    "default": "normal"
                },
                "briefing_id": {
                    "type": "string",
                    "description": "The ID of a specific briefing. Required for 'dismiss' action."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of briefings to return for 'list' action.",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                }
            },
            "required": ["action"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the briefing tool."""
        try:
            action = params.get("action", "create")
            user = params.get("user", self._default_user)
            
            # Normalize user
            user = self._normalize_user(user)
            
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Briefing -- %s", params)
            
            if action == "create":
                return self._create_briefing(params, user)
            elif action == "list":
                return self._list_briefings(params, user)
            elif action == "dismiss":
                return self._dismiss_briefing(params, user)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}. Valid actions: create, list, dismiss"
                }
        
        except Exception as e:
            self.logger.error(f"Error executing briefing tool: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _normalize_user(self, user: str) -> str:
        """Normalize user input to match configured users."""
        if not user:
            return self._default_user
        
        user_lower = user.lower().strip()
        
        # Check against configured users
        for configured_user in self._configured_users:
            if user_lower == configured_user.lower():
                return configured_user
        
        # Handle common aliases
        if user_lower in {"me", "my", "default"}:
            return self._default_user
        
        # Default to title case
        return user.title()
    
    def _parse_datetime(self, dt_string: str) -> datetime:
        """Parse a datetime string into a timezone-aware datetime."""
        if not dt_string:
            raise ValueError("Datetime string is empty")
        
        # Handle various formats
        if "T" in dt_string:
            # Full ISO datetime
            dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
            # If no timezone, assume local
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=self._tz)
        else:
            # Date only - assume start of day in local timezone
            dt = datetime.strptime(dt_string, "%Y-%m-%d")
            dt = dt.replace(tzinfo=self._tz)
        
        return dt
    
    def _format_time_until(self, minutes: int) -> str:
        """Format minutes into human-readable time description."""
        if minutes >= 1440:
            days = minutes // 1440
            hours = (minutes % 1440) // 60
            if hours > 0:
                return f"{days} day{'s' if days > 1 else ''} and {hours} hour{'s' if hours > 1 else ''}"
            return f"{days} day{'s' if days > 1 else ''}"
        elif minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            if mins > 0:
                return f"{hours} hour{'s' if hours > 1 else ''} and {mins} minute{'s' if mins > 1 else ''}"
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
    
    def _create_briefing(self, params: Dict[str, Any], user: str) -> Dict[str, Any]:
        """Create a new briefing announcement."""
        message = params.get("message")
        
        if not message:
            return {
                "success": False,
                "error": "Message is required to create a briefing. Please provide a message for the announcement (e.g., 'Call Merrick')."
            }
        
        event_time = params.get("event_time")
        remind_at = params.get("remind_at")
        remind_before_minutes = params.get("remind_before_minutes")
        priority = params.get("priority", "normal")
        
        # Validate timing parameters
        # Must have either:
        # 1. remind_at (absolute reminder time), OR
        # 2. event_time + remind_before_minutes (relative reminder)
        
        if not remind_at and not (event_time and remind_before_minutes):
            return {
                "success": False,
                "error": "Timing is required. Please provide either:\n"
                         "1. 'remind_at' - exact time to remind (e.g., '2026-01-13T12:30:00'), OR\n"
                         "2. 'event_time' AND 'remind_before_minutes' - to remind X minutes before the event\n\n"
                         "Examples:\n"
                         "- 'Remind me at 12:30' → remind_at='2026-01-13T12:30:00'\n"
                         "- 'Remind me 30 minutes before' → event_time='2026-01-13T13:00:00', remind_before_minutes=30"
            }
        
        # Parse and calculate the active_from (when reminder fires)
        event_dt = None
        active_from_dt = None
        
        try:
            if remind_at:
                # Absolute reminder time
                active_from_dt = self._parse_datetime(remind_at)
                
                # If event_time also provided, parse it for context
                if event_time:
                    event_dt = self._parse_datetime(event_time)
            else:
                # Relative to event time
                event_dt = self._parse_datetime(event_time)
                active_from_dt = event_dt - timedelta(minutes=remind_before_minutes)
                
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid datetime format: {e}. Use ISO 8601 format like '2026-01-13T13:00:00'"
            }
        
        # Validate that reminder time is in the future
        now = datetime.now(timezone.utc)
        if active_from_dt.astimezone(timezone.utc) < now:
            return {
                "success": False,
                "error": f"Reminder time {active_from_dt.isoformat()} is in the past. Please provide a future time."
            }
        
        # Generate unique ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        briefing_id = f"manual_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Build the message with time context
        # Include {{TIME_UNTIL_EVENT}} placeholder for dynamic time calculation at delivery
        if event_dt:
            display_message = f"{message}. Your event is in {{{{TIME_UNTIL_EVENT}}}}."
        else:
            display_message = message
        
        # Build content JSONB
        content = {
            "message": display_message,
            "llm_instructions": f"Remind the user: {message}. Be concise and natural.",
            "active_from": active_from_dt.isoformat(),
            "meta": {
                "source": "assistant_created",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_message": message
            }
        }
        
        # Store event time if available (for time-until calculations)
        if event_dt:
            content["meta"]["event_datetime_iso"] = event_dt.isoformat()
            content["meta"]["remind_before_minutes"] = remind_before_minutes
        
        # Build the briefing record
        record = {
            "id": briefing_id,
            "user_id": user,
            "content": content,
            "priority": priority,
            "status": "pending"
        }
        
        # Store to Supabase
        if not self._supabase_available:
            return {
                "success": False,
                "error": "Supabase is not available. Cannot store briefing."
            }
        
        try:
            # Convert content to JSON string for Supabase JSONB
            record_for_db = record.copy()
            record_for_db["content"] = json.dumps(content)
            
            self._supabase_client.table(self.TABLE_NAME).insert(record_for_db).execute()
            
            # Build response
            response = {
                "success": True,
                "action": "create",
                "briefing_id": briefing_id,
                "user": user,
                "message": message,
                "priority": priority,
                "status": "pending",
                "remind_at": active_from_dt.isoformat()
            }
            
            if event_dt:
                response["event_time"] = event_dt.isoformat()
                response["remind_before_minutes"] = remind_before_minutes
                response["note"] = f"Reminder set for {self._format_time_until(remind_before_minutes)} before your event at {event_dt.strftime('%I:%M %p on %b %d')}"
            else:
                response["note"] = f"Reminder set for {active_from_dt.strftime('%I:%M %p on %b %d')}"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to store briefing: {e}")
            return {
                "success": False,
                "error": f"Failed to store briefing: {str(e)}"
            }
    
    def _list_briefings(self, params: Dict[str, Any], user: str) -> Dict[str, Any]:
        """List pending briefings for a user."""
        limit = params.get("limit", 10)
        
        if not self._supabase_available:
            return {
                "success": False,
                "error": "Supabase is not available. Cannot retrieve briefings."
            }
        
        try:
            response = (
                self._supabase_client.table(self.TABLE_NAME)
                .select("id, content, priority, status, created_at")
                .eq("user_id", user)
                .eq("status", "pending")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            briefings = []
            for row in response.data or []:
                content = row.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        content = {}
                
                meta = content.get("meta", {})
                
                briefing = {
                    "id": row.get("id"),
                    "message": meta.get("original_message") or content.get("message", ""),
                    "priority": row.get("priority", "normal"),
                    "created_at": row.get("created_at"),
                    "remind_at": content.get("active_from"),
                    "event_time": meta.get("event_datetime_iso"),
                    "remind_before_minutes": meta.get("remind_before_minutes"),
                    "source": meta.get("source", "unknown")
                }
                briefings.append(briefing)
            
            return {
                "success": True,
                "action": "list",
                "user": user,
                "count": len(briefings),
                "briefings": briefings
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list briefings: {e}")
            return {
                "success": False,
                "error": f"Failed to list briefings: {str(e)}"
            }
    
    def _dismiss_briefing(self, params: Dict[str, Any], user: str) -> Dict[str, Any]:
        """Dismiss/cancel a pending briefing."""
        briefing_id = params.get("briefing_id")
        
        if not briefing_id:
            return {
                "success": False,
                "error": "briefing_id is required to dismiss a briefing. Use action='list' to see available briefings."
            }
        
        if not self._supabase_available:
            return {
                "success": False,
                "error": "Supabase is not available. Cannot dismiss briefing."
            }
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Update the briefing status
            self._supabase_client.table(self.TABLE_NAME).update({
                "status": "dismissed",
                "dismissed_at": now
            }).eq("id", briefing_id).eq("user_id", user).execute()
            
            return {
                "success": True,
                "action": "dismiss",
                "briefing_id": briefing_id,
                "user": user,
                "status": "dismissed",
                "dismissed_at": now
            }
            
        except Exception as e:
            self.logger.error(f"Failed to dismiss briefing: {e}")
            return {
                "success": False,
                "error": f"Failed to dismiss briefing: {str(e)}"
            }
