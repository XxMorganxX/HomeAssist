"""
Briefing Announcements Manager

Manages a persistent queue of briefing announcements that are reported to the user on wake word.
Briefings persist until explicitly dismissed by the user.

Supabase table setup (run in Supabase SQL editor):

CREATE TABLE briefing_announcements (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content JSONB NOT NULL,            -- { message: str, llm_instructions?: str, meta?: {...} }
    opener_text TEXT,                  -- Pre-generated conversation opener (via BriefingProcessor)
    priority TEXT DEFAULT 'normal',    -- 'high', 'normal', 'low'
    status TEXT DEFAULT 'pending',     -- 'pending', 'delivered', 'dismissed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    dismissed_at TIMESTAMPTZ
);

CREATE INDEX idx_briefings_user_status ON briefing_announcements(user_id, status);
CREATE INDEX idx_briefings_created ON briefing_announcements(created_at DESC);

Content structure:
{
    "message": "Your package was delivered at 2pm",        # Required
    "llm_instructions": "Mention this casually",           # Optional
    "active_from": "2026-01-05",                           # Optional - ISO date (YYYY-MM-DD) when briefing becomes active
    "meta": {                                              # Optional
        "timestamp": "2026-01-03T14:00:00Z",
        "source": "delivery_tracker"
    }
}

The `active_from` field allows scheduling briefings to become active on a future date.
If set, the briefing will NOT be included in openers until the current date >= active_from date.

Workflow:
1. Briefing is inserted (by scheduled script, MCP tool, etc.) with content but no opener_text
2. BriefingProcessor.process_pending_briefings() generates opener_text via LLM
3. On wake word, orchestrator fetches briefings with opener_text and speaks via TTS (no LLM latency)
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

from dotenv import load_dotenv

load_dotenv()


def _format_time_until(event_datetime: datetime) -> str:
    """
    Calculate and format the exact time remaining until an event.
    
    Args:
        event_datetime: The event's datetime (timezone-aware)
        
    Returns:
        Human-readable exact time string like "15 minutes", "1 hour and 23 minutes", "2 days"
    """
    now = datetime.now(timezone.utc)
    
    # Ensure event_datetime is timezone-aware
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=timezone.utc)
    else:
        event_datetime = event_datetime.astimezone(timezone.utc)
    
    delta = event_datetime - now
    total_seconds = delta.total_seconds()
    
    if total_seconds <= 0:
        return "right now"
    
    total_minutes = int(total_seconds / 60)
    
    # Less than 1 minute
    if total_minutes < 1:
        return "less than a minute"
    
    # Less than 60 minutes - show exact minutes
    if total_minutes < 60:
        if total_minutes == 1:
            return "1 minute"
        return f"{total_minutes} minutes"
    
    # Less than 24 hours - show hours and minutes
    hours = total_minutes // 60
    remaining_minutes = total_minutes % 60
    
    if total_minutes < 1440:  # Less than 24 hours
        if hours == 1:
            hour_str = "1 hour"
        else:
            hour_str = f"{hours} hours"
        
        if remaining_minutes == 0:
            return hour_str
        elif remaining_minutes == 1:
            return f"{hour_str} and 1 minute"
        else:
            return f"{hour_str} and {remaining_minutes} minutes"
    
    # 24 hours or more - show days and hours
    days = total_minutes // 1440
    remaining_hours = (total_minutes % 1440) // 60
    
    if days == 1:
        day_str = "1 day"
    else:
        day_str = f"{days} days"
    
    if remaining_hours == 0:
        return day_str
    elif remaining_hours == 1:
        return f"{day_str} and 1 hour"
    else:
        return f"{day_str} and {remaining_hours} hours"


def _substitute_time_placeholder(opener: str, briefing: Dict[str, Any]) -> str:
    """
    Replace {{TIME_UNTIL_EVENT}} placeholder with actual calculated time.
    
    Args:
        opener: The opener text with potential placeholder
        briefing: The briefing dict containing event metadata
        
    Returns:
        Opener with placeholder substituted
    """
    if "{{TIME_UNTIL_EVENT}}" not in opener:
        return opener
    
    # Get event datetime from briefing metadata
    content = briefing.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return opener.replace("{{TIME_UNTIL_EVENT}}", "soon")
    
    meta = content.get("meta", {})
    event_datetime_iso = meta.get("event_datetime_iso")
    
    if not event_datetime_iso:
        # Try alternative fields
        event_datetime_iso = content.get("event_datetime_iso") or meta.get("reminder_at_iso")
    
    if not event_datetime_iso:
        return opener.replace("{{TIME_UNTIL_EVENT}}", "soon")
    
    try:
        # Parse the event datetime
        event_dt = datetime.fromisoformat(event_datetime_iso.replace("Z", "+00:00"))
        time_until = _format_time_until(event_dt)
        return opener.replace("{{TIME_UNTIL_EVENT}}", time_until)
    except (ValueError, TypeError):
        return opener.replace("{{TIME_UNTIL_EVENT}}", "soon")


def _is_briefing_active(briefing: Dict[str, Any]) -> bool:
    """
    Check if a briefing is active based on its active_from datetime.
    
    Args:
        briefing: The briefing record (with 'content' field)
        
    Returns:
        True if the briefing is active (should be included in openers),
        False if it has a future active_from datetime
    """
    content = briefing.get("content", {})
    
    # Handle content as string
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return True  # No active_from, so it's active
    
    active_from = content.get("active_from")
    if not active_from:
        return True  # No active_from means it's always active
    
    now = datetime.now(timezone.utc)
    
    try:
        # Try parsing as full ISO datetime first (e.g., "2026-01-06T14:00:00-05:00")
        if "T" in active_from:
            active_dt = datetime.fromisoformat(active_from.replace("Z", "+00:00"))
            return now >= active_dt
        else:
            # Fall back to date-only format (YYYY-MM-DD) - active from start of that day
            active_date = datetime.strptime(active_from, "%Y-%m-%d").date()
            return now.date() >= active_date
    except (ValueError, TypeError):
        # If parsing fails, treat as active
        return True


class BriefingManager:
    """
    Manages briefing announcements stored in Supabase with local file fallback.
    
    Usage:
        manager = BriefingManager()
        
        # Get pending briefings for a user
        briefings = await manager.get_pending_briefings(user="Morgan")
        
        # Mark briefings as delivered after speaking them
        await manager.mark_delivered([b['id'] for b in briefings])
    """
    
    TABLE_NAME = "briefing_announcements"
    LOCAL_FALLBACK_PATH = Path(__file__).parent.parent.parent / "state_management" / "briefing_announcements.json"
    
    def __init__(self):
        """Initialize the BriefingManager with Supabase client."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self._client: Optional[Client] = None
        self._initialized = False
        
        if not SUPABASE_AVAILABLE:
            print("⚠️  BriefingManager: supabase package not installed, using local fallback")
            return
        
        if not self.url or not self.key:
            print("⚠️  BriefingManager: SUPABASE_URL or SUPABASE_KEY not set, using local fallback")
            return
        
        try:
            self._client = create_client(self.url, self.key)
            self._initialized = True
        except Exception as e:
            print(f"⚠️  BriefingManager: Failed to initialize Supabase - {e}, using local fallback")
    
    def is_available(self) -> bool:
        """Check if Supabase is available."""
        return self._initialized and self._client is not None
    
    async def get_pending_briefings(self, user: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending briefings for a user, ordered by priority then created_at.
        
        Args:
            user: User ID to fetch briefings for (e.g., "Morgan")
            limit: Maximum number of briefings to return
            
        Returns:
            List of briefing records with id, content, priority, status, created_at
        """
        if self.is_available():
            return await self._get_pending_from_supabase(user, limit)
        else:
            return self._get_pending_from_local(user, limit)
    
    async def _get_pending_from_supabase(self, user: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch pending briefings from Supabase."""
        try:
            response = (
                self._client.table(self.TABLE_NAME)
                .select("*")
                .eq("user_id", user)
                .eq("status", "pending")
                .order("priority", desc=False)
                .order("created_at", desc=False)
                .limit(limit * 2)  # Fetch extra to account for filtered inactive ones
                .execute()
            )
            
            if not response.data:
                return []
            
            # Filter out briefings that aren't active yet (future active_from date)
            active_briefings = [b for b in response.data if _is_briefing_active(b)]
            
            # Re-sort by priority properly (high > normal > low)
            priority_order = {"high": 0, "normal": 1, "low": 2}
            briefings = sorted(
                active_briefings,
                key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
            )[:limit]
            
            print(f"📋 BriefingManager: Found {len(briefings)} pending briefing(s) for {user}")
            return briefings
            
        except Exception as e:
            print(f"❌ BriefingManager: Error fetching from Supabase - {e}")
            return self._get_pending_from_local(user, limit)
    
    def _get_pending_from_local(self, user: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch pending briefings from local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return []
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            # Filter by user, status, and active_from date
            user_briefings = [
                b for b in data.get("briefings", [])
                if b.get("user_id") == user 
                and b.get("status") == "pending"
                and _is_briefing_active(b)
            ]
            
            priority_order = {"high": 0, "normal": 1, "low": 2}
            user_briefings.sort(
                key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
            )
            
            if user_briefings:
                print(f"📋 BriefingManager (local): Found {len(user_briefings)} pending briefing(s) for {user}")
            
            return user_briefings[:limit]
            
        except Exception as e:
            print(f"❌ BriefingManager: Error reading local fallback - {e}")
            return []
    
    async def mark_delivered(self, briefing_ids: List[str]) -> int:
        """
        Mark briefings as delivered (spoken to user).
        
        Args:
            briefing_ids: List of briefing IDs to mark as delivered
            
        Returns:
            Number of briefings successfully marked
        """
        if not briefing_ids:
            return 0
        
        if self.is_available():
            return await self._mark_delivered_supabase(briefing_ids)
        else:
            return self._mark_delivered_local(briefing_ids)
    
    async def _mark_delivered_supabase(self, briefing_ids: List[str]) -> int:
        """Mark briefings as delivered in Supabase."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            marked = 0
            
            for briefing_id in briefing_ids:
                try:
                    self._client.table(self.TABLE_NAME).update({
                        "status": "delivered",
                        "delivered_at": now
                    }).eq("id", briefing_id).execute()
                    marked += 1
                except Exception as e:
                    print(f"⚠️  BriefingManager: Failed to mark {briefing_id} as delivered - {e}")
            
            if marked:
                print(f"✅ BriefingManager: Marked {marked} briefing(s) as delivered")
            return marked
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking delivered in Supabase - {e}")
            return self._mark_delivered_local(briefing_ids)
    
    def _mark_delivered_local(self, briefing_ids: List[str]) -> int:
        """Mark briefings as delivered in local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return 0
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            now = datetime.now(timezone.utc).isoformat()
            marked = 0
            
            for briefing in data.get("briefings", []):
                if briefing.get("id") in briefing_ids and briefing.get("status") == "pending":
                    briefing["status"] = "delivered"
                    briefing["delivered_at"] = now
                    marked += 1
            
            with open(self.LOCAL_FALLBACK_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            if marked:
                print(f"✅ BriefingManager (local): Marked {marked} briefing(s) as delivered")
            return marked
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking delivered in local - {e}")
            return 0
    
    async def dismiss_briefings(self, briefing_ids: List[str]) -> int:
        """
        Dismiss briefings (remove from future queries).
        
        Args:
            briefing_ids: List of briefing IDs to dismiss
            
        Returns:
            Number of briefings successfully dismissed
        """
        if not briefing_ids:
            return 0
        
        if self.is_available():
            return await self._dismiss_supabase(briefing_ids)
        else:
            return self._dismiss_local(briefing_ids)
    
    async def _dismiss_supabase(self, briefing_ids: List[str]) -> int:
        """Dismiss briefings in Supabase."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            dismissed = 0
            
            for briefing_id in briefing_ids:
                try:
                    self._client.table(self.TABLE_NAME).update({
                        "status": "dismissed",
                        "dismissed_at": now
                    }).eq("id", briefing_id).execute()
                    dismissed += 1
                except Exception as e:
                    print(f"⚠️  BriefingManager: Failed to dismiss {briefing_id} - {e}")
            
            if dismissed:
                print(f"✅ BriefingManager: Dismissed {dismissed} briefing(s)")
            return dismissed
            
        except Exception as e:
            print(f"❌ BriefingManager: Error dismissing in Supabase - {e}")
            return 0
    
    def _dismiss_local(self, briefing_ids: List[str]) -> int:
        """Dismiss briefings in local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return 0
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            now = datetime.now(timezone.utc).isoformat()
            dismissed = 0
            
            for briefing in data.get("briefings", []):
                if briefing.get("id") in briefing_ids:
                    briefing["status"] = "dismissed"
                    briefing["dismissed_at"] = now
                    dismissed += 1
            
            with open(self.LOCAL_FALLBACK_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            if dismissed:
                print(f"✅ BriefingManager (local): Dismissed {dismissed} briefing(s)")
            return dismissed
            
        except Exception as e:
            print(f"❌ BriefingManager: Error dismissing in local - {e}")
            return 0
    
    # =========================================================================
    # OPENER TEXT MANAGEMENT (for pre-generated conversation openers)
    # =========================================================================
    
    async def get_pending_briefings_without_opener(self, user: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending briefings that don't have a pre-generated opener yet.
        
        Args:
            user: User ID
            limit: Max briefings to return
            
        Returns:
            List of briefings needing opener generation
        """
        if self.is_available():
            try:
                response = (
                    self._client.table(self.TABLE_NAME)
                    .select("*")
                    .eq("user_id", user)
                    .eq("status", "pending")
                    .is_("opener_text", "null")
                    .order("created_at", desc=False)
                    .limit(limit)
                    .execute()
                )
                return response.data or []
            except Exception as e:
                print(f"❌ BriefingManager: Error fetching briefings without opener - {e}")
                return []
        else:
            try:
                if not self.LOCAL_FALLBACK_PATH.exists():
                    return []
                with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                    data = json.load(f)
                return [
                    b for b in data.get("briefings", [])
                    if b.get("user_id") == user 
                    and b.get("status") == "pending"
                    and not b.get("opener_text")
                ][:limit]
            except Exception:
                return []
    
    async def get_pending_briefings_with_opener(self, user: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending briefings that have a pre-generated opener (ready for TTS).
        
        Only returns briefings that are currently active (active_from date <= today).
        
        Args:
            user: User ID
            limit: Max briefings to return
            
        Returns:
            List of briefings with opener_text ready to speak
        """
        if self.is_available():
            try:
                response = (
                    self._client.table(self.TABLE_NAME)
                    .select("*")
                    .eq("user_id", user)
                    .eq("status", "pending")
                    .not_.is_("opener_text", "null")
                    .order("priority", desc=False)
                    .order("created_at", desc=False)
                    .limit(limit * 2)  # Fetch extra to account for filtered inactive ones
                    .execute()
                )
                
                if not response.data:
                    return []
                
                # Filter out briefings that aren't active yet (future active_from date)
                active_briefings = [b for b in response.data if _is_briefing_active(b)]
                
                priority_order = {"high": 0, "normal": 1, "low": 2}
                briefings = sorted(
                    active_briefings,
                    key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
                )[:limit]
                
                if briefings:
                    print(f"📋 BriefingManager: Found {len(briefings)} briefing(s) with opener for {user}")
                return briefings
                
            except Exception as e:
                print(f"❌ BriefingManager: Error fetching briefings with opener - {e}")
                return []
        else:
            try:
                if not self.LOCAL_FALLBACK_PATH.exists():
                    return []
                with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                    data = json.load(f)
                
                # Filter by user, status, opener_text, and active_from date
                briefings = [
                    b for b in data.get("briefings", [])
                    if b.get("user_id") == user 
                    and b.get("status") == "pending"
                    and b.get("opener_text")
                    and _is_briefing_active(b)
                ]
                
                priority_order = {"high": 0, "normal": 1, "low": 2}
                briefings.sort(key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", "")))
                
                if briefings:
                    print(f"📋 BriefingManager (local): Found {len(briefings)} briefing(s) with opener for {user}")
                return briefings[:limit]
            except Exception:
                return []
    
    async def update_opener(self, briefing_id: str, opener_text: str) -> bool:
        """
        Store a pre-generated opener for a briefing.
        
        Args:
            briefing_id: ID of the briefing
            opener_text: The generated conversation opener
            
        Returns:
            True if successful
        """
        if self.is_available():
            try:
                self._client.table(self.TABLE_NAME).update({
                    "opener_text": opener_text
                }).eq("id", briefing_id).execute()
                print(f"✅ BriefingManager: Updated opener for {briefing_id}")
                return True
            except Exception as e:
                print(f"❌ BriefingManager: Error updating opener - {e}")
                return False
        else:
            try:
                if not self.LOCAL_FALLBACK_PATH.exists():
                    return False
                with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                    data = json.load(f)
                
                for briefing in data.get("briefings", []):
                    if briefing.get("id") == briefing_id:
                        briefing["opener_text"] = opener_text
                        break
                
                with open(self.LOCAL_FALLBACK_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ BriefingManager (local): Updated opener for {briefing_id}")
                return True
            except Exception as e:
                print(f"❌ BriefingManager: Error updating opener locally - {e}")
                return False
    
    def get_combined_opener(self, briefings: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get a combined opener from multiple briefings.
        
        Automatically substitutes {{TIME_UNTIL_EVENT}} placeholders with
        the actual calculated time remaining until each event.
        
        Args:
            briefings: List of briefings with opener_text
            
        Returns:
            Combined opener string with time placeholders resolved, or None if no openers
        """
        processed_openers = []
        
        for briefing in briefings:
            opener = briefing.get("opener_text")
            if opener:
                # Substitute time placeholder with real-time calculation
                opener = _substitute_time_placeholder(opener, briefing)
                processed_openers.append(opener)
        
        if not processed_openers:
            return None
        
        if len(processed_openers) == 1:
            return processed_openers[0]
        
        return " ".join(processed_openers)


def build_briefing_prompt(briefings: List[Dict[str, Any]]) -> str:
    """
    Build a briefing prompt from a list of briefings for the LLM.
    
    Args:
        briefings: List of briefing records from get_pending_briefings()
        
    Returns:
        A formatted string for the LLM to use when briefing the user
    """
    if not briefings:
        return ""
    
    parts = []
    parts.append("You have pending briefings to report to the user at the start of this conversation.")
    parts.append("Present these naturally and briefly before asking how you can help:\n")
    
    for i, briefing in enumerate(briefings, 1):
        content = briefing.get("content", {})
        
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"message": content}
        
        message = content.get("message", "") or content.get("fact", "")  # Support legacy 'fact' key
        llm_instructions = content.get("llm_instructions", "")
        meta = content.get("meta", {})
        
        part = f"Briefing {i}: {message}"
        
        if llm_instructions:
            part += f"\n   [Instruction: {llm_instructions}]"
        
        if meta:
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            part += f"\n   [Meta: {meta_str}]"
        
        parts.append(part)
    
    return "\n".join(parts)
