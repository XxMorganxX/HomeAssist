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
    status TEXT DEFAULT 'pending',     -- lifecycle: 'pending', 'dismissed', 'skipped', 'cancelled', 'expired'
    discord_status TEXT DEFAULT 'pending', -- delivery: 'pending', 'sent'
    discord_sent_at TIMESTAMPTZ,
    voice_status TEXT DEFAULT 'pending',   -- delivery: 'pending', 'read'
    voice_read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,          -- legacy compatibility only
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
from datetime import datetime, timezone, timedelta, time as dt_time
from pathlib import Path
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

from dotenv import load_dotenv

load_dotenv()


DISCORD_DELIVERY = "discord"
VOICE_DELIVERY = "voice"
DEFAULT_BRIEFING_WINDOWS_LOCAL = "08:30,12:30,17:30"
DEFAULT_BRIEFING_QUIET_HOURS_START = "21:00"
DEFAULT_BRIEFING_QUIET_HOURS_END = "07:30"
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")
DELIVERY_FIELD_MAP = {
    DISCORD_DELIVERY: ("discord_status", "discord_sent_at", "sent"),
    VOICE_DELIVERY: ("voice_status", "voice_read_at", "read"),
}


def _normalize_delivery_target(delivery_target: str) -> str:
    target = (delivery_target or "").strip().lower()
    if target in DELIVERY_FIELD_MAP:
        return target
    raise ValueError(f"Unsupported delivery target: {delivery_target}")


def _get_delivery_status(briefing: Dict[str, Any], delivery_target: str) -> str:
    target = _normalize_delivery_target(delivery_target)
    status_field, _, terminal_status = DELIVERY_FIELD_MAP[target]
    current = briefing.get(status_field)
    if isinstance(current, str) and current:
        return current

    # Backward compatibility for rows created before the split delivery schema.
    # Legacy delivery implied the assistant already spoke the item, but it says
    # nothing about whether Discord has posted it under the split delivery model.
    if briefing.get("status") == "delivered" or briefing.get("delivered_at"):
        if target == VOICE_DELIVERY:
            return terminal_status
        return "pending"
    return "pending"


def _is_delivery_pending(briefing: Dict[str, Any], delivery_target: str) -> bool:
    return _get_delivery_status(briefing, delivery_target) == "pending"


def _is_any_delivery_pending(briefing: Dict[str, Any]) -> bool:
    return _is_delivery_pending(briefing, DISCORD_DELIVERY) or _is_delivery_pending(briefing, VOICE_DELIVERY)


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
    Replace {{TIME_UNTIL_EVENT}} and {{TIME_UNTIL_DUE}} placeholders with
    real-time calculated durations.
    """
    if "{{TIME_UNTIL_EVENT}}" not in opener and "{{TIME_UNTIL_DUE}}" not in opener:
        return opener

    content = briefing.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = {}
    if not isinstance(content, dict):
        content = {}
    meta = content.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    def replace_placeholder(text: str, placeholder: str, candidates: List[Any]) -> str:
        if placeholder not in text:
            return text
        target_iso = next((str(value) for value in candidates if isinstance(value, str) and value), "")
        if not target_iso:
            return text.replace(placeholder, "soon")
        try:
            target_dt = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
            return text.replace(placeholder, _format_time_until(target_dt))
        except (ValueError, TypeError):
            return text.replace(placeholder, "soon")

    opener = replace_placeholder(
        opener,
        "{{TIME_UNTIL_EVENT}}",
        [
            meta.get("event_datetime_iso"),
            content.get("event_datetime_iso"),
            meta.get("reminder_at_iso"),
        ],
    )
    opener = replace_placeholder(
        opener,
        "{{TIME_UNTIL_DUE}}",
        [
            meta.get("due_at_iso"),
            content.get("due_at_iso"),
            meta.get("event_datetime_iso"),
        ],
    )
    return opener


def _parse_clock_time(raw: Any, fallback: dt_time) -> dt_time:
    value = str(raw or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if not match:
        return fallback
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return fallback
    return dt_time(hour=hour, minute=minute)


def _parse_window_times(raw: Any) -> List[dt_time]:
    fallback = [
        dt_time(hour=8, minute=30),
        dt_time(hour=12, minute=30),
        dt_time(hour=17, minute=30),
    ]
    if not raw:
        return fallback
    parsed: List[dt_time] = []
    for part in str(raw).split(","):
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", part.strip())
        if not match:
            continue
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            continue
        parsed.append(dt_time(hour=hour, minute=minute))
    if not parsed:
        return fallback
    deduped = sorted({(item.hour, item.minute) for item in parsed})
    return [dt_time(hour=hour, minute=minute) for hour, minute in deduped]


def _is_within_quiet_hours(local_dt: datetime, *, start: dt_time, end: dt_time) -> bool:
    local_time = local_dt.time().replace(tzinfo=None)
    if start == end:
        return False
    if start < end:
        return start <= local_time < end
    return local_time >= start or local_time < end


def _first_window_at_or_after(
    base_local: datetime,
    *,
    windows: List[dt_time],
    quiet_start: dt_time,
    quiet_end: dt_time,
    tz: ZoneInfo,
) -> datetime:
    start_date = base_local.date()
    for day_offset in range(0, 35):
        current_date = start_date + timedelta(days=day_offset)
        for window in windows:
            candidate = datetime.combine(current_date, window).replace(tzinfo=tz)
            if candidate < base_local:
                continue
            if _is_within_quiet_hours(candidate, start=quiet_start, end=quiet_end):
                continue
            return candidate
    return base_local


def _compute_digest_active_from(briefing: Dict[str, Any], meta: Dict[str, Any]) -> Optional[datetime]:
    source = str(meta.get("source") or "")
    if source not in {"todo_digest", "weather_digest"}:
        return None

    base_ref = briefing.get("created_at") or meta.get("generated_at") or meta.get("generated_on")
    try:
        base_dt = datetime.fromisoformat(str(base_ref).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        base_dt = datetime.now(timezone.utc)

    if base_dt.tzinfo is None:
        base_dt = base_dt.replace(tzinfo=timezone.utc)
    else:
        base_dt = base_dt.astimezone(timezone.utc)

    local_tz = ZoneInfo(DEFAULT_TIMEZONE)
    base_local = base_dt.astimezone(local_tz)
    windows = _parse_window_times(os.getenv("BRIEFING_WINDOWS_LOCAL", DEFAULT_BRIEFING_WINDOWS_LOCAL))
    quiet_start = _parse_clock_time(
        os.getenv("BRIEFING_QUIET_HOURS_START", DEFAULT_BRIEFING_QUIET_HOURS_START),
        fallback=dt_time(hour=21, minute=0),
    )
    quiet_end = _parse_clock_time(
        os.getenv("BRIEFING_QUIET_HOURS_END", DEFAULT_BRIEFING_QUIET_HOURS_END),
        fallback=dt_time(hour=7, minute=30),
    )
    planned_local = _first_window_at_or_after(
        base_local,
        windows=windows,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        tz=local_tz,
    )
    return planned_local.astimezone(timezone.utc)


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
    if not isinstance(content, dict):
        return True

    active_from = content.get("active_from")
    meta = content.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    if not active_from:
        digest_active_from = _compute_digest_active_from(briefing, meta)
        if digest_active_from is None:
            return True  # No active_from means it's always active
        return datetime.now(timezone.utc) >= digest_active_from

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


MAX_BRIEFING_AGE_HOURS = 24


def _is_briefing_expired(briefing: Dict[str, Any]) -> bool:
    """
    Check if a briefing has expired.
    
    A briefing is expired if:
    1. It has an event_datetime_iso and that event has already passed, OR
    2. It has no event datetime and is older than MAX_BRIEFING_AGE_HOURS
       (prevents stale weather/general briefings from lingering forever)
    
    Args:
        briefing: The briefing record (with 'content' and 'created_at' fields)
        
    Returns:
        True if the briefing should be skipped, False otherwise
    """
    content = briefing.get("content", {})
    
    # Handle content as string
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = {}
    
    meta = content.get("meta", {})
    now = datetime.now(timezone.utc)
    
    # Check explicit event datetime first
    event_datetime_iso = meta.get("event_datetime_iso")
    if not event_datetime_iso:
        event_datetime_iso = content.get("event_datetime_iso") or meta.get("reminder_at_iso")
    
    if event_datetime_iso:
        try:
            event_dt = datetime.fromisoformat(event_datetime_iso.replace("Z", "+00:00"))
            return event_dt < now
        except (ValueError, TypeError):
            pass
    
    # No event datetime — fall back to age-based expiry using created_at or
    # meta.generated_at (whichever is available).
    age_ref = briefing.get("created_at") or meta.get("generated_at")
    if age_ref:
        try:
            created_dt = datetime.fromisoformat(str(age_ref).replace("Z", "+00:00"))
            return (now - created_dt) > timedelta(hours=MAX_BRIEFING_AGE_HOURS)
        except (ValueError, TypeError):
            pass
    
    return False


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
    # Repo-level fallback path: <project_root>/state_management/briefing_announcements.json
    LOCAL_FALLBACK_PATH = Path(__file__).resolve().parents[3] / "state_management" / "briefing_announcements.json"
    
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
        Get pending briefings for a user that are still unread by voice.
        
        Args:
            user: User ID to fetch briefings for (e.g., "Morgan")
            limit: Maximum number of briefings to return
            
        Returns:
            List of briefing records with id, content, priority, status, created_at
        """
        return await self.get_pending_briefings_for_delivery(
            user=user,
            delivery_target=VOICE_DELIVERY,
            limit=limit,
        )

    async def get_pending_briefings_for_delivery(
        self,
        user: str,
        delivery_target: str,
        limit: int = 10,
        briefing_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get pending briefings for a specific delivery target."""
        target = _normalize_delivery_target(delivery_target)
        if self.is_available():
            return await self._get_pending_from_supabase(
                user=user,
                limit=limit,
                delivery_target=target,
                briefing_ids=briefing_ids,
            )
        return self._get_pending_from_local(
            user=user,
            limit=limit,
            delivery_target=target,
            briefing_ids=briefing_ids,
        )
    
    async def _get_pending_from_supabase(
        self,
        user: str,
        limit: int,
        delivery_target: str,
        briefing_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch pending briefings from Supabase for a specific delivery target."""
        try:
            query = (
                self._client.table(self.TABLE_NAME)
                .select("*")
                .eq("user_id", user)
                .eq("status", "pending")
                .order("priority", desc=False)
                .order("created_at", desc=False)
            )
            if briefing_ids:
                query = query.in_("id", briefing_ids)
            else:
                query = query.limit(limit * 3)
            response = query.execute()
            
            if not response.data:
                return []
            
            # Separate expired briefings (event already happened) from active ones
            expired_ids = []
            valid_briefings = []
            
            for b in response.data:
                if not _is_delivery_pending(b, delivery_target):
                    continue
                if _is_briefing_expired(b):
                    expired_ids.append(b.get("id"))
                elif _is_briefing_active(b):
                    valid_briefings.append(b)
            
            # Mark expired briefings as skipped (fire and forget)
            if expired_ids:
                await self.mark_skipped(expired_ids)
            
            # Re-sort by priority properly (high > normal > low)
            priority_order = {"high": 0, "normal": 1, "low": 2}
            briefings = sorted(
                valid_briefings,
                key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
            )[:limit]
            
            print(f"📋 BriefingManager: Found {len(briefings)} {delivery_target}-pending briefing(s) for {user}")
            return briefings
            
        except Exception as e:
            print(f"❌ BriefingManager: Error fetching from Supabase - {e}")
            return self._get_pending_from_local(user, limit, delivery_target)
    
    def _get_pending_from_local(
        self,
        user: str,
        limit: int,
        delivery_target: str,
        briefing_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch pending briefings from local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return []
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            id_set = set(briefing_ids) if briefing_ids else None
            
            # Separate expired from valid briefings
            expired_ids = []
            valid_briefings = []
            
            for b in data.get("briefings", []):
                if b.get("user_id") != user or b.get("status") != "pending":
                    continue
                if id_set and b.get("id") not in id_set:
                    continue
                if not _is_delivery_pending(b, delivery_target):
                    continue
                    
                if _is_briefing_expired(b):
                    expired_ids.append(b.get("id"))
                elif _is_briefing_active(b):
                    valid_briefings.append(b)
            
            # Mark expired briefings as skipped
            if expired_ids:
                self._mark_skipped_local(expired_ids)
            
            priority_order = {"high": 0, "normal": 1, "low": 2}
            valid_briefings.sort(
                key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
            )
            
            if valid_briefings:
                print(f"📋 BriefingManager (local): Found {len(valid_briefings)} {delivery_target}-pending briefing(s) for {user}")
            
            return valid_briefings[:limit]
            
        except Exception as e:
            print(f"❌ BriefingManager: Error reading local fallback - {e}")
            return []
    
    async def mark_delivered(self, briefing_ids: List[str]) -> int:
        """
        Backward-compatible alias for marking briefings as read by voice.
        """
        return await self.mark_voice_read(briefing_ids)

    async def mark_voice_read(self, briefing_ids: List[str]) -> int:
        """
        Mark briefings as read by the voice assistant.
        
        Args:
            briefing_ids: List of briefing IDs to mark as read
            
        Returns:
            Number of briefings successfully marked
        """
        if not briefing_ids:
            return 0
        
        if self.is_available():
            return await self._mark_delivery_status_supabase(briefing_ids, VOICE_DELIVERY)
        return self._mark_delivery_status_local(briefing_ids, VOICE_DELIVERY)
    
    async def mark_discord_sent(self, briefing_ids: List[str]) -> int:
        """Mark briefings as sent to Discord."""
        if not briefing_ids:
            return 0

        if self.is_available():
            return await self._mark_delivery_status_supabase(briefing_ids, DISCORD_DELIVERY)
        return self._mark_delivery_status_local(briefing_ids, DISCORD_DELIVERY)

    async def _mark_delivery_status_supabase(self, briefing_ids: List[str], delivery_target: str) -> int:
        """Mark briefings as delivered for a specific channel in Supabase."""
        try:
            status_field, timestamp_field, terminal_status = DELIVERY_FIELD_MAP[_normalize_delivery_target(delivery_target)]
            now = datetime.now(timezone.utc).isoformat()
            marked = 0
            finalized = 0
            
            for briefing_id in briefing_ids:
                try:
                    self._client.table(self.TABLE_NAME).update({
                        status_field: terminal_status,
                        timestamp_field: now,
                    }).eq("id", briefing_id).eq("status", "pending").execute()
                    marked += 1

                    # Finalize lifecycle when both delivery channels are complete.
                    finalize_response = (
                        self._client.table(self.TABLE_NAME)
                        .update({
                            "status": "delivered",
                            "delivered_at": now,
                        })
                        .eq("id", briefing_id)
                        .eq("status", "pending")
                        .eq("discord_status", DELIVERY_FIELD_MAP[DISCORD_DELIVERY][2])  # sent
                        .eq("voice_status", DELIVERY_FIELD_MAP[VOICE_DELIVERY][2])      # read
                        .execute()
                    )
                    if finalize_response.data:
                        finalized += len(finalize_response.data)
                except Exception as e:
                    print(f"⚠️  BriefingManager: Failed to mark {briefing_id} as {terminal_status} for {delivery_target} - {e}")
            
            if marked:
                print(f"✅ BriefingManager: Marked {marked} briefing(s) as {terminal_status} for {delivery_target}")
            if finalized:
                print(f"✅ BriefingManager: Finalized {finalized} briefing(s) as delivered")
            return marked
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking {delivery_target} delivery in Supabase - {e}")
            return self._mark_delivery_status_local(briefing_ids, delivery_target)
    
    def _mark_delivery_status_local(self, briefing_ids: List[str], delivery_target: str) -> int:
        """Mark briefings as delivered in local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return 0
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            status_field, timestamp_field, terminal_status = DELIVERY_FIELD_MAP[_normalize_delivery_target(delivery_target)]
            now = datetime.now(timezone.utc).isoformat()
            marked = 0
            finalized = 0
            
            for briefing in data.get("briefings", []):
                if briefing.get("id") in briefing_ids and briefing.get("status") == "pending":
                    briefing[status_field] = terminal_status
                    briefing[timestamp_field] = now
                    marked += 1

                # Finalize lifecycle when both delivery channels are complete.
                if (
                    briefing.get("status") == "pending"
                    and briefing.get("discord_status") == DELIVERY_FIELD_MAP[DISCORD_DELIVERY][2]  # sent
                    and briefing.get("voice_status") == DELIVERY_FIELD_MAP[VOICE_DELIVERY][2]      # read
                ):
                    briefing["status"] = "delivered"
                    briefing["delivered_at"] = now
                    finalized += 1
            
            with open(self.LOCAL_FALLBACK_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            if marked:
                print(f"✅ BriefingManager (local): Marked {marked} briefing(s) as {terminal_status} for {delivery_target}")
            if finalized:
                print(f"✅ BriefingManager (local): Finalized {finalized} briefing(s) as delivered")
            return marked
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking {delivery_target} delivery in local - {e}")
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
    
    async def mark_skipped(self, briefing_ids: List[str]) -> int:
        """
        Mark briefings as skipped (event already happened).
        
        Args:
            briefing_ids: List of briefing IDs to mark as skipped
            
        Returns:
            Number of briefings successfully marked as skipped
        """
        if not briefing_ids:
            return 0
        
        if self.is_available():
            return await self._mark_skipped_supabase(briefing_ids)
        else:
            return self._mark_skipped_local(briefing_ids)
    
    async def _mark_skipped_supabase(self, briefing_ids: List[str]) -> int:
        """Mark briefings as skipped in Supabase."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            skipped = 0
            
            for briefing_id in briefing_ids:
                try:
                    self._client.table(self.TABLE_NAME).update({
                        "status": "skipped",
                        "dismissed_at": now  # Reuse dismissed_at for skip timestamp
                    }).eq("id", briefing_id).execute()
                    skipped += 1
                except Exception as e:
                    print(f"⚠️  BriefingManager: Failed to mark {briefing_id} as skipped - {e}")
            
            if skipped:
                print(f"⏭️  BriefingManager: Marked {skipped} expired briefing(s) as skipped")
            return skipped
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking skipped in Supabase - {e}")
            return 0
    
    def _mark_skipped_local(self, briefing_ids: List[str]) -> int:
        """Mark briefings as skipped in local fallback file."""
        try:
            if not self.LOCAL_FALLBACK_PATH.exists():
                return 0
            
            with open(self.LOCAL_FALLBACK_PATH, 'r') as f:
                data = json.load(f)
            
            now = datetime.now(timezone.utc).isoformat()
            skipped = 0
            
            for briefing in data.get("briefings", []):
                if briefing.get("id") in briefing_ids and briefing.get("status") == "pending":
                    briefing["status"] = "skipped"
                    briefing["dismissed_at"] = now
                    skipped += 1
            
            with open(self.LOCAL_FALLBACK_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            if skipped:
                print(f"⏭️  BriefingManager (local): Marked {skipped} expired briefing(s) as skipped")
            return skipped
            
        except Exception as e:
            print(f"❌ BriefingManager: Error marking skipped in local - {e}")
            return 0
    
    # =========================================================================
    # OPENER TEXT MANAGEMENT (for pre-generated conversation openers)
    # =========================================================================
    
    async def get_pending_briefings_without_opener(self, user: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending briefings that don't have a pre-generated opener yet and still
        have at least one undelivered channel.
        
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
                return [
                    briefing
                    for briefing in (response.data or [])
                    if _is_any_delivery_pending(briefing)
                ]
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
                    and _is_any_delivery_pending(b)
                ][:limit]
            except Exception:
                return []
    
    async def get_pending_briefings_with_opener(
        self,
        user: str,
        limit: int = 10,
        delivery_target: str = VOICE_DELIVERY,
        briefing_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get pending briefings that have a pre-generated opener and are still
        pending for the requested delivery channel.
        
        Only returns briefings that are currently active (active_from date <= today)
        and whose events haven't already happened.
        
        Args:
            user: User ID
            limit: Max briefings to return
            delivery_target: Which delivery channel should still receive the briefing
            briefing_ids: Optional subset of IDs to check
            
        Returns:
            List of briefings with opener_text ready to speak
        """
        delivery_target = _normalize_delivery_target(delivery_target)
        if self.is_available():
            try:
                query = (
                    self._client.table(self.TABLE_NAME)
                    .select("*")
                    .eq("user_id", user)
                    .eq("status", "pending")
                    .not_.is_("opener_text", "null")
                    .order("priority", desc=False)
                    .order("created_at", desc=False)
                )
                if briefing_ids:
                    query = query.in_("id", briefing_ids)
                else:
                    query = query.limit(limit * 3)  # Fetch extra to account for filtered/skipped ones
                response = query.execute()
                
                if not response.data:
                    return []
                
                # Separate expired briefings from valid active ones
                expired_ids = []
                valid_briefings = []
                
                for b in response.data:
                    if not _is_delivery_pending(b, delivery_target):
                        continue
                    if _is_briefing_expired(b):
                        expired_ids.append(b.get("id"))
                    elif _is_briefing_active(b):
                        valid_briefings.append(b)
                
                # Mark expired briefings as skipped
                if expired_ids:
                    await self.mark_skipped(expired_ids)
                
                priority_order = {"high": 0, "normal": 1, "low": 2}
                briefings = sorted(
                    valid_briefings,
                    key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", ""))
                )[:limit]
                
                if briefings:
                    print(f"📋 BriefingManager: Found {len(briefings)} briefing(s) with opener for {user} ({delivery_target})")
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
                
                # Separate expired from valid briefings
                expired_ids = []
                valid_briefings = []
                
                for b in data.get("briefings", []):
                    if (b.get("user_id") != user 
                        or b.get("status") != "pending"
                        or not b.get("opener_text")):
                        continue
                    if briefing_ids and b.get("id") not in set(briefing_ids):
                        continue
                    if not _is_delivery_pending(b, delivery_target):
                        continue
                    
                    if _is_briefing_expired(b):
                        expired_ids.append(b.get("id"))
                    elif _is_briefing_active(b):
                        valid_briefings.append(b)
                
                # Mark expired briefings as skipped
                if expired_ids:
                    self._mark_skipped_local(expired_ids)
                
                priority_order = {"high": 0, "normal": 1, "low": 2}
                valid_briefings.sort(key=lambda b: (priority_order.get(b.get("priority", "normal"), 1), b.get("created_at", "")))
                
                if valid_briefings:
                    print(f"📋 BriefingManager (local): Found {len(valid_briefings)} briefing(s) with opener for {user} ({delivery_target})")
                return valid_briefings[:limit]
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
        
        Automatically substitutes {{TIME_UNTIL_EVENT}} / {{TIME_UNTIL_DUE}}
        placeholders with the actual calculated time remaining.
        
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
