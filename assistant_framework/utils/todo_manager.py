"""
Supabase-backed todo manager shared across MCP tools, scheduled jobs, and Discord.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    dateparser = None
    DATEPARSER_AVAILABLE = False

try:
    import parsedatetime as parsedatetime_module
    PARSEDATETIME_AVAILABLE = True
except ImportError:
    parsedatetime_module = None
    PARSEDATETIME_AVAILABLE = False

try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    Client = None
    create_client = None
    SUPABASE_AVAILABLE = False

from mcp_server.user_config import (
    get_calendar_users,
    get_default_calendar_user,
    get_notification_users,
    get_default_notification_user,
)


DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")
TODO_CACHE_ROOT = Path(__file__).resolve().parents[2] / "discord_bot" / "state" / "todos"
logger = logging.getLogger(__name__)
DEFAULT_BRIEFING_WINDOWS_LOCAL = "08:30,12:30,17:30"
DEFAULT_BRIEFING_QUIET_HOURS_START = "21:00"
DEFAULT_BRIEFING_QUIET_HOURS_END = "07:30"
DEFAULT_BRIEFING_URGENT_OVERRIDE_MINUTES = 120
_DATETIME_ALIASES = (
    (r"\btdy\b", "today"),
    (r"\b2day\b", "today"),
    (r"\btmrw\b", "tomorrow"),
    (r"\btmrrw\b", "tomorrow"),
    (r"\btmr\b", "tomorrow"),
    (r"\btmrrow\b", "tomorrow"),
    (r"\b2morrow\b", "tomorrow"),
    (r"\byday\b", "yesterday"),
    (r"\bystd\b", "yesterday"),
    (r"\btonite\b", "tonight"),
    (r"\b2nite\b", "tonight"),
    (r"\btn\b", "tonight"),
)


class TodoManager:
    """Shared CRUD and sync logic for persistent todos."""

    TABLE_NAME = "todos"
    BRIEFING_TABLE_NAME = "briefing_announcements"
    TODO_BRIEFING_SOURCE_TYPES = {"manual", "discord", "voice"}
    MAX_TIMED_TODO_BRIEFINGS = 5
    TODO_GROUP_METADATA_KEY = "todo_group"
    MAX_TODO_GROUP_LENGTH = 80
    TODO_UNDATED_BRIEFING_COOLDOWN_HOURS = 24

    def __init__(self):
        self._tz = ZoneInfo(DEFAULT_TIMEZONE)
        self._briefing_windows_local = self._parse_briefing_windows(
            os.getenv("BRIEFING_WINDOWS_LOCAL", DEFAULT_BRIEFING_WINDOWS_LOCAL)
        )
        self._briefing_quiet_hours_start = self._parse_clock_time_value(
            os.getenv("BRIEFING_QUIET_HOURS_START", DEFAULT_BRIEFING_QUIET_HOURS_START),
            fallback=dt_time(hour=21, minute=0),
        ) or dt_time(hour=21, minute=0)
        self._briefing_quiet_hours_end = self._parse_clock_time_value(
            os.getenv("BRIEFING_QUIET_HOURS_END", DEFAULT_BRIEFING_QUIET_HOURS_END),
            fallback=dt_time(hour=7, minute=30),
        ) or dt_time(hour=7, minute=30)
        self._briefing_urgent_override_minutes = self._parse_positive_int(
            os.getenv("BRIEFING_URGENT_OVERRIDE_MINUTES", str(DEFAULT_BRIEFING_URGENT_OVERRIDE_MINUTES)),
            fallback=DEFAULT_BRIEFING_URGENT_OVERRIDE_MINUTES,
        )
        self._calendar_users = get_calendar_users()
        self._default_calendar_user = get_default_calendar_user()
        self._configured_users = get_notification_users()
        self._default_user = get_default_notification_user()
        self._client: Optional[Client] = None
        self._initialized = False
        self._init_supabase()

    def _init_supabase(self) -> None:
        if not SUPABASE_AVAILABLE:
            return

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return

        try:
            self._client = create_client(url, key)
            self._initialized = True
        except Exception:
            self._client = None
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self._client is not None

    def _require_client(self) -> Client:
        if not self.is_available():
            raise RuntimeError("Supabase is not available for todos.")
        return self._client

    def normalize_user(self, user: Optional[str]) -> str:
        if not user:
            return self._default_user

        user_lower = user.lower().strip()
        for configured_user in self._configured_users:
            if user_lower == configured_user.lower():
                return configured_user

        if user_lower in {"me", "my", "default"}:
            return self._default_user

        return user.title()

    def _parse_clock_time_value(self, value: Any, *, fallback: Optional[dt_time]) -> Optional[dt_time]:
        raw = str(value or "").strip()
        if not raw:
            return fallback
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
        if not match:
            return fallback
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return fallback
        return dt_time(hour=hour, minute=minute)

    def _parse_briefing_windows(self, windows_raw: Any) -> List[dt_time]:
        fallback_windows = [
            dt_time(hour=8, minute=30),
            dt_time(hour=12, minute=30),
            dt_time(hour=17, minute=30),
        ]
        if not windows_raw:
            return fallback_windows

        parsed: List[dt_time] = []
        for part in str(windows_raw).split(","):
            parsed_time = self._parse_clock_time_value(part.strip(), fallback=None)
            if parsed_time is None:
                continue
            parsed.append(parsed_time)

        if not parsed:
            return fallback_windows

        unique_windows = sorted({(item.hour, item.minute) for item in parsed})
        return [dt_time(hour=hour, minute=minute) for hour, minute in unique_windows]

    def _parse_positive_int(self, value: Any, *, fallback: int) -> int:
        try:
            parsed = int(str(value).strip())
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
        return fallback

    def parse_due_datetime(
        self,
        due_at: Optional[str] = None,
        event_time: Optional[str] = None,
        remind_before_minutes: Optional[int] = None,
    ) -> Optional[datetime]:
        if due_at:
            return self._parse_datetime_string(due_at)

        if event_time:
            event_dt = self._parse_datetime_string(event_time)
            if event_dt and remind_before_minutes:
                return event_dt - timedelta(minutes=remind_before_minutes)
            return event_dt

        return None

    def _parse_datetime_string(self, raw: str) -> datetime:
        value = self._normalize_datetime_string(raw)
        if not value:
            raise ValueError("Datetime string is empty")

        try:
            if "T" in value:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self._tz)
                return dt
            if len(value) == 10 and value.count("-") == 2:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.replace(tzinfo=self._tz)
        except ValueError:
            pass

        if DATEPARSER_AVAILABLE and dateparser:
            settings = {
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": DEFAULT_TIMEZONE,
                "TO_TIMEZONE": DEFAULT_TIMEZONE,
            }
            parsed = dateparser.parse(value, settings=settings)
            if parsed:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=self._tz)
                return parsed

        if PARSEDATETIME_AVAILABLE and parsedatetime_module:
            calendar = parsedatetime_module.Calendar()
            parsed, status = calendar.parseDT(
                value,
                sourceTime=datetime.now(self._tz),
                tzinfo=self._tz,
            )
            if status:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=self._tz)
                return parsed

        relative_fallback = self._parse_relative_datetime_fallback(value)
        if relative_fallback is not None:
            return relative_fallback

        raise ValueError(
            f"Could not parse datetime '{raw}'. Use ISO 8601 or a natural phrase like 'tomorrow 5pm'."
        )

    def _normalize_datetime_string(self, raw: str) -> str:
        value = (raw or "").strip()
        if not value:
            return ""

        value = re.sub(r"\s+", " ", value)
        for pattern, replacement in _DATETIME_ALIASES:
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        value = re.sub(r"\b(a\.?\s*m\.?)\b", "am", value, flags=re.IGNORECASE)
        value = re.sub(r"\b(p\.?\s*m\.?)\b", "pm", value, flags=re.IGNORECASE)
        value = re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", value, flags=re.IGNORECASE)
        return value

    def _parse_relative_datetime_fallback(self, value: str) -> Optional[datetime]:
        match = re.fullmatch(r"(today|tomorrow|yesterday|tonight)(?:\s+at)?(?:\s+(.+))?", value.strip(), flags=re.IGNORECASE)
        if not match:
            return None

        keyword = match.group(1).lower()
        time_part = (match.group(2) or "").strip()
        now_local = datetime.now(self._tz)
        day_offset = {
            "yesterday": -1,
            "today": 0,
            "tonight": 0,
            "tomorrow": 1,
        }[keyword]
        target_date = now_local.date() + timedelta(days=day_offset)

        parsed_time = self._parse_time_component(time_part)
        if parsed_time is not None:
            hour, minute = parsed_time
        elif keyword == "tonight":
            hour, minute = 20, 0
        else:
            hour, minute = now_local.hour, now_local.minute

        return datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=hour,
            minute=minute,
            tzinfo=self._tz,
        )

    def _parse_time_component(self, value: str) -> Optional[Tuple[int, int]]:
        text = value.strip().lower()
        if not text:
            return None

        compact = text.replace(".", "")
        compact = re.sub(r"\s+", "", compact)
        if compact == "noon":
            return (12, 0)
        if compact == "midnight":
            return (0, 0)

        meridiem_match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)", compact)
        if meridiem_match:
            hour = int(meridiem_match.group(1))
            minute = int(meridiem_match.group(2) or 0)
            meridiem = meridiem_match.group(3)
            if not 1 <= hour <= 12 or minute > 59:
                return None
            if meridiem == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
            return (hour, minute)

        twenty_four_hour_match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))", compact)
        if twenty_four_hour_match:
            hour = int(twenty_four_hour_match.group(1))
            minute = int(twenty_four_hour_match.group(2))
            if hour > 23 or minute > 59:
                return None
            return (hour, minute)

        return None

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _generate_todo_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"todo_{stamp}_{uuid.uuid4().hex[:8]}"

    def _generate_source_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def _slugify_cache_component(self, value: str) -> str:
        text = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
        return text.strip("_") or "unknown"

    def _todo_cache_dir(self, user: str) -> Path:
        return TODO_CACHE_ROOT / self._slugify_cache_component(user)

    def _source_metadata_dict(self, todo: Dict[str, Any]) -> Dict[str, Any]:
        metadata = todo.get("source_metadata") or {}
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _normalize_todo_group(self, group: Optional[str]) -> Optional[str]:
        if group is None:
            return None
        normalized = re.sub(r"\s+", " ", str(group)).strip()
        if not normalized:
            return None
        if len(normalized) > self.MAX_TODO_GROUP_LENGTH:
            normalized = normalized[: self.MAX_TODO_GROUP_LENGTH].rstrip()
        return normalized or None

    def _todo_group_from_row(self, todo: Dict[str, Any]) -> Optional[str]:
        direct_group = self._normalize_todo_group(todo.get("group"))
        if direct_group:
            return direct_group
        metadata = self._source_metadata_dict(todo)
        return self._normalize_todo_group(metadata.get(self.TODO_GROUP_METADATA_KEY))

    def _linked_calendar_events(self, todo: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = self._source_metadata_dict(todo)
        raw_events = metadata.get("linked_calendar_events")
        normalized: List[Dict[str, Any]] = []

        if isinstance(raw_events, list):
            for item in raw_events:
                if not isinstance(item, dict):
                    continue
                event_id = str(item.get("id") or item.get("event_id") or "").strip()
                calendar_user = str(item.get("calendar_user") or "").strip()
                if not event_id or not calendar_user:
                    continue
                normalized.append(
                    {
                        "id": event_id,
                        "calendar_user": calendar_user,
                        "htmlLink": item.get("htmlLink") or item.get("linked_calendar_url"),
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "attendees": [
                            str(attendee).strip()
                            for attendee in (item.get("attendees") or [])
                            if str(attendee).strip()
                        ],
                    }
                )

        if normalized:
            return normalized

        legacy_event_id = str(
            metadata.get("linked_calendar_event_id")
            or metadata.get("calendar_event_id")
            or metadata.get("event_id")
            or ""
        ).strip()
        legacy_calendar_user = str(metadata.get("linked_calendar_user") or "").strip()
        if legacy_event_id:
            return [
                {
                    "id": legacy_event_id,
                    "calendar_user": legacy_calendar_user,
                    "htmlLink": metadata.get("linked_calendar_url"),
                    "start": metadata.get("linked_calendar_start"),
                    "end": metadata.get("linked_calendar_end"),
                    "attendees": [
                        str(attendee).strip()
                        for attendee in (metadata.get("linked_calendar_attendees") or [])
                        if str(attendee).strip()
                    ],
                }
            ]

        return []

    def _todo_has_linked_calendar_event(self, todo: Dict[str, Any], event_id: str) -> bool:
        target = str(event_id or "").strip()
        if not target:
            return False
        return any(str(item.get("id") or "").strip() == target for item in self._linked_calendar_events(todo))

    def _linked_calendar_event_id(self, todo: Dict[str, Any]) -> Optional[str]:
        linked_events = self._linked_calendar_events(todo)
        if linked_events:
            event_id = str(linked_events[0].get("id") or "").strip()
            return event_id or None
        return None

    def _default_calendar_for_user(self, user: Optional[str]) -> str:
        normalized_user = self.normalize_user(user)
        user_lower = normalized_user.lower().strip()
        preferred = f"{user_lower}_personal"
        if preferred in self._calendar_users:
            return preferred
        for calendar_user in self._calendar_users:
            if str(calendar_user).lower().startswith(f"{user_lower}_"):
                return str(calendar_user)
        return self._default_calendar_user

    def _merge_source_metadata(self, todo: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._source_metadata_dict(todo)
        merged.update(updates)
        return merged

    def refresh_todo_cache_files(self, *, user: Optional[str], limit: int = 500) -> Dict[str, Any]:
        """Write unified and per-source todo cache files for local inspection."""
        normalized_user = self.normalize_user(user)
        todos = self.list_todos(user=normalized_user, include_completed=True, limit=limit)
        todo_dir = self._todo_cache_dir(normalized_user)
        todo_dir.mkdir(parents=True, exist_ok=True)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for todo in todos:
            source = self._slugify_cache_component(str(todo.get("source_type") or "unknown"))
            grouped.setdefault(source, []).append(todo)

        generated_at = self._now_iso()

        def build_payload(source_type: str, source_todos: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "generated_at": generated_at,
                "user_id": normalized_user,
                "source_type": source_type,
                "counts": {
                    "total": len(source_todos),
                    "open": sum(1 for todo in source_todos if not todo.get("completed")),
                    "completed": sum(1 for todo in source_todos if todo.get("completed")),
                },
                "todos": source_todos,
            }

        (todo_dir / "all_todos_cache.json").write_text(
            json.dumps(build_payload("all", todos), indent=2),
            encoding="utf-8",
        )

        expected_names = {"all_todos_cache.json"}
        for source, source_todos in grouped.items():
            filename = f"{source}_todos.json"
            expected_names.add(filename)
            (todo_dir / filename).write_text(
                json.dumps(build_payload(source, source_todos), indent=2),
                encoding="utf-8",
            )

        for existing in todo_dir.glob("*.json"):
            if existing.name not in expected_names:
                existing.unlink(missing_ok=True)

        return {
            "success": True,
            "user_id": normalized_user,
            "directory": str(todo_dir),
            "files_written": sorted(expected_names),
            "todo_count": len(todos),
            "source_count": len(grouped),
        }

    def _refresh_daily_briefing_after_todo_change(
        self,
        user: Optional[str],
        *,
        invalidated_todo_id: Optional[str] = None,
    ) -> None:
        """Kick off a non-blocking cache + briefing refresh after todo mutations."""
        if not self.is_available():
            return

        normalized_user = self.normalize_user(user)
        threading.Thread(
            target=self._refresh_daily_briefing_worker,
            args=(normalized_user, invalidated_todo_id),
            daemon=True,
            name=f"todo-briefing-{normalized_user.lower()}",
        ).start()

    def _refresh_daily_briefing_worker(self, user: str, invalidated_todo_id: Optional[str]) -> None:
        """Worker used by background refresh threads."""
        try:
            self.refresh_todo_cache_files(user=user)
            self.upsert_daily_briefing(user=user, invalidated_todo_id=invalidated_todo_id)
        except Exception as exc:
            logger.warning("Failed to refresh todo post-change sync for %s: %s", user, exc)

    def create_todo(
        self,
        *,
        user: Optional[str],
        title: str,
        details: Optional[str] = None,
        due_at: Optional[str] = None,
        event_time: Optional[str] = None,
        remind_before_minutes: Optional[int] = None,
        source_type: str,
        source_id: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._require_client()
        normalized_user = self.normalize_user(user)
        due_dt = self.parse_due_datetime(
            due_at=due_at,
            event_time=event_time,
            remind_before_minutes=remind_before_minutes,
        )
        normalized_group = self._normalize_todo_group(group)
        normalized_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        if normalized_group:
            normalized_metadata[self.TODO_GROUP_METADATA_KEY] = normalized_group
        now = self._now_iso()

        todo = {
            "id": self._generate_todo_id(),
            "user_id": normalized_user,
            "title": title.strip(),
            "details": (details or "").strip() or None,
            "due_at": due_dt.astimezone(timezone.utc).isoformat() if due_dt else None,
            "completed": False,
            "completed_at": None,
            "source_type": source_type,
            "source_id": source_id or self._generate_source_id(source_type),
            "source_metadata": normalized_metadata,
            "created_at": now,
            "updated_at": now,
        }

        client.table(self.TABLE_NAME).insert(todo).execute()
        formatted = self._format_todo(todo)
        self._refresh_daily_briefing_after_todo_change(normalized_user)
        return formatted

    def list_todos(
        self,
        *,
        user: Optional[str],
        include_completed: bool = False,
        limit: int = 25,
        only_due_today: bool = False,
        only_overdue: bool = False,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        client = self._require_client()
        normalized_user = self.normalize_user(user)

        query = client.table(self.TABLE_NAME).select("*").eq("user_id", normalized_user)
        if not include_completed:
            query = query.eq("completed", False)
        if source_type:
            query = query.eq("source_type", source_type)

        response = query.limit(max(limit * 3, 50)).execute()
        todos = [self._format_todo(row) for row in (response.data or [])]
        todos = self._filter_todos_for_time(todos, only_due_today=only_due_today, only_overdue=only_overdue)
        todos.sort(key=self._todo_sort_key)
        return todos[:limit]

    def complete_todo(self, *, user: Optional[str], todo_id: Optional[str] = None, match: Optional[str] = None) -> Dict[str, Any]:
        return self._set_completed_state(user=user, completed=True, todo_id=todo_id, match=match)

    def reopen_todo(self, *, user: Optional[str], todo_id: Optional[str] = None, match: Optional[str] = None) -> Dict[str, Any]:
        return self._set_completed_state(user=user, completed=False, todo_id=todo_id, match=match)

    def _set_completed_state(
        self,
        *,
        user: Optional[str],
        completed: bool,
        todo_id: Optional[str],
        match: Optional[str],
    ) -> Dict[str, Any]:
        client = self._require_client()
        todo = self.resolve_todo(user=user, todo_id=todo_id, match=match, include_completed=True)
        update = {
            "completed": completed,
            "completed_at": self._now_iso() if completed else None,
            "updated_at": self._now_iso(),
        }
        client.table(self.TABLE_NAME).update(update).eq("id", todo["id"]).execute()
        todo.update(update)
        formatted = self._format_todo(todo)
        self._refresh_daily_briefing_after_todo_change(
            todo.get("user_id") or user,
            invalidated_todo_id=todo.get("id") if completed else None,
        )
        return formatted

    def update_todo(
        self,
        *,
        user: Optional[str],
        todo_id: Optional[str] = None,
        match: Optional[str] = None,
        title: Optional[str] = None,
        details: Optional[str] = None,
        due_at: Optional[str] = None,
        clear_due_at: bool = False,
        group: Optional[str] = None,
        clear_group: bool = False,
    ) -> Dict[str, Any]:
        client = self._require_client()
        todo = self.resolve_todo(user=user, todo_id=todo_id, match=match, include_completed=True)

        update: Dict[str, Any] = {"updated_at": self._now_iso()}
        if title is not None:
            update["title"] = title.strip()
        if details is not None:
            update["details"] = details.strip() or None
        if clear_due_at:
            update["due_at"] = None
        elif due_at is not None:
            parsed_due = self._parse_datetime_string(due_at)
            update["due_at"] = parsed_due.astimezone(timezone.utc).isoformat()
        if group is not None or clear_group:
            metadata = self._source_metadata_dict(todo)
            if clear_group:
                metadata.pop(self.TODO_GROUP_METADATA_KEY, None)
            else:
                normalized_group = self._normalize_todo_group(group)
                if normalized_group:
                    metadata[self.TODO_GROUP_METADATA_KEY] = normalized_group
                else:
                    metadata.pop(self.TODO_GROUP_METADATA_KEY, None)
            update["source_metadata"] = metadata

        client.table(self.TABLE_NAME).update(update).eq("id", todo["id"]).execute()
        todo.update(update)
        formatted = self._format_todo(todo)
        self._refresh_daily_briefing_after_todo_change(todo.get("user_id") or user)
        return formatted

    def delete_todo(
        self,
        *,
        user: Optional[str],
        todo_id: Optional[str] = None,
        match: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._require_client()
        todo = self.resolve_todo(user=user, todo_id=todo_id, match=match, include_completed=True)
        client.table(self.TABLE_NAME).delete().eq("id", todo["id"]).execute()
        formatted = self._format_todo(todo)
        self._refresh_daily_briefing_after_todo_change(
            todo.get("user_id") or user,
            invalidated_todo_id=todo.get("id"),
        )
        return formatted

    def add_todo_to_calendar(
        self,
        *,
        user: Optional[str],
        todo_id: str,
        calendar_user: Optional[str] = None,
        calendar_users: Optional[List[str]] = None,
        attendees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        client = self._require_client()
        todo = self.resolve_todo(user=user, todo_id=todo_id, include_completed=True)

        if str(todo.get("source_type") or "").lower() == "calendar":
            raise PermissionError("Calendar-backed todos are already linked to Google Calendar.")
        if bool(todo.get("completed")):
            raise ValueError("Reopen this todo before adding it to Google Calendar.")

        due_dt = self._parse_due_from_row(todo)
        if due_dt is None:
            raise ValueError("Add a due date before sending this todo to Google Calendar.")

        requested_calendar_users = [
            str(value).strip()
            for value in (calendar_users or [])
            if str(value).strip()
        ]
        if not requested_calendar_users:
            default_calendar = calendar_user or self._default_calendar_for_user(todo.get("user_id") or user)
            requested_calendar_users = [default_calendar]

        deduped_calendar_users: List[str] = []
        for requested in requested_calendar_users:
            if requested not in self._calendar_users:
                raise ValueError(f"Unknown calendar user '{requested}'.")
            if requested not in deduped_calendar_users:
                deduped_calendar_users.append(requested)

        existing_links = self._linked_calendar_events(todo)
        existing_users = {
            str(item.get("calendar_user") or "").strip()
            for item in existing_links
            if str(item.get("calendar_user") or "").strip()
        }
        target_calendar_users = [calendar for calendar in deduped_calendar_users if calendar not in existing_users]
        if not target_calendar_users:
            raise ValueError("This todo is already linked to the selected Google Calendar(s).")

        try:
            from mcp_server.clients.calendar_client import CalendarComponent
        except ImportError as exc:
            raise RuntimeError("Google Calendar integration is not available.") from exc

        due_local = due_dt.astimezone(self._tz)
        end_local = due_local + timedelta(hours=1)
        normalized_attendees = [str(attendee).strip() for attendee in (attendees or []) if str(attendee).strip()]
        new_links: List[Dict[str, Any]] = []

        for target_calendar_user in target_calendar_users:
            calendar = CalendarComponent(user=target_calendar_user)
            created_event = calendar.create_event(
                {
                    "title": str(todo.get("title") or "Untitled todo"),
                    "description": str(todo.get("details") or ""),
                    "date": due_local.date().isoformat(),
                    "start_time": due_local.strftime("%H:%M"),
                    "end_time": end_local.strftime("%H:%M"),
                    "calendar_name": "primary",
                    "time_zone": DEFAULT_TIMEZONE,
                    "attendees": normalized_attendees,
                }
            )

            event_id = str(created_event.get("id") or "").strip()
            if not event_id:
                raise RuntimeError("Google Calendar event creation did not return an event id.")

            new_links.append(
                {
                    "id": event_id,
                    "calendar_user": target_calendar_user,
                    "htmlLink": created_event.get("htmlLink"),
                    "start": created_event.get("start"),
                    "end": created_event.get("end"),
                    "attendees": normalized_attendees,
                }
            )

        merged_links = existing_links + new_links
        primary_link = merged_links[0]

        update = {
            "source_metadata": self._merge_source_metadata(
                todo,
                {
                    "linked_calendar_event_id": primary_link["id"],
                    "linked_calendar_user": primary_link["calendar_user"],
                    "linked_calendar_url": primary_link.get("htmlLink"),
                    "linked_calendar_added_at": self._now_iso(),
                    "linked_calendar_attendees": primary_link.get("attendees", []),
                    "linked_calendar_events": merged_links,
                },
            ),
            "updated_at": self._now_iso(),
        }
        client.table(self.TABLE_NAME).update(update).eq("id", todo["id"]).execute()
        todo.update(update)

        return {
            "todo": self._format_todo(todo),
            "calendar_event": dict(new_links[-1]),
            "calendar_events": [dict(item) for item in new_links],
        }

    def resolve_todo(
        self,
        *,
        user: Optional[str],
        todo_id: Optional[str] = None,
        match: Optional[str] = None,
        include_completed: bool = False,
    ) -> Dict[str, Any]:
        normalized_user = self.normalize_user(user)
        todos = self.list_todos(
            user=normalized_user,
            include_completed=include_completed,
            limit=100,
        )

        if todo_id:
            for todo in todos:
                if todo["id"] == todo_id:
                    return todo
            raise ValueError(f"No todo found with id '{todo_id}'.")

        if not match:
            raise ValueError("Provide either todo_id or match.")

        needle = match.lower().strip()
        exact = [t for t in todos if t["title"].lower() == needle]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(self._ambiguous_match_error(match, exact))

        contains = [t for t in todos if needle in t["title"].lower()]
        if len(contains) == 1:
            return contains[0]
        if len(contains) > 1:
            raise ValueError(self._ambiguous_match_error(match, contains))

        raise ValueError(f"No todo found matching '{match}'.")

    def _ambiguous_match_error(self, match: str, matches: List[Dict[str, Any]]) -> str:
        options = ", ".join(f"{todo['id']}: {todo['title']}" for todo in matches[:5])
        return f"Multiple todos matched '{match}'. Be more specific or use an id. Matches: {options}"

    def upsert_calendar_todo(self, *, calendar_user: str, event: Dict[str, Any]) -> Dict[str, Any]:
        client = self._require_client()
        assistant_user = self._map_calendar_user_to_assistant_user(calendar_user)
        source_id = event.get("id") or event.get("event_id")
        if not source_id:
            raise ValueError("Calendar event is missing an id.")

        title = event.get("summary") or event.get("event_title") or "Untitled event"
        description = event.get("description") or ""
        location = event.get("location") or ""
        due_at = self._calendar_event_due_at(event)
        metadata = {
            "calendar_user": calendar_user,
            "calendar_name": event.get("calendar_name"),
            "event_id": source_id,
            "location": location,
            "description": description,
            "start_date": event.get("start_date"),
            "start_time": event.get("start_time"),
            "end_date": event.get("end_date"),
            "end_time": event.get("end_time"),
            "all_day": event.get("all_day", False),
        }

        linked_manual_response = (
            client.table(self.TABLE_NAME)
            .select("*")
            .eq("user_id", assistant_user)
            .limit(300)
            .execute()
        )
        for row in linked_manual_response.data or []:
            if str(row.get("source_type") or "").lower() == "calendar":
                continue
            if not self._todo_has_linked_calendar_event(row, str(source_id)):
                continue

            linked_events = self._linked_calendar_events(row)
            refreshed_links: List[Dict[str, Any]] = []
            for linked_event in linked_events:
                if str(linked_event.get("id") or "").strip() == str(source_id):
                    refreshed = dict(linked_event)
                    refreshed["calendar_user"] = calendar_user
                    refreshed["linked_calendar_last_synced_at"] = self._now_iso()
                    refreshed["linked_calendar_name"] = event.get("calendar_name")
                    refreshed_links.append(refreshed)
                else:
                    refreshed_links.append(dict(linked_event))
            primary_link = refreshed_links[0] if refreshed_links else None

            update = {
                "source_metadata": self._merge_source_metadata(
                    row,
                    {
                        "linked_calendar_event_id": str(primary_link.get("id") if primary_link else source_id),
                        "linked_calendar_user": str(primary_link.get("calendar_user") if primary_link else calendar_user),
                        "linked_calendar_last_synced_at": self._now_iso(),
                        "linked_calendar_name": event.get("calendar_name"),
                        "linked_calendar_events": refreshed_links,
                    },
                ),
                "updated_at": self._now_iso(),
            }
            client.table(self.TABLE_NAME).update(update).eq("id", row["id"]).execute()
            row.update(update)
            return self._format_todo(row)

        existing_resp = (
            client.table(self.TABLE_NAME)
            .select("*")
            .eq("user_id", assistant_user)
            .eq("source_type", "calendar")
            .eq("source_id", source_id)
            .limit(1)
            .execute()
        )
        existing = (existing_resp.data or [None])[0]
        now = self._now_iso()

        update = {
            "user_id": assistant_user,
            "title": title,
            "details": self._combine_calendar_details(description=description, location=location),
            "due_at": due_at,
            "source_type": "calendar",
            "source_id": source_id,
            "source_metadata": metadata,
            "updated_at": now,
        }

        if existing:
            client.table(self.TABLE_NAME).update(update).eq("id", existing["id"]).execute()
            existing.update(update)
            return self._format_todo(existing)

        record = {
            "id": self._generate_todo_id(),
            "completed": False,
            "completed_at": None,
            "created_at": now,
            **update,
        }
        client.table(self.TABLE_NAME).insert(record).execute()
        return self._format_todo(record)

    def sync_calendar_events(self, *, calendar_user: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        client = self._require_client()
        assistant_user = self._map_calendar_user_to_assistant_user(calendar_user)
        synced = 0
        removed = 0
        failures: List[str] = []
        current_source_ids: set[str] = set()

        for event in events:
            try:
                source_id = event.get("id") or event.get("event_id")
                if source_id:
                    current_source_ids.add(str(source_id))
                self.upsert_calendar_todo(calendar_user=calendar_user, event=event)
                synced += 1
            except Exception as exc:
                failures.append(str(exc))

        try:
            existing_resp = (
                client.table(self.TABLE_NAME)
                .select("id, source_id, source_metadata")
                .eq("user_id", assistant_user)
                .eq("source_type", "calendar")
                .execute()
            )
            existing_rows = existing_resp.data or []
            stale_ids = [
                str(row.get("id"))
                for row in existing_rows
                if str((row.get("source_metadata") or {}).get("calendar_user") or "") == calendar_user
                and str(row.get("source_id") or "") not in current_source_ids
                and row.get("id")
            ]
            for stale_id in stale_ids:
                client.table(self.TABLE_NAME).delete().eq("id", stale_id).execute()
                removed += 1
        except Exception as exc:
            failures.append(f"Failed to prune stale calendar events for {calendar_user}: {exc}")

        try:
            self.refresh_todo_cache_files(user=assistant_user)
        except Exception as exc:
            failures.append(f"Failed to refresh todo cache files for {assistant_user}: {exc}")

        return {
            "success": len(failures) == 0,
            "synced": synced,
            "removed": removed,
            "failed": len(failures),
            "errors": failures[:5],
        }

    def _todo_digest_briefing_id(self, user: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"todo_digest_{user.lower()}_{stamp}_{uuid.uuid4().hex[:8]}"

    def _briefing_content_dict(self, briefing: Dict[str, Any]) -> Dict[str, Any]:
        content = briefing.get("content") or {}
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return {}
        return content if isinstance(content, dict) else {}

    def _is_todo_digest_briefing(self, briefing: Dict[str, Any]) -> bool:
        content = self._briefing_content_dict(briefing)
        meta = content.get("meta") or {}
        return str(meta.get("source") or "") == "todo_digest"

    def _briefing_todo_ids(self, briefing: Dict[str, Any]) -> List[str]:
        content = self._briefing_content_dict(briefing)
        meta = content.get("meta") or {}
        todo_ids = meta.get("todo_ids") or []
        if not isinstance(todo_ids, list):
            return []
        return [str(todo_id) for todo_id in todo_ids]

    def _get_pending_todo_digest_briefings(self, *, user: str) -> List[Dict[str, Any]]:
        client = self._require_client()
        response = (
            client.table(self.BRIEFING_TABLE_NAME)
            .select("*")
            .eq("user_id", user)
            .eq("status", "pending")
            .limit(50)
            .execute()
        )
        return [briefing for briefing in (response.data or []) if self._is_todo_digest_briefing(briefing)]

    def _briefing_key(self, briefing: Dict[str, Any]) -> str:
        content = self._briefing_content_dict(briefing)
        meta = content.get("meta") or {}
        return str(meta.get("briefing_key") or "")

    def _should_include_in_todo_briefings(self, todo: Dict[str, Any]) -> bool:
        source_type = str(todo.get("source_type") or "").lower()
        if source_type in self.TODO_BRIEFING_SOURCE_TYPES:
            return True
        return not source_type or source_type == "unknown"

    def _source_label(self, source_types: List[str]) -> str:
        cleaned = sorted({str(source or "").lower() for source in source_types if source})
        if not cleaned:
            return "todo"
        if len(cleaned) > 1:
            return "open tasks"
        source = cleaned[0]
        return {
            "manual": "todo",
            "discord": "Discord todo",
            "voice": "voice todo",
        }.get(source, f"{source} todo")

    def _timed_briefing_lead(self, due_local: datetime, now_local: datetime) -> timedelta:
        delta = due_local - now_local
        if delta <= timedelta(hours=2):
            return timedelta(minutes=15)
        if delta <= timedelta(hours=8):
            return timedelta(minutes=45)
        if delta <= timedelta(days=1):
            return timedelta(hours=2)
        return timedelta(hours=12)

    def _is_within_quiet_hours(self, local_dt: datetime) -> bool:
        local_time = local_dt.time().replace(tzinfo=None)
        start = self._briefing_quiet_hours_start
        end = self._briefing_quiet_hours_end
        if start == end:
            return False
        if start < end:
            return start <= local_time < end
        return local_time >= start or local_time < end

    def _first_window_at_or_after(
        self,
        base_local: datetime,
        *,
        windows: Optional[List[dt_time]] = None,
    ) -> datetime:
        window_times = windows or self._briefing_windows_local
        if not window_times:
            return base_local

        start_date = base_local.date()
        for day_offset in range(0, 35):
            current_date = start_date + timedelta(days=day_offset)
            for window in window_times:
                candidate = datetime.combine(current_date, window).replace(tzinfo=self._tz)
                if candidate < base_local:
                    continue
                if self._is_within_quiet_hours(candidate):
                    continue
                return candidate
        return base_local

    def _latest_window_before(
        self,
        deadline_local: datetime,
        *,
        not_before_local: datetime,
        windows: Optional[List[dt_time]] = None,
    ) -> Optional[datetime]:
        if deadline_local <= not_before_local:
            return None

        window_times = windows or self._briefing_windows_local
        if not window_times:
            return None

        candidates: List[datetime] = []
        current_date = not_before_local.date()
        end_date = deadline_local.date()
        while current_date <= end_date:
            for window in window_times:
                candidate = datetime.combine(current_date, window).replace(tzinfo=self._tz)
                if candidate < not_before_local or candidate >= deadline_local:
                    continue
                if self._is_within_quiet_hours(candidate):
                    continue
                candidates.append(candidate)
            current_date += timedelta(days=1)

        if not candidates:
            return None
        return max(candidates)

    def _plan_timed_briefing_send(self, *, due_local: datetime, now_local: datetime) -> Dict[str, Any]:
        delta = due_local - now_local
        if delta <= timedelta(minutes=self._briefing_urgent_override_minutes):
            base_local = max(now_local, due_local - timedelta(minutes=15))
            return {
                "base_time": base_local.astimezone(timezone.utc).isoformat(),
                "planned_send_time": base_local.astimezone(timezone.utc).isoformat(),
                "send_reason": "urgent_due_soon",
                "urgent_override_applied": True,
            }

        lead = self._timed_briefing_lead(due_local, now_local)
        base_local = max(now_local, due_local - lead)
        planned_local = self._first_window_at_or_after(base_local)
        send_reason = "timed_due_window_snap"
        if planned_local >= due_local:
            fallback = self._latest_window_before(due_local, not_before_local=now_local)
            if fallback is not None:
                planned_local = fallback
                send_reason = "timed_due_window_before_deadline"
            else:
                planned_local = max(now_local, due_local - timedelta(minutes=15))
                send_reason = "timed_due_fallback_immediate"

        return {
            "base_time": base_local.astimezone(timezone.utc).isoformat(),
            "planned_send_time": planned_local.astimezone(timezone.utc).isoformat(),
            "send_reason": send_reason,
            "urgent_override_applied": False,
        }

    def _plan_overdue_briefing_send(self, *, now_local: datetime) -> Dict[str, Any]:
        planned_local = self._first_window_at_or_after(now_local)
        return {
            "base_time": now_local.astimezone(timezone.utc).isoformat(),
            "planned_send_time": planned_local.astimezone(timezone.utc).isoformat(),
            "send_reason": "overdue_next_window",
            "urgent_override_applied": False,
        }

    def _plan_undated_briefing_send(self, *, now_local: datetime) -> Dict[str, Any]:
        morning_window = self._briefing_windows_local[0] if self._briefing_windows_local else dt_time(hour=8, minute=30)
        planned_local = self._first_window_at_or_after(now_local, windows=[morning_window])
        return {
            "base_time": now_local.astimezone(timezone.utc).isoformat(),
            "planned_send_time": planned_local.astimezone(timezone.utc).isoformat(),
            "send_reason": "undated_morning_window",
            "urgent_override_applied": False,
        }

    def _urgency_bucket_for_timed_due(self, *, due_local: datetime, now_local: datetime) -> str:
        delta = due_local - now_local
        if delta <= timedelta(0):
            return "overdue"
        if delta <= timedelta(minutes=self._briefing_urgent_override_minutes):
            return "urgent"
        if delta <= timedelta(hours=8):
            return "soon"
        return "normal"

    def _compose_todo_briefing_message(
        self,
        *,
        kind: str,
        todos: List[Dict[str, Any]],
        total_count: int,
    ) -> Tuple[str, str]:
        title_list = self._format_title_list(todos)
        if kind == "overdue":
            suggested_action = "Choose one overdue task to tackle next, and I can update it."
            message = (
                f"Quick check-in: you have {total_count} overdue todo{'s' if total_count != 1 else ''}, "
                f"including {title_list}. {suggested_action}"
            )
            return message, suggested_action

        if kind == "undated":
            suggested_action = "Pick one and add a due time so I can remind you at the right moment."
            message = (
                f"Quick check-in: you have {total_count} open todo{'s' if total_count != 1 else ''} without due times, "
                f"including {title_list}. {suggested_action}"
            )
            return message, suggested_action

        # timed_due
        title = str(todos[0].get("title") or "Untitled task").strip()
        suggested_action = "Want to start it now or move the due time?"
        message = f"Quick heads up: '{title}' is due in {{{{TIME_UNTIL_DUE}}}}. {suggested_action}"
        return message, suggested_action

    def _undated_briefing_recently_generated(self, *, user: str) -> bool:
        if not self.is_available():
            return False

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.TODO_UNDATED_BRIEFING_COOLDOWN_HOURS)
        client = self._require_client()
        try:
            response = (
                client.table(self.BRIEFING_TABLE_NAME)
                .select("id, content, created_at")
                .eq("user_id", user)
                .order("created_at", desc=True)
                .limit(200)
                .execute()
            )
        except Exception:
            return False

        for row in response.data or []:
            content = self._briefing_content_dict(row)
            meta = content.get("meta") or {}
            if str(meta.get("source") or "") != "todo_digest":
                continue
            if str(meta.get("briefing_kind") or "") != "undated":
                continue

            created_at = row.get("created_at")
            created_dt: Optional[datetime] = None
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                except ValueError:
                    created_dt = None
            if created_dt is None:
                generated_on = meta.get("generated_on")
                try:
                    created_dt = datetime.fromisoformat(str(generated_on).replace("Z", "+00:00"))
                except ValueError:
                    created_dt = None

            if created_dt is None:
                return True

            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            else:
                created_dt = created_dt.astimezone(timezone.utc)

            if created_dt >= cutoff:
                return True
            return False

        return False

    def _build_timed_todo_briefing(self, todo: Dict[str, Any], due_local: datetime, now_local: datetime) -> Dict[str, Any]:
        source_type = str(todo.get("source_type") or "manual").lower()
        planner = self._plan_timed_briefing_send(due_local=due_local, now_local=now_local)
        message, suggested_action = self._compose_todo_briefing_message(
            kind="timed_due",
            todos=[todo],
            total_count=1,
        )
        priority = "high" if planner["urgent_override_applied"] else "normal"
        return {
            "briefing_key": f"todo_due_{todo['id']}",
            "message": message,
            "todo_ids": [str(todo["id"])],
            "active_from": planner["planned_send_time"],
            "priority": priority,
            "kind": "timed_due",
            "source_types": [source_type],
            "urgency_bucket": self._urgency_bucket_for_timed_due(due_local=due_local, now_local=now_local),
            "due_at_iso": due_local.astimezone(timezone.utc).isoformat(),
            "suggested_action": suggested_action,
            "planner": planner,
        }

    def _build_grouped_todo_briefing(
        self,
        *,
        kind: str,
        todos: List[Dict[str, Any]],
        priority: str,
        urgency_bucket: str,
        suggested_action: str,
        planner: Dict[str, Any],
        message: str,
        due_at_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_types = sorted({str(todo.get("source_type") or "unknown").lower() for todo in todos})
        return {
            "briefing_key": f"{kind}:" + ",".join(sorted(str(todo["id"]) for todo in todos)),
            "message": message,
            "todo_ids": [str(todo["id"]) for todo in todos],
            "active_from": planner["planned_send_time"],
            "priority": priority,
            "kind": kind,
            "source_types": source_types,
            "urgency_bucket": urgency_bucket,
            "due_at_iso": due_at_iso,
            "suggested_action": suggested_action,
            "planner": planner,
        }

    def build_todo_briefing_summaries(self, *, user: Optional[str], max_items: int = 12) -> List[Dict[str, Any]]:
        normalized_user = self.normalize_user(user)
        todos = [
            todo
            for todo in self.list_todos(user=normalized_user, include_completed=False, limit=max_items + 50)
            if self._should_include_in_todo_briefings(todo)
        ]
        if not todos:
            return []

        now_local = datetime.now(self._tz)
        overdue: List[Dict[str, Any]] = []
        undated: List[Dict[str, Any]] = []
        timed: List[Tuple[datetime, Dict[str, Any]]] = []

        for todo in todos:
            due_dt = self._parse_due_from_row(todo)
            if due_dt is None:
                undated.append(todo)
                continue
            due_local = due_dt.astimezone(self._tz)
            if due_local <= now_local:
                overdue.append(todo)
            else:
                timed.append((due_local, todo))

        timed.sort(key=lambda item: item[0])
        summaries: List[Dict[str, Any]] = []

        if overdue:
            selected = overdue[:3]
            count = len(overdue)
            message, suggested_action = self._compose_todo_briefing_message(
                kind="overdue",
                todos=selected,
                total_count=count,
            )
            planner = self._plan_overdue_briefing_send(now_local=now_local)
            summaries.append(
                self._build_grouped_todo_briefing(
                    kind="overdue",
                    todos=selected,
                    message=message,
                    priority="high",
                    urgency_bucket="overdue",
                    suggested_action=suggested_action,
                    planner=planner,
                )
            )

        for due_local, todo in timed[: self.MAX_TIMED_TODO_BRIEFINGS]:
            summaries.append(self._build_timed_todo_briefing(todo, due_local, now_local))

        if undated and not self._undated_briefing_recently_generated(user=normalized_user):
            selected = undated[:3]
            count = len(undated)
            message, suggested_action = self._compose_todo_briefing_message(
                kind="undated",
                todos=selected,
                total_count=count,
            )
            planner = self._plan_undated_briefing_send(now_local=now_local)
            summaries.append(
                self._build_grouped_todo_briefing(
                    kind="undated",
                    todos=selected,
                    message=message,
                    priority="low",
                    urgency_bucket="normal",
                    suggested_action=suggested_action,
                    planner=planner,
                )
            )

        summaries.sort(key=lambda summary: (summary["active_from"], summary["priority"] != "high"))
        return summaries

    def _mark_briefings_skipped(self, briefing_ids: List[str]) -> int:
        if not briefing_ids:
            return 0

        client = self._require_client()
        now = self._now_iso()
        skipped = 0
        for briefing_id in briefing_ids:
            client.table(self.BRIEFING_TABLE_NAME).update(
                {
                    "status": "skipped",
                    "dismissed_at": now,
                }
            ).eq("id", briefing_id).eq("status", "pending").execute()
            skipped += 1
        return skipped

    def _todo_llm_instructions_for_kind(self, *, kind: str) -> str:
        if kind == "timed_due":
            return "Deliver this as a brief, conversational heads-up and include one concrete next step."
        if kind == "overdue":
            return "Mention overdue items naturally, keep it short, and suggest the single most helpful next action."
        if kind == "undated":
            return "Briefly surface undated tasks and prompt the user to assign one due time."
        return "Share the user's open todos naturally and only when they become time-relevant."

    def upsert_daily_briefing(
        self,
        *,
        user: Optional[str],
        max_items: int = 5,
        invalidated_todo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._require_client()
        normalized_user = self.normalize_user(user)
        pending_digests = self._get_pending_todo_digest_briefings(user=normalized_user)
        planned_summaries = self.build_todo_briefing_summaries(user=normalized_user, max_items=max_items)

        if not planned_summaries:
            skipped = self._mark_briefings_skipped([str(briefing.get("id")) for briefing in pending_digests if briefing.get("id")])
            return {"success": True, "briefing_created": False, "reason": "no_open_todos", "briefings_skipped": skipped}
        planned_by_key = {summary["briefing_key"]: summary for summary in planned_summaries}
        existing_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for briefing in pending_digests:
            existing_by_key.setdefault(self._briefing_key(briefing), []).append(briefing)

        stale_ids: List[str] = []
        created_ids: List[str] = []

        for key, briefings in existing_by_key.items():
            planned = planned_by_key.get(key)
            matching_existing = None
            for briefing in briefings:
                if planned is None:
                    stale_ids.append(str(briefing.get("id") or ""))
                    continue
                existing_content = self._briefing_content_dict(briefing)
                existing_meta = existing_content.get("meta") or {}
                if (
                    briefing.get("opener_text") == planned["message"]
                    and self._briefing_todo_ids(briefing) == planned["todo_ids"]
                    and existing_content.get("active_from") == planned["active_from"]
                    and str(briefing.get("priority") or "normal") == planned["priority"]
                    and str(existing_meta.get("urgency_bucket") or "") == str(planned.get("urgency_bucket") or "")
                    and str(existing_meta.get("due_at_iso") or "") == str(planned.get("due_at_iso") or "")
                    and str(existing_meta.get("suggested_action") or "") == str(planned.get("suggested_action") or "")
                    and str(existing_meta.get("base_time") or "") == str((planned.get("planner") or {}).get("base_time") or "")
                    and str(existing_meta.get("planned_send_time") or "") == str((planned.get("planner") or {}).get("planned_send_time") or "")
                    and str(existing_meta.get("send_reason") or "") == str((planned.get("planner") or {}).get("send_reason") or "")
                    and bool(existing_meta.get("urgent_override_applied")) == bool((planned.get("planner") or {}).get("urgent_override_applied"))
                ):
                    matching_existing = briefing
                else:
                    stale_ids.append(str(briefing.get("id") or ""))

            if matching_existing is not None:
                planned_by_key.pop(key, None)
            elif planned is not None and invalidated_todo_id:
                # If the invalidated todo wasn't part of an existing briefing with this key,
                # leave it alone unless content actually changed.
                pass

        skipped = self._mark_briefings_skipped([briefing_id for briefing_id in stale_ids if briefing_id])

        for summary in planned_by_key.values():
            briefing_id = self._todo_digest_briefing_id(normalized_user)
            planner_meta = summary.get("planner") or {}
            record = {
                "id": briefing_id,
                "user_id": normalized_user,
                "content": {
                    "message": summary["message"],
                    "llm_instructions": self._todo_llm_instructions_for_kind(kind=summary["kind"]),
                    "active_from": summary["active_from"],
                    "meta": {
                        "source": "todo_digest",
                        "briefing_key": summary["briefing_key"],
                        "briefing_kind": summary["kind"],
                        "source_types": summary["source_types"],
                        "urgency_bucket": summary.get("urgency_bucket"),
                        "due_at_iso": summary.get("due_at_iso"),
                        "todo_ids": summary["todo_ids"],
                        "suggested_action": summary.get("suggested_action"),
                        "generated_on": datetime.now(timezone.utc).isoformat(),
                        "base_time": planner_meta.get("base_time"),
                        "planned_send_time": planner_meta.get("planned_send_time"),
                        "send_reason": planner_meta.get("send_reason"),
                        "urgent_override_applied": bool(planner_meta.get("urgent_override_applied")),
                    },
                },
                "opener_text": summary["message"],
                "priority": summary["priority"],
                "status": "pending",
                "discord_status": "pending",
                "discord_sent_at": None,
                "voice_status": "pending",
                "voice_read_at": None,
                "dismissed_at": None,
            }
            client.table(self.BRIEFING_TABLE_NAME).insert(record).execute()
            created_ids.append(briefing_id)

        return {
            "success": True,
            "briefing_created": bool(created_ids),
            "briefing_ids": created_ids,
            "briefings_skipped": skipped,
            "planned_count": len(planned_summaries),
        }

    def build_daily_briefing_summary(self, *, user: Optional[str], max_items: int = 5) -> Optional[Dict[str, Any]]:
        todos = self.list_todos(user=user, include_completed=False, limit=max_items + 10)
        if not todos:
            return None

        now_local = datetime.now(self._tz)
        overdue: List[Dict[str, Any]] = []
        due_today: List[Dict[str, Any]] = []
        upcoming: List[Dict[str, Any]] = []
        undated: List[Dict[str, Any]] = []

        for todo in todos:
            due_dt = self._parse_due_from_row(todo)
            if due_dt is None:
                undated.append(todo)
                continue
            due_local = due_dt.astimezone(self._tz)
            if due_local.date() < now_local.date():
                overdue.append(todo)
            elif due_local.date() == now_local.date():
                due_today.append(todo)
            else:
                upcoming.append(todo)

        parts: List[str] = []
        shown_ids: List[str] = []

        if overdue:
            selected = overdue[:2]
            shown_ids.extend(todo["id"] for todo in selected)
            parts.append(f"Overdue: {self._format_title_list(selected)}.")

        if due_today:
            selected = due_today[:3]
            shown_ids.extend(todo["id"] for todo in selected)
            parts.append(f"Due today: {self._format_title_list(selected)}.")

        if upcoming and len(shown_ids) < max_items:
            remaining = max_items - len(shown_ids)
            selected = upcoming[:remaining]
            shown_ids.extend(todo["id"] for todo in selected)
            parts.append(f"Coming up: {self._format_title_list(selected)}.")

        if undated and not parts:
            selected = undated[: min(3, max_items)]
            shown_ids.extend(todo["id"] for todo in selected)
            parts.append(f"You still have open tasks like {self._format_title_list(selected)}.")

        if not parts:
            return None

        count = len(todos)
        message = f"You have {count} incomplete todo{'s' if count != 1 else ''}. " + " ".join(parts)
        return {
            "message": message.strip(),
            "todo_ids": shown_ids,
            "active_from": datetime.now(timezone.utc).isoformat(),
        }

    def _calendar_event_due_at(self, event: Dict[str, Any]) -> Optional[str]:
        start_date = event.get("start_date")
        start_time = event.get("start_time")
        if not start_date:
            return None

        if not start_time or str(start_time).lower() == "all day":
            dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=self._tz, hour=9, minute=0)
            return dt.astimezone(timezone.utc).isoformat()

        time_value = str(start_time)
        fmt = "%H:%M:%S" if time_value.count(":") == 2 else "%H:%M"
        dt = datetime.strptime(f"{start_date} {time_value}", f"%Y-%m-%d {fmt}").replace(tzinfo=self._tz)
        return dt.astimezone(timezone.utc).isoformat()

    def _combine_calendar_details(self, *, description: str, location: str) -> Optional[str]:
        parts = []
        if description:
            parts.append(description.strip())
        if location:
            parts.append(f"Location: {location.strip()}")
        return "\n".join(parts) or None

    def _map_calendar_user_to_assistant_user(self, calendar_user: str) -> str:
        lower = (calendar_user or "").lower()
        if "morgan" in lower:
            return "Morgan"
        if "spencer" in lower:
            return "Spencer"
        return self._default_user

    def _filter_todos_for_time(
        self,
        todos: List[Dict[str, Any]],
        *,
        only_due_today: bool,
        only_overdue: bool,
    ) -> List[Dict[str, Any]]:
        if not only_due_today and not only_overdue:
            return todos

        today = datetime.now(self._tz).date()
        filtered = []
        for todo in todos:
            due_dt = self._parse_due_from_row(todo)
            if due_dt is None:
                continue
            due_date = due_dt.astimezone(self._tz).date()
            if only_due_today and due_date == today:
                filtered.append(todo)
            elif only_overdue and due_date < today:
                filtered.append(todo)
        return filtered

    def _todo_sort_key(self, todo: Dict[str, Any]) -> Tuple[int, datetime, str]:
        due_dt = self._parse_due_from_row(todo)
        if due_dt is None:
            due_dt = datetime.max.replace(tzinfo=timezone.utc)
        completed_rank = 1 if todo.get("completed") else 0
        return (completed_rank, due_dt, todo.get("created_at", ""))

    def _parse_due_from_row(self, todo: Dict[str, Any]) -> Optional[datetime]:
        due_at = todo.get("due_at")
        if not due_at:
            return None
        if isinstance(due_at, datetime):
            return due_at
        try:
            return datetime.fromisoformat(str(due_at).replace("Z", "+00:00"))
        except ValueError:
            return None

    def _format_title_list(self, todos: List[Dict[str, Any]]) -> str:
        titles = [todo["title"] for todo in todos]
        if len(titles) == 1:
            return titles[0]
        if len(titles) == 2:
            return f"{titles[0]} and {titles[1]}"
        return ", ".join(titles[:-1]) + f", and {titles[-1]}"

    def advance_todo(
        self,
        *,
        user: Optional[str],
        todo_id: Optional[str] = None,
        match: Optional[str] = None,
        title: str,
        details: Optional[str] = None,
        due_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete the current todo and create a new one as the next step in a chain."""
        client = self._require_client()
        current = self.resolve_todo(user=user, todo_id=todo_id, match=match, include_completed=False)

        current_metadata = self._source_metadata_dict(current)
        chain_id = current_metadata.get("chain_id") or current["id"]
        chain_position = int(current_metadata.get("chain_position") or 1)
        current_group = self._todo_group_from_row(current)

        complete_update = {
            "completed": True,
            "completed_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "source_metadata": self._merge_source_metadata(current, {
                "chain_id": chain_id,
                "chain_position": chain_position,
                "advanced_to": None,
            }),
        }
        client.table(self.TABLE_NAME).update(complete_update).eq("id", current["id"]).execute()

        normalized_user = self.normalize_user(user)
        due_dt = self.parse_due_datetime(due_at=due_at) if due_at else None
        now = self._now_iso()
        new_id = self._generate_todo_id()

        new_todo = {
            "id": new_id,
            "user_id": normalized_user,
            "title": title.strip(),
            "details": (details or "").strip() or None,
            "due_at": due_dt.astimezone(timezone.utc).isoformat() if due_dt else None,
            "completed": False,
            "completed_at": None,
            "source_type": current.get("source_type") or "manual",
            "source_id": current.get("source_id") or self._generate_source_id("chain"),
            "source_metadata": {
                **(current_metadata.get("created_via") and {"created_via": current_metadata["created_via"]} or {}),
                **(current_group and {self.TODO_GROUP_METADATA_KEY: current_group} or {}),
                "chain_id": chain_id,
                "chain_parent_id": current["id"],
                "chain_position": chain_position + 1,
            },
            "created_at": now,
            "updated_at": now,
        }

        client.table(self.TABLE_NAME).insert(new_todo).execute()

        advanced_update = {
            "source_metadata": self._merge_source_metadata(current, {
                "chain_id": chain_id,
                "chain_position": chain_position,
                "advanced_to": new_id,
            }),
        }
        client.table(self.TABLE_NAME).update(advanced_update).eq("id", current["id"]).execute()

        formatted = self._format_todo(new_todo)
        self._refresh_daily_briefing_after_todo_change(normalized_user, invalidated_todo_id=current["id"])
        return {
            "previous": self._format_todo({**current, **complete_update}),
            "current": formatted,
            "chain_id": chain_id,
            "chain_position": chain_position + 1,
        }

    def get_chain(
        self,
        *,
        user: Optional[str],
        todo_id: str,
    ) -> List[Dict[str, Any]]:
        """Return all todos in a chain, ordered by position."""
        client = self._require_client()
        normalized_user = self.normalize_user(user)

        anchor = self.resolve_todo(user=user, todo_id=todo_id, include_completed=True)
        anchor_metadata = self._source_metadata_dict(anchor)
        chain_id = anchor_metadata.get("chain_id") or anchor["id"]

        response = (
            client.table(self.TABLE_NAME)
            .select("*")
            .eq("user_id", normalized_user)
            .limit(200)
            .execute()
        )

        chain_todos = []
        for row in response.data or []:
            row_metadata = self._source_metadata_dict(row)
            row_chain_id = row_metadata.get("chain_id")
            if row_chain_id == chain_id or row["id"] == chain_id:
                chain_todos.append(self._format_todo(row))

        chain_todos.sort(key=lambda t: int(self._source_metadata_dict(t).get("chain_position") or 1))
        return chain_todos

    def _format_todo(self, todo: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(todo)
        due_dt = self._parse_due_from_row(row)
        if due_dt:
            row["due_at"] = due_dt.astimezone(timezone.utc).isoformat()
            row["due_display"] = due_dt.astimezone(self._tz).strftime("%Y-%m-%d %I:%M %p %Z")
        else:
            row["due_display"] = None
        row["group"] = self._todo_group_from_row(row)
        return row
