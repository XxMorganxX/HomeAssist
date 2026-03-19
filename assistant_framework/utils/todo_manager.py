"""
Supabase-backed todo manager shared across MCP tools, scheduled jobs, and Discord.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone, timedelta
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

from mcp_server.user_config import get_notification_users, get_default_notification_user


DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIME_ZONE", "America/New_York")
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

    def __init__(self):
        self._tz = ZoneInfo(DEFAULT_TIMEZONE)
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
    ) -> Dict[str, Any]:
        client = self._require_client()
        normalized_user = self.normalize_user(user)
        due_dt = self.parse_due_datetime(
            due_at=due_at,
            event_time=event_time,
            remind_before_minutes=remind_before_minutes,
        )
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
            "source_metadata": source_metadata or {},
            "created_at": now,
            "updated_at": now,
        }

        client.table(self.TABLE_NAME).insert(todo).execute()
        return self._format_todo(todo)

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
        return self._format_todo(todo)

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

        client.table(self.TABLE_NAME).update(update).eq("id", todo["id"]).execute()
        todo.update(update)
        return self._format_todo(todo)

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
        return self._format_todo(todo)

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
        synced = 0
        failures: List[str] = []
        for event in events:
            try:
                self.upsert_calendar_todo(calendar_user=calendar_user, event=event)
                synced += 1
            except Exception as exc:
                failures.append(str(exc))
        return {
            "success": len(failures) == 0,
            "synced": synced,
            "failed": len(failures),
            "errors": failures[:5],
        }

    def upsert_daily_briefing(self, *, user: Optional[str], max_items: int = 5) -> Dict[str, Any]:
        client = self._require_client()
        normalized_user = self.normalize_user(user)
        summary = self.build_daily_briefing_summary(user=normalized_user, max_items=max_items)
        briefing_id = self._daily_briefing_id(normalized_user)

        if summary is None:
            client.table(self.BRIEFING_TABLE_NAME).update({
                "status": "dismissed",
                "dismissed_at": self._now_iso(),
            }).eq("id", briefing_id).eq("status", "pending").execute()
            return {"success": True, "briefing_created": False, "reason": "no_open_todos"}

        record = {
            "id": briefing_id,
            "user_id": normalized_user,
            "content": {
                "message": summary["message"],
                "llm_instructions": "Share the user's open todos naturally as part of the autonomous daily briefing.",
                "active_from": summary["active_from"],
                "meta": {
                    "source": "todo_digest",
                    "todo_ids": summary["todo_ids"],
                    "generated_on": datetime.now(self._tz).date().isoformat(),
                },
            },
            "opener_text": summary["message"],
            "priority": "normal",
            "status": "pending",
        }
        client.table(self.BRIEFING_TABLE_NAME).upsert(record).execute()
        return {"success": True, "briefing_created": True, "briefing_id": briefing_id}

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

    def _daily_briefing_id(self, user: str) -> str:
        day = datetime.now(self._tz).date().isoformat()
        return f"todo_digest_{user.lower()}_{day}"

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

    def _format_todo(self, todo: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(todo)
        due_dt = self._parse_due_from_row(row)
        if due_dt:
            row["due_at"] = due_dt.astimezone(timezone.utc).isoformat()
            row["due_display"] = due_dt.astimezone(self._tz).strftime("%Y-%m-%d %I:%M %p %Z")
        else:
            row["due_display"] = None
        return row
