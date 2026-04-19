from datetime import datetime, timedelta, timezone

from assistant_framework.utils.briefing.briefing_manager import (
    BriefingManager,
    DISCORD_DELIVERY,
    VOICE_DELIVERY,
    _get_delivery_status,
)
from discord_bot.bot import _extract_briefing_text
from mcp_server.tools.briefing import BriefingTool
from scripts.scheduled.calendar_briefing.briefing_creator import BriefingCreator


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTableQuery:
    def __init__(self, client, op):
        self._client = client
        self._op = op
        self._payload = None
        self._filters = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, field, value):
        self._filters.append((field, value))
        return self

    def like(self, field, value):
        self._filters.append((field, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def execute(self):
        if self._op == "select":
            rows = list(self._client.select_data)
            for field, value in self._filters:
                if field == "id" and isinstance(value, str) and value.endswith("%"):
                    prefix = value[:-1]
                    rows = [row for row in rows if str(row.get("id", "")).startswith(prefix)]
                else:
                    rows = [row for row in rows if row.get(field) == value]
            return _FakeResponse(rows)
        if self._op == "update":
            self._client.updates.append((self._payload, list(self._filters)))
            return _FakeResponse([])
        raise AssertionError(f"Unsupported operation: {self._op}")


class _FakeSupabaseClient:
    def __init__(self, select_data):
        self.select_data = select_data
        self.updates = []

    def table(self, _name):
        return _FakeTableQuery(self, "select")


def test_legacy_delivery_status_keeps_discord_pending():
    legacy_row = {
        "status": "delivered",
        "delivered_at": "2026-03-22T10:00:00+00:00",
    }

    assert _get_delivery_status(legacy_row, DISCORD_DELIVERY) == "pending"
    assert _get_delivery_status(legacy_row, VOICE_DELIVERY) == "read"


def test_extract_briefing_text_prefers_opener_then_message_then_fact():
    opener, source = _extract_briefing_text(
        {
            "opener_text": "Prebuilt opener",
            "content": {"message": "Ignored", "meta": {"source": "calendar"}},
        }
    )
    assert opener == "Prebuilt opener"
    assert source == ""

    message, source = _extract_briefing_text(
        {"content": {"message": "Body message", "meta": {"source": "calendar"}}}
    )
    assert message == "Body message"
    assert source == "calendar"

    fact, source = _extract_briefing_text(
        {"content": {"fact": "Legacy fact", "meta": {"source": "weather"}}}
    )
    assert fact == "Legacy fact"
    assert source == "weather"


def test_get_combined_opener_substitutes_time_until_due_placeholder():
    manager = BriefingManager()
    due_at = (datetime.now(timezone.utc) + timedelta(minutes=90)).isoformat()

    combined = manager.get_combined_opener(
        [
            {
                "opener_text": "Quick reminder: this task is due in {{TIME_UNTIL_DUE}}.",
                "content": {
                    "meta": {
                        "source": "todo_digest",
                        "due_at_iso": due_at,
                    }
                },
            }
        ]
    )

    assert combined is not None
    assert "{{TIME_UNTIL_DUE}}" not in combined
    assert "hour" in combined or "minute" in combined


def test_briefing_tool_lists_legacy_rows_as_discord_pending():
    tool = BriefingTool()
    tool._supabase_available = True
    tool._supabase_client = _FakeSupabaseClient(
        [
            {
                "id": "legacy-1",
                "user_id": "Morgan",
                "content": {"message": "Legacy row", "meta": {"source": "calendar"}},
                "priority": "normal",
                "status": "pending",
                "created_at": "2026-03-22T10:00:00+00:00",
                "delivered_at": "2026-03-22T10:05:00+00:00",
                "discord_status": None,
                "voice_status": None,
            }
        ]
    )

    result = tool._list_briefings({"limit": 10}, user="Morgan")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["briefings"][0]["discord_status"] == "pending"
    assert result["briefings"][0]["voice_status"] == "read"


def test_calendar_expiry_uses_event_time_not_active_from():
    now = datetime.now(timezone.utc)
    reminder_rows = [
        {
            "id": "calendar_future_event",
            "user_id": "Morgan",
            "status": "pending",
            "content": {
                "active_from": (now - timedelta(hours=2)).isoformat(),
                "meta": {
                    "event_datetime_iso": (now + timedelta(hours=1)).isoformat(),
                },
            },
            "discord_status": "pending",
            "voice_status": "pending",
        },
        {
            "id": "calendar_past_event",
            "user_id": "Morgan",
            "status": "pending",
            "content": {
                "active_from": (now - timedelta(hours=4)).isoformat(),
                "meta": {
                    "event_datetime_iso": (now - timedelta(minutes=10)).isoformat(),
                },
            },
            "discord_status": "pending",
            "voice_status": "pending",
        },
    ]

    creator = object.__new__(BriefingCreator)
    creator._initialized = True
    creator._client = _FakeSupabaseClient(reminder_rows)

    expired_count = creator._mark_expired_reminders()

    assert expired_count == 1
    assert len(creator._client.updates) == 1
    payload, filters = creator._client.updates[0]
    assert payload == {"status": "expired"}
    assert ("id", "calendar_past_event") in filters
