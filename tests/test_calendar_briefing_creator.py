from copy import deepcopy

from scripts.scheduled.calendar_briefing.briefing_creator import BriefingCreator, EventCache


def _sample_calendar_event(**overrides):
    base = {
        "event_id": "evt_123",
        "event_title": "Team sync",
        "event_date": "2026-04-25",
        "event_time": "14:00:00",
        "priority": "medium",
        "reminders_minutes_before": [30, 10],
    }
    base.update(overrides)
    return base


def test_calendar_briefing_ids_are_deterministic_across_runs():
    creator = BriefingCreator(generate_openers=False)
    suggestions = {"morgan_personal": [_sample_calendar_event()]}

    first = creator.create_briefings_from_suggestions(suggestions)
    second = creator.create_briefings_from_suggestions(suggestions)

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert all(item["id"].startswith("calendar_") for item in first)


def test_calendar_briefing_content_updates_without_changing_id():
    creator = BriefingCreator(generate_openers=False)
    before = creator.create_briefings_from_suggestions(
        {"morgan_personal": [_sample_calendar_event(event_title="Team sync")]}
    )
    after = creator.create_briefings_from_suggestions(
        {"morgan_personal": [_sample_calendar_event(event_title="Team sync (updated)")]}
    )

    assert [item["id"] for item in before] == [item["id"] for item in after]
    assert before[0]["content"]["message"] != after[0]["content"]["message"]
    assert (
        before[0]["content"]["meta"]["event_fingerprint"]
        != after[0]["content"]["meta"]["event_fingerprint"]
    )


def test_event_cache_uses_fingerprint_not_just_event_id():
    cache = object.__new__(EventCache)
    cache._cache = {}
    cache._use_supabase = False

    event = {
        "id": "evt_123",
        "summary": "Math class",
        "start_date": "2026-04-25",
        "start_time": "09:00:00",
        "reminders_minutes_before": [30],
    }
    first = cache.filter_unseen_events([event], namespace="morgan_personal")
    assert len(first) == 1
    fingerprint = first[0]["__event_fingerprint"]
    cache.mark_seen_batch([fingerprint])

    second = cache.filter_unseen_events([deepcopy(event)], namespace="morgan_personal")
    assert second == []

    changed = deepcopy(event)
    changed["start_time"] = "10:00:00"
    third = cache.filter_unseen_events([changed], namespace="morgan_personal")
    assert len(third) == 1


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

    def update(self, payload):
        self._payload = payload
        self._op = "update"
        return self

    def execute(self):
        if self._op == "select":
            rows = list(self._client.rows)
            for field, value in self._filters:
                if field == "id" and isinstance(value, str) and value.endswith("%"):
                    rows = [row for row in rows if str(row.get("id", "")).startswith(value[:-1])]
                else:
                    rows = [row for row in rows if row.get(field) == value]
            return _FakeResponse(rows)

        if self._op == "update":
            self._client.updates.append((self._payload, list(self._filters)))
            return _FakeResponse([])

        raise AssertionError(f"Unsupported op: {self._op}")


class _FakeSupabaseClient:
    def __init__(self, rows):
        self.rows = rows
        self.updates = []

    def table(self, _name):
        return _FakeTableQuery(self, "select")


def test_mark_stale_pending_event_reminders_skips_removed_offsets():
    rows = [
        {
            "id": "calendar_keep",
            "user_id": "Morgan",
            "status": "pending",
            "discord_status": "pending",
            "voice_status": "pending",
            "content": {
                "meta": {
                    "source": "calendar_briefing",
                    "calendar_user": "morgan_personal",
                    "event_id": "evt_123",
                }
            },
        },
        {
            "id": "calendar_stale",
            "user_id": "Morgan",
            "status": "pending",
            "discord_status": "pending",
            "voice_status": "pending",
            "content": {
                "meta": {
                    "source": "calendar_briefing",
                    "calendar_user": "morgan_personal",
                    "event_id": "evt_123",
                }
            },
        },
        {
            "id": "calendar_other",
            "user_id": "Morgan",
            "status": "pending",
            "discord_status": "pending",
            "voice_status": "pending",
            "content": {
                "meta": {
                    "source": "calendar_briefing",
                    "calendar_user": "morgan_personal",
                    "event_id": "evt_999",
                }
            },
        },
    ]

    creator = object.__new__(BriefingCreator)
    creator._initialized = True
    creator._client = _FakeSupabaseClient(rows)

    skipped = creator._mark_stale_pending_event_reminders(
        {"Morgan|morgan_personal|evt_123": {"calendar_keep"}}
    )

    assert skipped == 1
    assert len(creator._client.updates) == 1
    payload, filters = creator._client.updates[0]
    assert payload["status"] == "skipped"
    assert ("id", "calendar_stale") in filters
