from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import types


def _load_todo_manager_module():
    module_path = Path(__file__).parent.parent / "assistant_framework" / "utils" / "todo_manager.py"
    spec = importlib.util.spec_from_file_location("todo_manager_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_todo_manager_class():
    return _load_todo_manager_module().TodoManager


def test_normalize_datetime_string_expands_common_shorthand(mock_user_config_testuser):
    manager = _load_todo_manager_class()()

    normalized = manager._normalize_datetime_string("tmrw 5 pm")

    assert normalized == "tomorrow 5pm"


def test_parse_due_datetime_accepts_tmrw_shorthand(mock_user_config_testuser):
    manager = _load_todo_manager_class()()

    parsed = manager.parse_due_datetime(due_at="tmrw 5pm")

    assert parsed is not None
    expected_date = (datetime.now(manager._tz) + timedelta(days=1)).date()
    assert parsed.astimezone(manager._tz).date() == expected_date
    assert parsed.astimezone(manager._tz).hour == 17


def test_format_todo_includes_normalized_group_from_metadata(mock_user_config_testuser):
    manager = _load_todo_manager_class()()

    formatted = manager._format_todo(
        {
            "id": "todo_grouped",
            "title": "Ship release notes",
            "source_metadata": {"todo_group": " Work   Ops "},
            "due_at": None,
        }
    )

    assert formatted["group"] == "Work Ops"


def test_format_todo_prefers_explicit_group_field(mock_user_config_testuser):
    manager = _load_todo_manager_class()()

    formatted = manager._format_todo(
        {
            "id": "todo_group_explicit",
            "title": "Plan sprint",
            "group": " Product ",
            "source_metadata": {"todo_group": "Engineering"},
            "due_at": None,
        }
    )

    assert formatted["group"] == "Product"


def test_refresh_daily_briefing_after_todo_change_uses_background_thread(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True
    manager._client = object()

    calls = []

    class FakeThread:
        def __init__(self, *, target, args, daemon, name):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = name

        def start(self):
            calls.append(
                {
                    "args": self.args,
                    "daemon": self.daemon,
                    "name": self.name,
                }
            )
            self.target(*self.args)

    cache_invoked = []
    briefing_invoked = []

    def fake_refresh_todo_cache_files(*, user, limit=500):
        cache_invoked.append((user, limit))
        return {"success": True}

    def fake_upsert_daily_briefing(*, user, max_items=5, invalidated_todo_id=None, invalidated_todo_groups=None):
        briefing_invoked.append((user, max_items, invalidated_todo_id, invalidated_todo_groups))
        return {"success": True}

    monkeypatch.setattr(module.threading, "Thread", FakeThread)
    monkeypatch.setattr(manager, "refresh_todo_cache_files", fake_refresh_todo_cache_files)
    monkeypatch.setattr(manager, "upsert_daily_briefing", fake_upsert_daily_briefing)

    manager._refresh_daily_briefing_after_todo_change("morgan")

    assert calls == [
        {
            "args": ("Morgan", None, []),
            "daemon": True,
            "name": "todo-briefing-morgan",
        }
    ]
    assert cache_invoked == [("Morgan", 500)]
    assert briefing_invoked == [("Morgan", 5, None, [])]


def test_refresh_todo_cache_files_writes_all_and_per_source_files(mock_user_config_testuser, monkeypatch, tmp_path):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True
    manager._client = object()

    monkeypatch.setattr(module, "TODO_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        manager,
        "list_todos",
        lambda **_kwargs: [
            {
                "id": "todo_manual_open",
                "user_id": "Morgan",
                "title": "Call Logan",
                "completed": False,
                "source_type": "manual",
            },
            {
                "id": "todo_calendar_open",
                "user_id": "Morgan",
                "title": "Calendar Event",
                "completed": False,
                "source_type": "calendar",
            },
            {
                "id": "todo_manual_done",
                "user_id": "Morgan",
                "title": "Done Task",
                "completed": True,
                "source_type": "manual",
            },
        ],
    )

    result = manager.refresh_todo_cache_files(user="Morgan")

    user_dir = tmp_path / "morgan"
    assert result["success"] is True
    assert (user_dir / "all_todos_cache.json").exists()
    assert (user_dir / "manual_todos.json").exists()
    assert (user_dir / "calendar_todos.json").exists()
    all_payload = module.json.loads((user_dir / "all_todos_cache.json").read_text(encoding="utf-8"))
    manual_payload = module.json.loads((user_dir / "manual_todos.json").read_text(encoding="utf-8"))
    assert all_payload["counts"] == {"total": 3, "open": 2, "completed": 1}
    assert manual_payload["counts"] == {"total": 2, "open": 1, "completed": 1}


def test_create_todo_with_group_invalidates_group_briefings_for_non_calendar(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.payload = None
            self.action = "insert"

        def insert(self, payload):
            self.payload = payload
            self.action = "insert"
            return self

        def execute(self):
            if self.action == "insert":
                self.client.inserted.append(self.payload)
                return FakeResponse([self.payload])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.inserted = []

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()
    refresh_calls = []
    monkeypatch.setattr(
        manager,
        "_refresh_daily_briefing_after_todo_change",
        lambda user, **kwargs: refresh_calls.append((user, kwargs)),
    )

    created = manager.create_todo(
        user="TestUser",
        title="Outline launch email",
        source_type="manual",
        group="Work Ops",
    )

    assert created["group"] == "Work Ops"
    assert manager._client.inserted
    assert manager._client.inserted[-1]["source_metadata"]["todo_group"] == "Work Ops"
    assert refresh_calls
    _user, kwargs = refresh_calls[-1]
    assert kwargs.get("invalidated_todo_groups") == ["Work Ops"]


def test_create_todo_group_does_not_invalidate_for_calendar_source(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.payload = None
            self.action = "insert"

        def insert(self, payload):
            self.payload = payload
            self.action = "insert"
            return self

        def execute(self):
            if self.action == "insert":
                self.client.inserted.append(self.payload)
                return FakeResponse([self.payload])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.inserted = []

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()
    refresh_calls = []
    monkeypatch.setattr(
        manager,
        "_refresh_daily_briefing_after_todo_change",
        lambda user, **kwargs: refresh_calls.append((user, kwargs)),
    )

    manager.create_todo(
        user="TestUser",
        title="Calendar imported event",
        source_type="calendar",
        group="School",
    )

    assert refresh_calls
    _user, kwargs = refresh_calls[-1]
    assert kwargs.get("invalidated_todo_groups") is None


def test_build_todo_briefing_summaries_schedules_timed_items_and_excludes_calendar(mock_user_config_testuser, monkeypatch):
    manager = _load_todo_manager_class()()
    now = datetime.now(manager._tz)

    monkeypatch.setattr(
        manager,
        "list_todos",
        lambda **_kwargs: [
            {
                "id": "todo_manual_due",
                "user_id": "Morgan",
                "title": "Call Logan",
                "completed": False,
                "source_type": "manual",
                "due_at": (now + timedelta(hours=1)).astimezone().isoformat(),
            },
            {
                "id": "todo_calendar_due",
                "user_id": "Morgan",
                "title": "Calendar Event",
                "completed": False,
                "source_type": "calendar",
                "due_at": (now + timedelta(hours=1)).astimezone().isoformat(),
            },
            {
                "id": "todo_manual_undated",
                "user_id": "Morgan",
                "title": "Buy groceries",
                "completed": False,
                "source_type": "manual",
                "due_at": None,
            },
        ],
    )

    summaries = manager.build_todo_briefing_summaries(user="Morgan")

    assert sorted(summary["kind"] for summary in summaries) == ["timed_due", "undated"]
    timed_summary = next(summary for summary in summaries if summary["kind"] == "timed_due")
    assert timed_summary["todo_ids"] == ["todo_manual_due"]
    assert "Call Logan" in timed_summary["message"]
    assert all("todo_calendar_due" not in summary["todo_ids"] for summary in summaries)
    due_dt = manager._parse_due_from_row({"due_at": (now + timedelta(hours=1)).astimezone().isoformat()}).astimezone(manager._tz)
    active_from_dt = datetime.fromisoformat(timed_summary["active_from"].replace("Z", "+00:00")).astimezone(manager._tz)
    lead = due_dt - active_from_dt
    assert timedelta(minutes=14) <= lead <= timedelta(minutes=16)


def test_build_todo_briefing_timed_due_includes_placeholder_and_structured_meta(mock_user_config_testuser, monkeypatch):
    manager = _load_todo_manager_class()()
    now = datetime.now(manager._tz)

    monkeypatch.setattr(
        manager,
        "list_todos",
        lambda **_kwargs: [
            {
                "id": "todo_manual_due",
                "user_id": "Morgan",
                "title": "Finalize launch notes",
                "completed": False,
                "source_type": "manual",
                "due_at": (now + timedelta(hours=1)).astimezone(timezone.utc).isoformat(),
            }
        ],
    )

    summaries = manager.build_todo_briefing_summaries(user="Morgan")

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["kind"] == "timed_due"
    assert "{{TIME_UNTIL_DUE}}" in summary["message"]
    assert summary["todo_groups"] == []
    assert summary["urgency_bucket"] == "urgent"
    assert summary["due_at_iso"]
    assert summary["suggested_action"]
    assert summary["planner"]["base_time"]
    assert summary["planner"]["planned_send_time"]
    assert summary["planner"]["send_reason"] == "urgent_due_soon"
    assert summary["planner"]["urgent_override_applied"] is True


def test_send_time_planner_applies_window_snap_quiet_hours_and_urgent_override(mock_user_config_testuser):
    manager = _load_todo_manager_class()()

    now_local = datetime(2026, 4, 18, 10, 0, tzinfo=manager._tz)
    due_local = datetime(2026, 4, 19, 18, 0, tzinfo=manager._tz)
    snapped = manager._plan_timed_briefing_send(due_local=due_local, now_local=now_local)
    snapped_local = datetime.fromisoformat(snapped["planned_send_time"].replace("Z", "+00:00")).astimezone(manager._tz)
    assert snapped["send_reason"] == "timed_due_window_snap"
    assert snapped["urgent_override_applied"] is False
    assert (snapped_local.hour, snapped_local.minute) == (8, 30)

    urgent_due = datetime(2026, 4, 18, 11, 0, tzinfo=manager._tz)
    urgent = manager._plan_timed_briefing_send(due_local=urgent_due, now_local=now_local)
    urgent_local = datetime.fromisoformat(urgent["planned_send_time"].replace("Z", "+00:00")).astimezone(manager._tz)
    assert urgent["send_reason"] == "urgent_due_soon"
    assert urgent["urgent_override_applied"] is True
    assert (urgent_local.hour, urgent_local.minute) == (10, 45)

    quiet_now = datetime(2026, 4, 18, 22, 0, tzinfo=manager._tz)
    overdue_plan = manager._plan_overdue_briefing_send(now_local=quiet_now)
    overdue_local = datetime.fromisoformat(overdue_plan["planned_send_time"].replace("Z", "+00:00")).astimezone(manager._tz)
    assert overdue_plan["send_reason"] == "overdue_next_window"
    assert (overdue_local.hour, overdue_local.minute) == (8, 30)

    undated_plan = manager._plan_undated_briefing_send(now_local=datetime(2026, 4, 18, 13, 0, tzinfo=manager._tz))
    undated_local = datetime.fromisoformat(undated_plan["planned_send_time"].replace("Z", "+00:00")).astimezone(manager._tz)
    assert undated_plan["send_reason"] == "undated_morning_window"
    assert (undated_local.hour, undated_local.minute) == (8, 30)


def test_build_todo_briefing_undated_throttle_once_per_day(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.filters = []

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            rows = list(self.client.rows)
            for field, value in self.filters:
                rows = [row for row in rows if row.get(field) == value]
            return FakeResponse(rows)

    class FakeClient:
        def __init__(self):
            self.rows = [
                {
                    "id": "todo_digest_recent_undated",
                    "user_id": "TestUser",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "content": {
                        "meta": {
                            "source": "todo_digest",
                            "briefing_kind": "undated",
                        }
                    },
                }
            ]

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()
    monkeypatch.setattr(
        manager,
        "list_todos",
        lambda **_kwargs: [
            {
                "id": "todo_undated_1",
                "user_id": "TestUser",
                "title": "Pick dentist",
                "completed": False,
                "source_type": "manual",
                "due_at": None,
            }
        ],
    )

    summaries = manager.build_todo_briefing_summaries(user="TestUser")

    assert summaries == []


def test_upsert_daily_briefing_skips_stale_digest_and_inserts_new_rows(mock_user_config_testuser):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.action = "select"
            self.payload = None
            self.filters = []

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def insert(self, payload):
            self.payload = payload
            self.action = "insert"
            return self

        def update(self, payload):
            self.payload = payload
            self.action = "update"
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def execute(self):
            if self.action == "select":
                rows = list(self.client.briefings)
                for field, value in self.filters:
                    rows = [row for row in rows if row.get(field) == value]
                return FakeResponse(rows)
            if self.action == "update":
                self.client.updated.append((self.payload, list(self.filters)))
                for row in self.client.briefings:
                    if all(row.get(field) == value for field, value in self.filters):
                        row.update(self.payload)
                return FakeResponse([])
            if self.action == "insert":
                self.client.inserted.append(self.payload)
                self.client.briefings.append(dict(self.payload))
                return FakeResponse([self.payload])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.briefings = [
                {
                    "id": "todo_digest_morgan_old",
                    "user_id": "Morgan",
                    "status": "pending",
                    "opener_text": "Old digest",
                    "priority": "normal",
                    "content": {
                        "message": "Old digest",
                        "active_from": "2026-03-19T12:00:00+00:00",
                        "meta": {
                            "source": "todo_digest",
                            "briefing_key": "todo_due_todo_old",
                            "todo_ids": ["todo_old"],
                        },
                    },
                }
            ]
            self.inserted = []
            self.updated = []

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()
    manager.build_todo_briefing_summaries = lambda **_kwargs: [
        {
            "briefing_key": "todo_due_todo_new",
            "message": "Quick heads up: 'Test' is due in {{TIME_UNTIL_DUE}}. Want to start it now or move the due time?",
            "todo_ids": ["todo_new"],
            "active_from": "2026-03-19T12:00:00+00:00",
            "priority": "high",
            "kind": "timed_due",
            "source_types": ["manual"],
            "todo_groups": ["Work Ops"],
            "urgency_bucket": "urgent",
            "due_at_iso": "2026-03-19T13:00:00+00:00",
            "suggested_action": "Want to start it now or move the due time?",
            "planner": {
                "base_time": "2026-03-19T12:45:00+00:00",
                "planned_send_time": "2026-03-19T12:00:00+00:00",
                "send_reason": "urgent_due_soon",
                "urgent_override_applied": True,
            },
        }
    ]

    result = manager.upsert_daily_briefing(user="Morgan", invalidated_todo_id="todo_old")

    assert result["briefing_created"] is True
    assert result["briefings_skipped"] == 1
    assert len(result["briefing_ids"]) == 1
    assert manager._client.updated
    update_payload, _filters = manager._client.updated[-1]
    assert update_payload["status"] == "skipped"
    payload = manager._client.inserted[-1]
    assert payload["id"].startswith("todo_digest_morgan_")
    assert payload["priority"] == "high"
    assert payload["status"] == "pending"
    assert payload["discord_status"] == "pending"
    assert payload["voice_status"] == "pending"
    assert payload["discord_sent_at"] is None
    assert payload["voice_read_at"] is None
    assert payload["dismissed_at"] is None
    assert payload["content"]["meta"]["briefing_key"] == "todo_due_todo_new"
    assert payload["content"]["meta"]["todo_groups"] == ["Work Ops"]
    assert payload["content"]["meta"]["urgency_bucket"] == "urgent"
    assert payload["content"]["meta"]["due_at_iso"] == "2026-03-19T13:00:00+00:00"
    assert payload["content"]["meta"]["suggested_action"] == "Want to start it now or move the due time?"
    assert payload["content"]["meta"]["send_reason"] == "urgent_due_soon"
    assert payload["content"]["meta"]["urgent_override_applied"] is True


def test_upsert_daily_briefing_invalidated_group_forces_refresh_only_for_that_group(mock_user_config_testuser):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.action = "select"
            self.payload = None
            self.filters = []

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def insert(self, payload):
            self.payload = payload
            self.action = "insert"
            return self

        def update(self, payload):
            self.payload = payload
            self.action = "update"
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def execute(self):
            if self.action == "select":
                rows = list(self.client.briefings)
                for field, value in self.filters:
                    rows = [row for row in rows if row.get(field) == value]
                return FakeResponse(rows)
            if self.action == "update":
                self.client.updated.append((self.payload, list(self.filters)))
                for row in self.client.briefings:
                    if all(row.get(field) == value for field, value in self.filters):
                        row.update(self.payload)
                return FakeResponse([])
            if self.action == "insert":
                self.client.inserted.append(self.payload)
                self.client.briefings.append(dict(self.payload))
                return FakeResponse([self.payload])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.briefings = [
                {
                    "id": "todo_digest_work_existing",
                    "user_id": "Morgan",
                    "status": "pending",
                    "opener_text": "Work digest opener",
                    "priority": "normal",
                    "content": {
                        "message": "Work digest opener",
                        "active_from": "2026-03-19T12:00:00+00:00",
                        "meta": {
                            "source": "todo_digest",
                            "briefing_key": "todo_due_todo_work",
                            "briefing_kind": "timed_due",
                            "source_types": ["manual"],
                            "todo_groups": ["Work Ops"],
                            "todo_ids": ["todo_work"],
                            "urgency_bucket": "soon",
                            "due_at_iso": "2026-03-19T13:00:00+00:00",
                            "suggested_action": "Start it now?",
                            "base_time": "2026-03-19T11:15:00+00:00",
                            "planned_send_time": "2026-03-19T12:00:00+00:00",
                            "send_reason": "timed_due_window_snap",
                            "urgent_override_applied": False,
                        },
                    },
                },
                {
                    "id": "todo_digest_home_existing",
                    "user_id": "Morgan",
                    "status": "pending",
                    "opener_text": "Home digest opener",
                    "priority": "low",
                    "content": {
                        "message": "Home digest opener",
                        "active_from": "2026-03-19T12:30:00+00:00",
                        "meta": {
                            "source": "todo_digest",
                            "briefing_key": "undated:todo_home",
                            "briefing_kind": "undated",
                            "source_types": ["manual"],
                            "todo_groups": ["Home"],
                            "todo_ids": ["todo_home"],
                            "urgency_bucket": "normal",
                            "due_at_iso": None,
                            "suggested_action": "Pick one to schedule.",
                            "base_time": "2026-03-19T12:00:00+00:00",
                            "planned_send_time": "2026-03-19T12:30:00+00:00",
                            "send_reason": "undated_morning_window",
                            "urgent_override_applied": False,
                        },
                    },
                },
            ]
            self.inserted = []
            self.updated = []

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()
    manager.build_todo_briefing_summaries = lambda **_kwargs: [
        {
            "briefing_key": "todo_due_todo_work",
            "message": "Work digest opener",
            "todo_ids": ["todo_work"],
            "active_from": "2026-03-19T12:00:00+00:00",
            "priority": "normal",
            "kind": "timed_due",
            "source_types": ["manual"],
            "todo_groups": ["Work Ops"],
            "urgency_bucket": "soon",
            "due_at_iso": "2026-03-19T13:00:00+00:00",
            "suggested_action": "Start it now?",
            "planner": {
                "base_time": "2026-03-19T11:15:00+00:00",
                "planned_send_time": "2026-03-19T12:00:00+00:00",
                "send_reason": "timed_due_window_snap",
                "urgent_override_applied": False,
            },
        },
        {
            "briefing_key": "undated:todo_home",
            "message": "Home digest opener",
            "todo_ids": ["todo_home"],
            "active_from": "2026-03-19T12:30:00+00:00",
            "priority": "low",
            "kind": "undated",
            "source_types": ["manual"],
            "todo_groups": ["Home"],
            "urgency_bucket": "normal",
            "due_at_iso": None,
            "suggested_action": "Pick one to schedule.",
            "planner": {
                "base_time": "2026-03-19T12:00:00+00:00",
                "planned_send_time": "2026-03-19T12:30:00+00:00",
                "send_reason": "undated_morning_window",
                "urgent_override_applied": False,
            },
        },
    ]
    manager.list_todos = lambda **_kwargs: [
        {
            "id": "todo_work",
            "title": "Work todo",
            "source_type": "manual",
            "source_metadata": {"todo_group": "Work Ops"},
            "completed": False,
        },
        {
            "id": "todo_home",
            "title": "Home todo",
            "source_type": "manual",
            "source_metadata": {"todo_group": "Home"},
            "completed": False,
        },
    ]

    result = manager.upsert_daily_briefing(
        user="Morgan",
        invalidated_todo_groups=["Work Ops"],
    )

    assert result["briefings_skipped"] == 1
    assert result["briefing_created"] is True
    assert len(manager._client.inserted) == 1
    update_payload, filters = manager._client.updated[-1]
    assert update_payload["status"] == "skipped"
    assert ("id", "todo_digest_work_existing") in filters


def test_add_todo_to_calendar_links_manual_todo_without_creating_duplicate(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    due_at = (datetime.now(manager._tz) + timedelta(hours=2)).astimezone().isoformat()

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.action = "select"
            self.payload = None
            self.filters = []

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def update(self, payload):
            self.payload = payload
            self.action = "update"
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def execute(self):
            if self.action == "select":
                rows = list(self.client.rows)
                for field, value in self.filters:
                    rows = [row for row in rows if row.get(field) == value]
                return FakeResponse(rows)
            if self.action == "update":
                self.client.updated.append((self.payload, list(self.filters)))
                for row in self.client.rows:
                    if all(row.get(field) == value for field, value in self.filters):
                        row.update(self.payload)
                return FakeResponse([])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.rows = [
                {
                    "id": "todo_manual",
                    "user_id": "TestUser",
                    "title": "Call Logan",
                    "details": "Bring notes",
                    "due_at": due_at,
                    "completed": False,
                    "source_type": "manual",
                    "source_id": "manual_1",
                    "source_metadata": {"created_via": "todo_overlay"},
                    "created_at": due_at,
                    "updated_at": due_at,
                }
            ]
            self.updated = []

        def table(self, _name):
            return FakeQuery(self)

    created_events = []

    class FakeCalendarComponent:
        def __init__(self, user):
            self.user = user

        def create_event(self, event_data):
            created_events.append((self.user, dict(event_data)))
            return {
                "id": "evt_123",
                "htmlLink": "https://calendar.google.com/event?eid=evt_123",
                "start": {"dateTime": f"{event_data['date']}T{event_data['start_time']}:00"},
                "end": {"dateTime": f"{event_data['date']}T{event_data['end_time']}:00"},
            }

    fake_calendar_module = types.ModuleType("mcp_server.clients.calendar_client")
    fake_calendar_module.CalendarComponent = FakeCalendarComponent
    monkeypatch.setitem(sys.modules, "mcp_server.clients.calendar_client", fake_calendar_module)

    manager._client = FakeClient()

    result = manager.add_todo_to_calendar(user="TestUser", todo_id="todo_manual")

    assert result["calendar_event"]["id"] == "evt_123"
    assert created_events
    calendar_user, event_data = created_events[-1]
    assert calendar_user == "testuser_personal"
    assert event_data["title"] == "Call Logan"
    assert event_data["attendees"] == []
    assert manager._client.updated
    update_payload, filters = manager._client.updated[-1]
    assert ("id", "todo_manual") in filters
    assert update_payload["source_metadata"]["created_via"] == "todo_overlay"
    assert update_payload["source_metadata"]["linked_calendar_event_id"] == "evt_123"
    assert update_payload["source_metadata"]["linked_calendar_user"] == "testuser_personal"
    assert update_payload["source_metadata"]["linked_calendar_attendees"] == []
    assert update_payload["source_metadata"]["linked_calendar_events"] == [
        {
            "id": "evt_123",
            "calendar_user": "testuser_personal",
            "htmlLink": "https://calendar.google.com/event?eid=evt_123",
            "start": {"dateTime": f"{event_data['date']}T{event_data['start_time']}:00"},
            "end": {"dateTime": f"{event_data['date']}T{event_data['end_time']}:00"},
            "attendees": [],
        }
    ]


def test_add_todo_to_multiple_calendars(mock_user_config_testuser, monkeypatch):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    due_at = "2026-03-25T17:00:00+00:00"

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.action = "select"
            self.payload = None
            self.filters = []

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def update(self, payload):
            self.payload = payload
            self.action = "update"
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def execute(self):
            if self.action == "select":
                rows = list(self.client.rows)
                for field, value in self.filters:
                    rows = [row for row in rows if row.get(field) == value]
                return FakeResponse(rows)
            if self.action == "update":
                self.client.updated.append((self.payload, list(self.filters)))
                for row in self.client.rows:
                    if all(row.get(field) == value for field, value in self.filters):
                        row.update(self.payload)
                return FakeResponse([])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.rows = [
                {
                    "id": "todo_manual",
                    "user_id": "TestUser",
                    "title": "Call Logan",
                    "details": "Bring notes",
                    "due_at": due_at,
                    "completed": False,
                    "source_type": "manual",
                    "source_id": "manual_1",
                    "source_metadata": {"created_via": "todo_overlay"},
                    "created_at": due_at,
                    "updated_at": due_at,
                }
            ]
            self.updated = []

        def table(self, _name):
            return FakeQuery(self)

    created_events = []

    class FakeCalendarComponent:
        def __init__(self, user):
            self.user = user

        def create_event(self, event_data):
            created_events.append((self.user, dict(event_data)))
            suffix = self.user.replace("testuser_", "")
            return {
                "id": f"evt_{suffix}",
                "htmlLink": f"https://calendar.google.com/event?eid=evt_{suffix}",
                "start": {"dateTime": f"{event_data['date']}T{event_data['start_time']}:00"},
                "end": {"dateTime": f"{event_data['date']}T{event_data['end_time']}:00"},
            }

    fake_calendar_module = types.ModuleType("mcp_server.clients.calendar_client")
    fake_calendar_module.CalendarComponent = FakeCalendarComponent
    monkeypatch.setitem(sys.modules, "mcp_server.clients.calendar_client", fake_calendar_module)

    manager._client = FakeClient()

    result = manager.add_todo_to_calendar(
        user="TestUser",
        todo_id="todo_manual",
        calendar_users=["testuser_personal", "testuser_school"],
    )

    assert [event["calendar_user"] for event in result["calendar_events"]] == [
        "testuser_personal",
        "testuser_school",
    ]
    assert [user for user, _event_data in created_events] == [
        "testuser_personal",
        "testuser_school",
    ]
    update_payload, filters = manager._client.updated[-1]
    assert ("id", "todo_manual") in filters
    assert update_payload["source_metadata"]["linked_calendar_event_id"] == "evt_personal"
    assert update_payload["source_metadata"]["linked_calendar_user"] == "testuser_personal"
    assert [event["calendar_user"] for event in update_payload["source_metadata"]["linked_calendar_events"]] == [
        "testuser_personal",
        "testuser_school",
    ]
    assert [event["id"] for event in update_payload["source_metadata"]["linked_calendar_events"]] == [
        "evt_personal",
        "evt_school",
    ]


def test_upsert_calendar_todo_skips_duplicate_for_linked_manual_todo(mock_user_config_testuser):
    module = _load_todo_manager_module()
    manager = module.TodoManager()
    manager._initialized = True

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, client):
            self.client = client
            self.action = "select"
            self.payload = None
            self.filters = []

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def update(self, payload):
            self.payload = payload
            self.action = "update"
            return self

        def insert(self, payload):
            self.payload = payload
            self.action = "insert"
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            return self

        def execute(self):
            if self.action == "select":
                rows = list(self.client.rows)
                for field, value in self.filters:
                    rows = [row for row in rows if row.get(field) == value]
                return FakeResponse(rows)
            if self.action == "update":
                self.client.updated.append((self.payload, list(self.filters)))
                for row in self.client.rows:
                    if all(row.get(field) == value for field, value in self.filters):
                        row.update(self.payload)
                return FakeResponse([])
            if self.action == "insert":
                self.client.inserted.append(self.payload)
                return FakeResponse([self.payload])
            raise AssertionError(f"Unsupported action: {self.action}")

    class FakeClient:
        def __init__(self):
            self.rows = [
                {
                    "id": "todo_manual",
                    "user_id": "TestUser",
                    "title": "Call Logan",
                    "details": "Bring notes",
                    "due_at": "2026-03-25T17:00:00+00:00",
                    "completed": False,
                    "source_type": "manual",
                    "source_id": "manual_1",
                    "source_metadata": {"linked_calendar_event_id": "evt_123"},
                    "created_at": "2026-03-24T10:00:00+00:00",
                    "updated_at": "2026-03-24T10:00:00+00:00",
                }
            ]
            self.updated = []
            self.inserted = []

        def table(self, _name):
            return FakeQuery(self)

    manager._client = FakeClient()

    result = manager.upsert_calendar_todo(
        calendar_user="testuser_personal",
        event={
            "id": "evt_123",
            "summary": "Call Logan",
            "description": "Bring notes",
            "location": "",
            "start_date": "2026-03-25",
            "start_time": "17:00",
            "end_date": "2026-03-25",
            "end_time": "18:00",
            "calendar_name": "testuser_personal",
            "all_day": False,
        },
    )

    assert result["id"] == "todo_manual"
    assert manager._client.updated
    assert not manager._client.inserted
    update_payload, filters = manager._client.updated[-1]
    assert ("id", "todo_manual") in filters
    assert update_payload["source_metadata"]["linked_calendar_events"][0]["id"] == "evt_123"
    assert update_payload["source_metadata"]["linked_calendar_events"][0]["calendar_user"] == "testuser_personal"
