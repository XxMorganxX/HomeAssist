from datetime import datetime, timedelta
import importlib.util
from pathlib import Path


def _load_todo_manager_class():
    module_path = Path(__file__).parent.parent / "assistant_framework" / "utils" / "todo_manager.py"
    spec = importlib.util.spec_from_file_location("todo_manager_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TodoManager


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
