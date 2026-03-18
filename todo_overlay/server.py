"""Local browser overlay for viewing and editing HomeAssist todos."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from assistant_framework.utils.todo_manager import TodoManager


logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
CALENDAR_CACHE_PATH = Path(__file__).resolve().parent.parent / "discord_bot" / "state" / "calendar_todo_cache.json"
TODO_CALENDAR_WINDOW_DAYS = 7
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8421
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


class TodoOverlayServer(ThreadingHTTPServer):
    """Threaded HTTP server with a shared TodoManager instance."""

    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], request_handler):
        super().__init__(server_address, request_handler)
        self.todo_manager = TodoManager()
        self.default_user = self.todo_manager.normalize_user(None)

    def refresh_briefing(self, user: Optional[str]) -> bool:
        """Best-effort refresh for the daily todo digest."""
        if not self.todo_manager.is_available():
            return False

        try:
            self.todo_manager.upsert_daily_briefing(user=user or self.default_user)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to refresh todo briefing: %s", exc)
            return False


def _is_calendar_todo(todo: Dict[str, Any]) -> bool:
    return str(todo.get("source_type") or "").lower() == "calendar"


def _decorate_todo(todo: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(todo)
    is_calendar = _is_calendar_todo(row)
    row["readonly"] = is_calendar
    row["can_edit"] = not is_calendar
    row["can_toggle_complete"] = not is_calendar
    row["can_delete"] = not is_calendar
    return row


def _is_calendar_todo_in_window(todo_manager: TodoManager, todo: Dict[str, Any], *, days: int) -> bool:
    if not _is_calendar_todo(todo):
        return False

    due_dt = todo_manager._parse_due_from_row(todo)
    if due_dt is None:
        return False

    due_local = due_dt.astimezone(todo_manager._tz)
    now_local = datetime.now(todo_manager._tz)
    latest_local = now_local + timedelta(days=days)
    return now_local <= due_local <= latest_local


def _sort_todos_for_ui(todo_manager: TodoManager, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(todos, key=todo_manager._todo_sort_key)


def _load_cached_calendar_todos(todo_manager: TodoManager, *, user: str) -> List[Dict[str, Any]]:
    if not CALENDAR_CACHE_PATH.exists():
        return []

    try:
        payload = json.loads(CALENDAR_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to read calendar todo cache: %s", exc)
        return []

    cached_todos = payload.get("calendar_todos", [])
    if not isinstance(cached_todos, list):
        return []

    merged: List[Dict[str, Any]] = []
    for todo in cached_todos:
        if not isinstance(todo, dict):
            continue
        if str(todo.get("user_id") or "").lower() != user.lower():
            continue
        if bool(todo.get("completed")):
            continue
        formatted = todo_manager._format_todo(todo)
        if _is_calendar_todo_in_window(todo_manager, formatted, days=TODO_CALENDAR_WINDOW_DAYS):
            merged.append(formatted)
    return _sort_todos_for_ui(todo_manager, merged)


def _get_todos_for_mode(todo_manager: TodoManager, *, user: str, mode: str, limit: int) -> List[Dict[str, Any]]:
    cached_calendar_todos = _load_cached_calendar_todos(todo_manager, user=user)

    if mode == "due":
        regular_due = [
            todo
            for todo in todo_manager.list_todos(
                user=user,
                include_completed=False,
                limit=limit,
                only_due_today=True,
            )
            if not _is_calendar_todo(todo)
        ]
        today = datetime.now(todo_manager._tz).date()
        due_cached = [
            todo
            for todo in cached_calendar_todos
            if (todo_manager._parse_due_from_row(todo) and todo_manager._parse_due_from_row(todo).astimezone(todo_manager._tz).date() == today)
        ]
        return _sort_todos_for_ui(todo_manager, regular_due + due_cached)[:limit]

    if mode == "completed":
        completed = [
            todo
            for todo in todo_manager.list_todos(
                user=user,
                include_completed=True,
                limit=limit,
            )
            if todo.get("completed")
        ]
        return _sort_todos_for_ui(todo_manager, completed)[:limit]

    if mode == "all":
        regular_all = [
            todo
            for todo in todo_manager.list_todos(
                user=user,
                include_completed=True,
                limit=limit,
            )
            if not _is_calendar_todo(todo)
        ]
        return _sort_todos_for_ui(todo_manager, regular_all + cached_calendar_todos)[:limit]

    regular_open = [
        todo
        for todo in todo_manager.list_todos(
            user=user,
            include_completed=False,
            limit=limit,
        )
        if not _is_calendar_todo(todo)
    ]
    return _sort_todos_for_ui(todo_manager, regular_open + cached_calendar_todos)[:limit]


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _parse_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}

    raw = handler.rfile.read(content_length)
    if not raw:
        return {}

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc.msg}") from exc

    if not isinstance(decoded, dict):
        raise ValueError("JSON body must be an object.")
    return decoded


def make_handler() -> type[BaseHTTPRequestHandler]:
    class TodoOverlayHandler(BaseHTTPRequestHandler):
        server: TodoOverlayServer

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "success": True,
                        "available": self.server.todo_manager.is_available(),
                        "default_user": self.server.default_user,
                    }
                )
                return

            if parsed.path == "/api/config":
                self._send_json(
                    {
                        "success": True,
                        "default_user": self.server.default_user,
                        "refresh_interval_seconds": 30,
                    }
                )
                return

            if parsed.path == "/api/todos/summary":
                self._handle_summary(parsed.query)
                return

            if parsed.path == "/api/todos":
                self._handle_list(parsed.query)
                return

            self._serve_static(parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/todos":
                self._handle_create()
                return

            route = self._todo_route(parsed.path)
            if route and route["action"] == "complete":
                self._handle_toggle(route["todo_id"], completed=True)
                return
            if route and route["action"] == "reopen":
                self._handle_toggle(route["todo_id"], completed=False)
                return

            self._send_json({"success": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = self._todo_route(parsed.path)
            if route and route["action"] == "update":
                self._handle_update(route["todo_id"])
                return

            self._send_json({"success": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = self._todo_route(parsed.path)
            if route and route["action"] == "delete":
                self._handle_delete(route["todo_id"])
                return

            self._send_json({"success": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _serve_static(self, path: str) -> None:
            normalized = path or "/"
            if normalized == "/":
                normalized = "/index.html"

            requested = (STATIC_DIR / normalized.lstrip("/")).resolve()
            if STATIC_DIR not in requested.parents and requested != STATIC_DIR:
                self._send_json({"success": False, "error": "Forbidden."}, status=HTTPStatus.FORBIDDEN)
                return

            if not requested.is_file():
                self._send_json({"success": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)
                return

            content_type = "text/plain; charset=utf-8"
            if requested.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif requested.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif requested.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"

            payload = requested.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _handle_summary(self, query: str) -> None:
            manager = self.server.todo_manager
            user = self._query_value(query, "user") or self.server.default_user
            cached_calendar_todos = _load_cached_calendar_todos(manager, user=user)

            if not manager.is_available() and not cached_calendar_todos:
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            if manager.is_available():
                open_todos = _get_todos_for_mode(manager, user=user, mode="open", limit=200)
                due_today = _get_todos_for_mode(manager, user=user, mode="due", limit=200)
                completed = _get_todos_for_mode(manager, user=user, mode="completed", limit=200)
            else:
                today = datetime.now(manager._tz).date()
                open_todos = cached_calendar_todos
                due_today = [
                    todo
                    for todo in cached_calendar_todos
                    if (manager._parse_due_from_row(todo) and manager._parse_due_from_row(todo).astimezone(manager._tz).date() == today)
                ]
                completed = []
            self._send_json(
                {
                    "success": True,
                    "counts": {
                        "open": len(open_todos),
                        "due": len(due_today),
                        "completed": len(completed),
                    },
                }
            )

        def _handle_list(self, query: str) -> None:
            manager = self.server.todo_manager
            user = self._query_value(query, "user") or self.server.default_user
            mode = (self._query_value(query, "mode") or "open").lower()
            limit = self._query_int(query, "limit", default=100, minimum=1, maximum=500)
            cached_calendar_todos = _load_cached_calendar_todos(manager, user=user)

            if not manager.is_available() and not cached_calendar_todos:
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            if manager.is_available():
                todos = _get_todos_for_mode(manager, user=user, mode=mode, limit=limit)
            else:
                if mode not in {"open", "due", "all"}:
                    todos = []
                elif mode == "due":
                    today = datetime.now(manager._tz).date()
                    todos = [
                        todo
                        for todo in cached_calendar_todos
                        if (manager._parse_due_from_row(todo) and manager._parse_due_from_row(todo).astimezone(manager._tz).date() == today)
                    ][:limit]
                else:
                    todos = cached_calendar_todos[:limit]

            self._send_json(
                {
                    "success": True,
                    "mode": mode,
                    "user": user,
                    "count": len(todos),
                    "todos": [_decorate_todo(todo) for todo in todos],
                }
            )

        def _handle_create(self) -> None:
            manager = self.server.todo_manager
            if not manager.is_available():
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            try:
                payload = _parse_json_body(self)
                title = str(payload.get("title") or "").strip()
                if not title:
                    raise ValueError("Title is required.")

                user = str(payload.get("user") or self.server.default_user)
                todo = manager.create_todo(
                    user=user,
                    title=title,
                    details=self._optional_string(payload, "details"),
                    due_at=self._optional_string(payload, "due_at"),
                    source_type="manual",
                    source_id=self._optional_string(payload, "source_id"),
                    source_metadata={"created_via": "todo_overlay"},
                )
                briefing_refreshed = self.server.refresh_briefing(user)
                self._send_json(
                    {
                        "success": True,
                        "action": "create",
                        "briefing_refreshed": briefing_refreshed,
                        "todo": _decorate_todo(todo),
                    },
                    status=HTTPStatus.CREATED,
                )
            except ValueError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_update(self, todo_id: str) -> None:
            manager = self.server.todo_manager
            if not manager.is_available():
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            try:
                payload = _parse_json_body(self)
                user = str(payload.get("user") or self.server.default_user)
                current = manager.resolve_todo(user=user, todo_id=todo_id, include_completed=True)
                self._ensure_mutable(current)

                title_supplied = "title" in payload
                details_supplied = "details" in payload
                due_supplied = "due_at" in payload
                clear_due_at = bool(payload.get("clear_due_at"))
                if due_supplied and not str(payload.get("due_at") or "").strip():
                    clear_due_at = True

                if not any(key in payload for key in ("title", "details", "due_at", "clear_due_at")):
                    raise ValueError("Update requires at least one field change.")

                title = None
                if title_supplied:
                    title = str(payload.get("title") or "").strip()
                    if not title:
                        raise ValueError("Title cannot be empty.")

                details = None
                if details_supplied:
                    details = "" if payload.get("details") is None else str(payload.get("details"))

                due_at = None
                if due_supplied and not clear_due_at:
                    due_at = str(payload.get("due_at") or "").strip()

                todo = manager.update_todo(
                    user=user,
                    todo_id=todo_id,
                    title=title,
                    details=details,
                    due_at=due_at,
                    clear_due_at=clear_due_at,
                )
                briefing_refreshed = self.server.refresh_briefing(user)
                self._send_json(
                    {
                        "success": True,
                        "action": "update",
                        "briefing_refreshed": briefing_refreshed,
                        "todo": _decorate_todo(todo),
                    }
                )
            except PermissionError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ValueError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_toggle(self, todo_id: str, *, completed: bool) -> None:
            manager = self.server.todo_manager
            if not manager.is_available():
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            try:
                payload = _parse_json_body(self)
                user = str(payload.get("user") or self.server.default_user)
                current = manager.resolve_todo(user=user, todo_id=todo_id, include_completed=True)
                self._ensure_mutable(current)

                if completed:
                    todo = manager.complete_todo(user=user, todo_id=todo_id)
                    action = "complete"
                else:
                    todo = manager.reopen_todo(user=user, todo_id=todo_id)
                    action = "reopen"

                briefing_refreshed = self.server.refresh_briefing(user)
                self._send_json(
                    {
                        "success": True,
                        "action": action,
                        "briefing_refreshed": briefing_refreshed,
                        "todo": _decorate_todo(todo),
                    }
                )
            except PermissionError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ValueError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_delete(self, todo_id: str) -> None:
            manager = self.server.todo_manager
            if not manager.is_available():
                self._send_json(
                    {"success": False, "error": "Supabase is not available for todos."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            try:
                payload = _parse_json_body(self)
                user = str(payload.get("user") or self.server.default_user)
                current = manager.resolve_todo(user=user, todo_id=todo_id, include_completed=True)
                self._ensure_mutable(current)
                todo = manager.delete_todo(user=user, todo_id=todo_id)
                briefing_refreshed = self.server.refresh_briefing(user)
                self._send_json(
                    {
                        "success": True,
                        "action": "delete",
                        "briefing_refreshed": briefing_refreshed,
                        "todo": _decorate_todo(todo),
                    }
                )
            except PermissionError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ValueError as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _todo_route(self, path: str) -> Optional[Dict[str, str]]:
            parts = [segment for segment in path.split("/") if segment]
            if len(parts) < 3 or parts[0] != "api" or parts[1] != "todos":
                return None

            todo_id = parts[2]
            if len(parts) == 3:
                return {"todo_id": todo_id, "action": "delete" if self.command == "DELETE" else "update"}
            if len(parts) == 4 and parts[3] in {"complete", "reopen"}:
                return {"todo_id": todo_id, "action": parts[3]}
            return None

        def _ensure_mutable(self, todo: Dict[str, Any]) -> None:
            if _is_calendar_todo(todo):
                raise PermissionError("Calendar-backed todos are read-only in the overlay.")

        def _query_value(self, query: str, key: str) -> Optional[str]:
            values = parse_qs(query).get(key, [])
            if not values:
                return None
            return values[0]

        def _query_int(
            self,
            query: str,
            key: str,
            *,
            default: int,
            minimum: int,
            maximum: int,
        ) -> int:
            raw = self._query_value(query, key)
            if raw is None:
                return default

            try:
                value = int(raw)
            except ValueError:
                return default
            return max(minimum, min(maximum, value))

        def _optional_string(self, payload: Dict[str, Any], key: str) -> Optional[str]:
            if key not in payload:
                return None
            value = payload.get(key)
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, payload: Dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return TodoOverlayHandler


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> TodoOverlayServer:
    """Create a configured overlay server instance without starting it."""
    return TodoOverlayServer((host, port), make_handler())


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    server = create_server(host=host, port=port)
    logger.info("Starting todo overlay on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        logger.info("Stopping todo overlay")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HomeAssist todo overlay")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to.")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
