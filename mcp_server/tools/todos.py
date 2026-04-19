"""
Persistent todo management tool backed by Supabase.
"""

from typing import Any, Dict

from assistant_framework.utils.todo_manager import TodoManager
from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS


class TodosTool(BaseTool):
    """Manage persistent todos and user-driven reminder intent."""

    name = "todos"
    description = """Manage the user's persistent todo list.

Use this tool when the user wants to:
- Add a task or reminder to their todo list
- Ask what tasks they still have open
- Mark a task done or reopen it
- Edit or delete a task
- Ask what is due today or overdue
- Replace a task with its next step (advance a chain)
- View the history of a chained task

IMPORTANT:
- Use `todos` for persistent action items and user-driven reminders.
- Use `briefing` only for autonomous announcement management.
- Use `stickies` for desktop notes/macOS Stickies content, not task tracking.

Timing:
- For a direct reminder-style request like "remind me tomorrow at 5pm to call mom", use action='create' with `title` and `due_at`.
- For an event-relative reminder like "remind me 30 minutes before my 3pm meeting", use action='create' with `title`, `event_time`, and `remind_before_minutes`.

Chains:
- Use action='advance' when the user wants to replace a todo with its next step. This completes the current todo and creates a new one linked in a chain. Requires `title` for the next step, plus `todo_id` or `match` to identify the current todo.
- Use action='chain' with `todo_id` to view the full history of a chained todo.
"""
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._todos = TodoManager()

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "complete", "reopen", "update", "delete", "advance", "chain"],
                    "description": "Todo action to perform. 'advance' completes a todo and creates its next step in a chain. 'chain' returns the full history of a chained todo.",
                    "default": "create",
                },
                "user": {
                    "type": "string",
                    "description": "Target user for the todo. Defaults to the primary user.",
                },
                "title": {
                    "type": "string",
                    "description": "Todo title/task text. Required for create. Also used for update when renaming.",
                },
                "message": {
                    "type": "string",
                    "description": "Alias for title when the request is phrased as a reminder message.",
                },
                "details": {
                    "type": "string",
                    "description": "Optional longer details or context for the todo.",
                },
                "group": {
                    "type": "string",
                    "description": "Optional user-defined group/folder name for organizing todos.",
                },
                "due_at": {
                    "type": "string",
                    "description": "Optional due time. Accepts ISO 8601 or natural language like 'tomorrow 5pm'.",
                },
                "event_time": {
                    "type": "string",
                    "description": "Optional event time for event-relative reminders.",
                },
                "remind_before_minutes": {
                    "type": "integer",
                    "description": "Minutes before `event_time` to set the todo due time.",
                    "minimum": 1,
                    "maximum": 10080,
                },
                "todo_id": {
                    "type": "string",
                    "description": "Specific todo id for complete, reopen, update, or delete.",
                },
                "match": {
                    "type": "string",
                    "description": "Case-insensitive title match when todo_id is not known.",
                },
                "include_completed": {
                    "type": "boolean",
                    "description": "For list: include completed todos too.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "For list: max todos to return.",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "only_due_today": {
                    "type": "boolean",
                    "description": "For list: return only incomplete todos due today.",
                    "default": False,
                },
                "only_overdue": {
                    "type": "boolean",
                    "description": "For list: return only incomplete overdue todos.",
                    "default": False,
                },
                "clear_due_at": {
                    "type": "boolean",
                    "description": "For update: clear any due time on the todo.",
                    "default": False,
                },
                "clear_group": {
                    "type": "boolean",
                    "description": "For update: remove the todo from any custom group.",
                    "default": False,
                },
                "source_type": {
                    "type": "string",
                    "enum": ["voice", "discord", "calendar", "manual"],
                    "description": "Origin of the todo. Defaults to 'voice' for assistant-created tasks.",
                    "default": "voice",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional source identifier. Usually omitted for voice-created todos.",
                },
            },
            "required": ["action"],
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Todos -- %s", params)

            if not self._todos.is_available():
                return {
                    "success": False,
                    "error": "Supabase is not available. Cannot access persistent todos.",
                }

            action = params.get("action", "create")
            if action == "create":
                return self._create(params)
            if action == "list":
                return self._list(params)
            if action == "complete":
                return self._complete(params)
            if action == "reopen":
                return self._reopen(params)
            if action == "update":
                return self._update(params)
            if action == "delete":
                return self._delete(params)
            if action == "advance":
                return self._advance(params)
            if action == "chain":
                return self._chain(params)
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as exc:
            self.logger.error("Todos tool error: %s", exc)
            return {"success": False, "error": str(exc)}

    def _create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        title = (params.get("title") or params.get("message") or "").strip()
        if not title:
            return {
                "success": False,
                "error": "Title is required to create a todo.",
            }

        todo = self._todos.create_todo(
            user=params.get("user"),
            title=title,
            details=params.get("details"),
            due_at=params.get("due_at"),
            event_time=params.get("event_time"),
            remind_before_minutes=params.get("remind_before_minutes"),
            source_type=params.get("source_type", "voice"),
            source_id=params.get("source_id"),
            group=params.get("group"),
            source_metadata={
                "created_via": "todos_tool",
                "event_time": params.get("event_time"),
                "remind_before_minutes": params.get("remind_before_minutes"),
            },
        )
        return {
            "success": True,
            "action": "create",
            "todo": todo,
        }

    def _list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        todos = self._todos.list_todos(
            user=params.get("user"),
            include_completed=params.get("include_completed", False),
            limit=params.get("limit", 10),
            only_due_today=params.get("only_due_today", False),
            only_overdue=params.get("only_overdue", False),
        )
        return {
            "success": True,
            "action": "list",
            "count": len(todos),
            "todos": todos,
        }

    def _complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        todo = self._todos.complete_todo(
            user=params.get("user"),
            todo_id=params.get("todo_id"),
            match=params.get("match"),
        )
        return {"success": True, "action": "complete", "todo": todo}

    def _reopen(self, params: Dict[str, Any]) -> Dict[str, Any]:
        todo = self._todos.reopen_todo(
            user=params.get("user"),
            todo_id=params.get("todo_id"),
            match=params.get("match"),
        )
        return {"success": True, "action": "reopen", "todo": todo}

    def _update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not any(key in params for key in ("title", "details", "due_at", "clear_due_at", "group", "clear_group")):
            return {
                "success": False,
                "error": "Update requires at least one field change: title, details, due_at, clear_due_at, group, or clear_group.",
            }

        todo = self._todos.update_todo(
            user=params.get("user"),
            todo_id=params.get("todo_id"),
            match=params.get("match"),
            title=params.get("title"),
            details=params.get("details"),
            due_at=params.get("due_at"),
            clear_due_at=params.get("clear_due_at", False),
            group=params.get("group"),
            clear_group=params.get("clear_group", False),
        )
        return {"success": True, "action": "update", "todo": todo}

    def _delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        todo = self._todos.delete_todo(
            user=params.get("user"),
            todo_id=params.get("todo_id"),
            match=params.get("match"),
        )
        return {"success": True, "action": "delete", "todo": todo}

    def _advance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        title = (params.get("title") or params.get("message") or "").strip()
        if not title:
            return {
                "success": False,
                "error": "Title is required for the next step.",
            }

        result = self._todos.advance_todo(
            user=params.get("user"),
            todo_id=params.get("todo_id"),
            match=params.get("match"),
            title=title,
            details=params.get("details"),
            due_at=params.get("due_at"),
        )
        return {
            "success": True,
            "action": "advance",
            "previous_todo": result["previous"],
            "todo": result["current"],
            "chain_id": result["chain_id"],
            "chain_position": result["chain_position"],
        }

    def _chain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        todo_id = params.get("todo_id")
        if not todo_id:
            return {
                "success": False,
                "error": "todo_id is required to view a chain.",
            }

        chain = self._todos.get_chain(
            user=params.get("user"),
            todo_id=todo_id,
        )
        return {
            "success": True,
            "action": "chain",
            "count": len(chain),
            "chain": chain,
        }
