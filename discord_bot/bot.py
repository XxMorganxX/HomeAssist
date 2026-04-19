"""
Discord bot that bridges a text channel to the HomeAssist orchestrator.

Features:
  - Listens in a single configured channel
  - Passes messages through TextOrchestrator (LLM + MCP tools)
  - Shows which tools the assistant invoked (replies to own message with schema)
  - Sends missed briefings on startup and subscribes to live briefing inserts/updates
"""

import asyncio
import contextlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import discord
from discord import app_commands

from assistant_framework.utils.todo_manager import TodoManager
from assistant_framework.utils.briefing.briefing_manager import BriefingManager, _substitute_time_placeholder
from discord_bot.text_orchestrator import TextOrchestrator

try:
    from supabase import acreate_client
    ASYNC_SUPABASE_AVAILABLE = True
except ImportError:
    acreate_client = None
    ASYNC_SUPABASE_AVAILABLE = False

try:
    from mcp_server.user_config import get_default_notification_user
except ImportError:
    def get_default_notification_user() -> str:
        return os.getenv("EMAIL_NOTIFICATION_RECIPIENT", "Morgan")

try:
    from mcp_server.clients.calendar_client import CalendarComponent
    from mcp_server.config import CALENDAR_USERS
    CALENDAR_SYNC_AVAILABLE = True
except ImportError:
    CalendarComponent = None
    CALENDAR_USERS = {}
    CALENDAR_SYNC_AVAILABLE = False

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
DISCORD_TODO_CHANNEL_ID = int(os.getenv("DISCORD_TODO_CHANNEL_ID", "0"))
DISCORD_BRIEFING_CHANNEL_ID = int(os.getenv("DISCORD_BRIEFING_CHANNEL_ID", "0"))
BRIEFING_USER = get_default_notification_user()
BRIEFING_BOOT_LIMIT = 25
BRIEFING_POLL_INTERVAL_SECONDS = 5
TODO_PAGE_SIZE = 15
TODO_CALENDAR_WINDOW_DAYS = 7
TODO_CALENDAR_SYNC_INTERVAL_SECONDS = 60
TODO_CALENDAR_CACHE_PATH = Path(__file__).resolve().parent / "state" / "calendar_todo_cache.json"
TODO_SURFACE_STATE_PATH = Path(__file__).resolve().parent / "state" / "todo_surface_state.json"
TODO_STATE_PREFIX = "\u2063"
TODO_STATE_ZERO = "\u200b"
TODO_STATE_ONE = "\u200c"
TODO_STATE_SUFFIX = "\u2064"

# Discord limits messages to 2000 chars
MAX_MESSAGE_LENGTH = 2000


def _truncate(text: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_briefing_text(briefing: Dict[str, object]) -> tuple[str, str]:
    """Resolve the best available Discord text and source metadata for a briefing."""
    opener = briefing.get("opener_text")
    if isinstance(opener, str) and opener:
        return opener, ""

    content = briefing.get("content") or {}
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = {"message": content}

    if not isinstance(content, dict):
        return "", ""

    message = content.get("message") or content.get("fact", "")
    source = (content.get("meta") or {}).get("source", "")
    if not isinstance(message, str):
        message = ""
    if not isinstance(source, str):
        source = ""
    return message, source


def _format_tool_calls(tool_calls: list) -> str:
    """Format tool calls as Discord subtext (-# renders as small/muted text)."""
    parts = []
    for tc in tool_calls:
        name = getattr(tc, "name", None) or "unknown"
        args = getattr(tc, "arguments", None) or {}
        try:
            args_str = json.dumps(args, separators=(",", ":"))
        except (TypeError, ValueError):
            args_str = str(args)
        parts.append(f"-# {name}({args_str})")
    return "\n".join(parts)


def _format_todo_help() -> str:
    """Return the help text for the dedicated todo channel."""
    return "\n".join(
        [
            "Todo channel shortcuts:",
            "- `!` or `!help` - Open or refresh the todo dashboard",
            "- `/todo add` - Add a todo item",
            "- `/todo list` - Open the paged todo list",
            "- `/todo due-today` - Open tasks due today",
            "- `/todo done` - Mark a todo completed by title or id",
            "- `/todo reopen` - Reopen a completed todo by title or id",
            "- `/todo delete` - Delete a todo by title or id",
            "",
            "You can use the dashboard buttons for the common flows without typing.",
        ]
    )


def _todo_log(event: str, **fields) -> None:
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    if details:
        print(f"🧭 [TodoUI] {event} {details}")
    else:
        print(f"🧭 [TodoUI] {event}")


class TodoCreateModal(discord.ui.Modal):
    """Modal for creating a todo from the dashboard UI."""

    def __init__(
        self,
        bot: "HomeAssistBot",
        *,
        source_kind: str,
        source_mode: str = "open",
        source_page: int = 0,
        source_message_id: Optional[int] = None,
    ):
        super().__init__(title="Add Todo")
        self.bot = bot
        self.source_kind = source_kind
        self.source_mode = source_mode
        self.source_page = source_page
        self.source_message_id = source_message_id

        self.todo_title = discord.ui.TextInput(
            label="Task title",
            placeholder="What should I add?",
            max_length=200,
        )
        self.todo_due = discord.ui.TextInput(
            label="Due time",
            placeholder="Optional: tomorrow 5pm",
            required=False,
            max_length=100,
        )
        self.todo_details = discord.ui.TextInput(
            label="Details",
            placeholder="Optional extra context",
            required=False,
            style=discord.TextStyle.paragraph,
            max_length=500,
        )
        self.add_item(self.todo_title)
        self.add_item(self.todo_due)
        self.add_item(self.todo_details)

    async def on_submit(self, interaction: discord.Interaction):
        if not self.bot._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        try:
            todo = self.bot._todo_manager.create_todo(
                user=BRIEFING_USER,
                title=str(self.todo_title.value).strip(),
                details=str(self.todo_details.value).strip() or None,
                due_at=str(self.todo_due.value).strip() or None,
                source_type="discord",
                source_id=str(interaction.id),
                source_metadata={
                    "channel_id": interaction.channel_id,
                    "guild_id": interaction.guild_id,
                    "created_via": "discord_modal",
                },
            )
        except Exception as exc:
            await interaction.response.send_message(f"Failed to add todo: {exc}", ephemeral=True)
            return

        status_message = self.bot._format_todo_confirmation("Added", todo)
        await interaction.response.send_message(status_message, ephemeral=True)
        await self.bot._refresh_todo_surface(
            channel=interaction.channel,
            source_kind=self.source_kind,
            source_mode=self.source_mode,
            source_page=self.source_page,
            source_message_id=self.source_message_id,
            status_message=status_message,
        )


class TodoDashboardButton(discord.ui.Button):
    """Persistent dashboard button."""

    def __init__(self, bot: "HomeAssistBot", action: str, label: str, style: discord.ButtonStyle):
        super().__init__(label=label, style=style, custom_id=f"todo:dashboard:{action}")
        self.bot = bot
        self.action = action

    async def callback(self, interaction: discord.Interaction):
        await self.bot._handle_dashboard_button(interaction, self.action)


class TodoDashboardView(discord.ui.View):
    """Persistent dashboard controls."""

    def __init__(self, bot: "HomeAssistBot"):
        super().__init__(timeout=None)
        self.add_item(TodoDashboardButton(bot, "add", "Add Todo", discord.ButtonStyle.success))
        self.add_item(TodoDashboardButton(bot, "open", "Open Todos", discord.ButtonStyle.primary))
        self.add_item(TodoDashboardButton(bot, "due", "Due Today", discord.ButtonStyle.primary))
        self.add_item(TodoDashboardButton(bot, "completed", "Completed", discord.ButtonStyle.secondary))
        self.add_item(TodoDashboardButton(bot, "refresh", "Refresh", discord.ButtonStyle.secondary))


class TodoListItemButton(discord.ui.Button):
    """A numbered todo action button."""

    def __init__(self, bot: "HomeAssistBot", slot_index: int):
        super().__init__(
            label=f"Task {slot_index + 1}",
            style=discord.ButtonStyle.secondary,
            row=slot_index // 5,
            disabled=True,
            custom_id=f"todo:list:item:{slot_index}",
        )
        self.bot = bot
        self.slot_index = slot_index

    async def callback(self, interaction: discord.Interaction):
        await self.bot._handle_todo_list_item(interaction, self.slot_index)


class TodoListNavButton(discord.ui.Button):
    """Shared navigation button for list views."""

    def __init__(self, bot: "HomeAssistBot", action: str, label: str, style: discord.ButtonStyle):
        super().__init__(label=label, style=style, row=3, custom_id=f"todo:list:{action}")
        self.bot = bot
        self.action = action

    async def callback(self, interaction: discord.Interaction):
        await self.bot._handle_todo_list_navigation(interaction, self.action)


class TodoListView(discord.ui.View):
    """Persistent paginated todo list controls."""

    def __init__(self, bot: "HomeAssistBot", *, mode: str = "open", page_todos: Optional[List[Dict[str, object]]] = None, page: int = 0, total_pages: int = 1):
        super().__init__(timeout=None)
        self.bot = bot
        self.item_buttons: List[TodoListItemButton] = []
        for slot_index in range(TODO_PAGE_SIZE):
            button = TodoListItemButton(bot, slot_index)
            self.item_buttons.append(button)
            self.add_item(button)

        self.prev_button = TodoListNavButton(bot, "prev", "Prev", discord.ButtonStyle.secondary)
        self.next_button = TodoListNavButton(bot, "next", "Next", discord.ButtonStyle.secondary)
        self.back_button = TodoListNavButton(bot, "back", "Back", discord.ButtonStyle.secondary)
        self.refresh_button = TodoListNavButton(bot, "refresh", "Refresh", discord.ButtonStyle.secondary)
        self.add_button = TodoListNavButton(bot, "add", "Add Todo", discord.ButtonStyle.success)
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self.add_item(self.back_button)
        self.add_item(self.refresh_button)
        self.add_item(self.add_button)

        self.configure(mode=mode, page_todos=page_todos or [], page=page, total_pages=total_pages)

    def configure(self, *, mode: str, page_todos: List[Dict[str, object]], page: int, total_pages: int) -> None:
        for slot_index, button in enumerate(self.item_buttons):
            if slot_index < len(page_todos):
                todo = page_todos[slot_index]
                verb = self.bot._todo_item_action_for(todo, mode)
                button.label = f"{verb} {slot_index + 1}"
                button.style = discord.ButtonStyle.success if verb == "Complete" else discord.ButtonStyle.primary
                button.disabled = False
            else:
                button.label = f"Task {slot_index + 1}"
                button.style = discord.ButtonStyle.secondary
                button.disabled = True

        self.prev_button.disabled = page <= 0 or total_pages <= 1
        self.next_button.disabled = total_pages <= 1 or page >= total_pages - 1


class HomeAssistBot(discord.Client):
    """Thin discord.Client subclass wired to the TextOrchestrator."""

    def __init__(self, orchestrator: TextOrchestrator, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.tree = app_commands.CommandTree(self)
        self.orchestrator = orchestrator
        self._briefing_manager: Optional[BriefingManager] = None
        self._todo_manager = TodoManager()
        self._todo_dashboard_message_id: Optional[int] = None
        self._briefing_task: Optional[asyncio.Task] = None
        self._briefings_in_flight: set[str] = set()
        self._calendar_sync_task: Optional[asyncio.Task] = None
        self._calendar_sync_loop_task: Optional[asyncio.Task] = None
        self._calendar_sync_lock = asyncio.Lock()
        self._calendar_cache: List[Dict[str, object]] = []
        self._calendar_cache_last_pull_at: Optional[str] = None
        self._todo_surface_state: Dict[str, Dict[str, object]] = {}
        self._response_lock = asyncio.Lock()
        self._registered_app_commands = False

        try:
            self._briefing_manager = BriefingManager()
        except Exception as e:
            print(f"⚠️  Briefing manager unavailable: {e}")

    async def setup_hook(self):
        if not self._registered_app_commands:
            self._register_todo_commands()
            self._registered_app_commands = True

        self.add_view(TodoDashboardView(self))
        self.add_view(TodoListView(self))

        try:
            synced = await self.tree.sync()
            print(f"✅ Synced {len(synced)} Discord app command(s)")
        except Exception as e:
            print(f"⚠️  Failed to sync Discord app commands: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self):
        print(f"✅ Discord bot logged in as {self.user}")
        self._load_todo_surface_state_from_disk()
        self._load_calendar_cache_from_disk()
        if DISCORD_CHANNEL_ID:
            print(f"📡 Listening in channel {DISCORD_CHANNEL_ID}")
            channel = self.get_channel(DISCORD_CHANNEL_ID)
            if channel:
                await channel.send("HomeAssist online.")
        else:
            print("⚠️  DISCORD_CHANNEL_ID not set -- bot will not respond to any channel")

        if DISCORD_TODO_CHANNEL_ID:
            print(f"✅ Todo slash commands restricted to channel {DISCORD_TODO_CHANNEL_ID}")
            todo_channel = self.get_channel(DISCORD_TODO_CHANNEL_ID)
            if todo_channel:
                removed = await self._cleanup_todo_channel_surfaces(todo_channel)
                if removed:
                    print(f"🧹 Removed {removed} stale todo surface message(s) on startup")
                await self._ensure_todo_dashboard_message(status_message="Todo dashboard online.", sync_calendar=False)
            if self._calendar_sync_loop_task is None:
                self._calendar_sync_loop_task = asyncio.create_task(self._calendar_sync_loop())
        else:
            print("⚠️  DISCORD_TODO_CHANNEL_ID not set -- /todo commands will work in any channel")

        if DISCORD_BRIEFING_CHANNEL_ID:
            print(f"📣 Briefings will post in channel {DISCORD_BRIEFING_CHANNEL_ID} for {BRIEFING_USER}")
        else:
            print("⚠️  DISCORD_BRIEFING_CHANNEL_ID not set -- proactive briefings are disabled")

        if self._briefing_manager and DISCORD_BRIEFING_CHANNEL_ID and self._briefing_task is None:
            self._briefing_task = asyncio.create_task(self._briefing_delivery_loop())

    def _register_todo_commands(self):
        todo_group = app_commands.Group(name="todo", description="Manage HomeAssist todos")

        @todo_group.command(name="add", description="Add a todo item")
        @app_commands.describe(title="Task title", due="Optional due time", details="Optional extra details")
        async def todo_add(interaction: discord.Interaction, title: str, due: Optional[str] = None, details: Optional[str] = None):
            if not await self._ensure_todo_channel(interaction):
                return
            if not self._todo_manager.is_available():
                await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
                return

            try:
                todo = self._todo_manager.create_todo(
                    user=BRIEFING_USER,
                    title=title,
                    details=details,
                    due_at=due,
                    source_type="discord",
                    source_id=str(interaction.id),
                    source_metadata={
                        "channel_id": interaction.channel_id,
                        "guild_id": interaction.guild_id,
                    },
                )
                await self._send_dashboard_response(
                    interaction,
                    status_message=self._format_todo_confirmation("Added", todo),
                )
            except Exception as exc:
                await interaction.response.send_message(f"Failed to add todo: {exc}", ephemeral=True)

        @todo_group.command(name="list", description="List open todos")
        @app_commands.describe(include_completed="Include completed tasks too")
        async def todo_list(interaction: discord.Interaction, include_completed: bool = False):
            if not await self._ensure_todo_channel(interaction):
                return
            await self._send_todo_list(interaction, include_completed=include_completed)

        @todo_group.command(name="done", description="Mark a todo as completed")
        @app_commands.describe(item="Todo title or id")
        async def todo_done(interaction: discord.Interaction, item: str):
            if not await self._ensure_todo_channel(interaction):
                return
            await self._mutate_todo(interaction, action="complete", item=item)

        @todo_group.command(name="reopen", description="Reopen a completed todo")
        @app_commands.describe(item="Todo title or id")
        async def todo_reopen(interaction: discord.Interaction, item: str):
            if not await self._ensure_todo_channel(interaction):
                return
            await self._mutate_todo(interaction, action="reopen", item=item)

        @todo_group.command(name="delete", description="Delete a todo")
        @app_commands.describe(item="Todo title or id")
        async def todo_delete(interaction: discord.Interaction, item: str):
            if not await self._ensure_todo_channel(interaction):
                return
            await self._mutate_todo(interaction, action="delete", item=item)

        @todo_group.command(name="due-today", description="Show todos due today")
        async def todo_due_today(interaction: discord.Interaction):
            if not await self._ensure_todo_channel(interaction):
                return
            await self._send_todo_list(interaction, only_due_today=True)

        self.tree.add_command(todo_group)

    async def _ensure_todo_channel(self, interaction: discord.Interaction) -> bool:
        if DISCORD_TODO_CHANNEL_ID and interaction.channel_id != DISCORD_TODO_CHANNEL_ID:
            await interaction.response.send_message(
                f"Use /todo commands in the configured todo channel ({DISCORD_TODO_CHANNEL_ID}).",
                ephemeral=True,
            )
            return False
        return True

    def _format_todo_confirmation(self, verb: str, todo: dict) -> str:
        suffix = f" (due {todo['due_display']})" if todo.get("due_display") else ""
        return f"{verb} todo: `{todo['title']}`{suffix}"

    def _todo_heading(self, mode: str) -> str:
        headings = {
            "open": "Open todos:",
            "due": "Todos due today:",
            "completed": "Completed todos:",
            "all": "All todos:",
        }
        return headings.get(mode, "Todos:")

    def _todo_source_label(self, todo: Dict[str, object]) -> str:
        source = str(todo.get("source_type") or "uncategorized")
        return source.replace("_", " ").title()

    def _truncate_component_text(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _clip_table_cell(self, value: object, width: int) -> str:
        text = str(value or "-")
        if len(text) <= width:
            return text.ljust(width)
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _format_due_table_cell(self, due_display: Optional[str], width: int) -> str:
        if not due_display:
            return self._clip_table_cell("-", width)

        text = str(due_display)
        if len(text) <= width:
            return text.ljust(width)

        parts = text.split()
        date_part = parts[0] if parts else text
        time_part = " ".join(parts[1:4]) if len(parts) >= 4 else " ".join(parts[1:])

        for candidate in (
            f"{date_part} {time_part}".strip(),
            date_part,
        ):
            if len(candidate) <= width:
                return candidate.ljust(width)

        return self._clip_table_cell(date_part, width)

    def _summarize_todo_for_component(self, todo: Dict[str, object]) -> str:
        parts = [self._todo_source_label(todo)]
        if todo.get("due_display"):
            parts.insert(0, f"Due {todo['due_display']}")
        return self._truncate_component_text(" | ".join(parts), 100)

    def _todo_item_action_for(self, todo: Dict[str, object], mode: str) -> str:
        if mode == "completed":
            return "Reopen"
        if mode == "all":
            return "Reopen" if todo.get("completed") else "Complete"
        return "Complete"

    def _encode_hidden_state(self, raw: str) -> str:
        bits = "".join(format(byte, "08b") for byte in raw.encode("utf-8"))
        payload = bits.replace("0", TODO_STATE_ZERO).replace("1", TODO_STATE_ONE)
        return f"{TODO_STATE_PREFIX}{payload}{TODO_STATE_SUFFIX}"

    def _decode_hidden_state(self, payload: str) -> Optional[str]:
        if len(payload) % 8 != 0:
            return None
        bits = payload.replace(TODO_STATE_ZERO, "0").replace(TODO_STATE_ONE, "1")
        try:
            byte_values = [int(bits[index:index + 8], 2) for index in range(0, len(bits), 8)]
            return bytes(byte_values).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None

    def _load_todo_surface_state_from_disk(self) -> None:
        if not TODO_SURFACE_STATE_PATH.exists():
            return

        try:
            with open(TODO_SURFACE_STATE_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            states = payload.get("states", {})
            if isinstance(states, dict):
                self._todo_surface_state = states
            _todo_log("todo_surface_state_loaded", entries=len(self._todo_surface_state))
        except Exception as exc:
            _todo_log("todo_surface_state_load_failed", error=str(exc))

    def _write_todo_surface_state_to_disk(self) -> None:
        try:
            TODO_SURFACE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TODO_SURFACE_STATE_PATH, "w", encoding="utf-8") as handle:
                json.dump({"states": self._todo_surface_state}, handle, indent=2)
        except Exception as exc:
            _todo_log("todo_surface_state_write_failed", error=str(exc))

    def _store_todo_surface_state(self, todo_ids: List[str]) -> str:
        token = uuid4().hex[:12]
        self._todo_surface_state[token] = {
            "todo_ids": list(todo_ids),
            "updated_at": datetime.now().isoformat(),
        }
        if len(self._todo_surface_state) > 100:
            sorted_items = sorted(
                self._todo_surface_state.items(),
                key=lambda item: str(item[1].get("updated_at") or ""),
            )
            self._todo_surface_state = dict(sorted_items[-100:])
        self._write_todo_surface_state_to_disk()
        _todo_log("todo_surface_state_stored", token=token, todo_ids_count=len(todo_ids))
        return token

    def _resolve_todo_ids_state(self, raw_ids: str) -> List[str]:
        if raw_ids.startswith("ref:"):
            token = raw_ids[4:]
            state = self._todo_surface_state.get(token, {})
            todo_ids = state.get("todo_ids", [])
            if isinstance(todo_ids, list):
                _todo_log("todo_surface_state_resolved", token=token, todo_ids_count=len(todo_ids))
                return [str(todo_id) for todo_id in todo_ids if todo_id]
            _todo_log("todo_surface_state_missing", token=token)
            return []
        return [todo_id for todo_id in raw_ids.split(",") if todo_id]

    def _attach_hidden_state(self, content: str, *, kind: str, mode: Optional[str] = None, page: int = 0, todo_ids: Optional[List[str]] = None) -> str:
        ids_text = ""
        if todo_ids:
            ids_text = f"ref:{self._store_todo_surface_state([str(todo_id) for todo_id in todo_ids])}"
        raw = f"{kind}|{mode or 'open'}|{page}|{ids_text}"
        hidden_state = self._encode_hidden_state(raw)
        visible_limit = max(1, MAX_MESSAGE_LENGTH - len(hidden_state) - 1)
        _todo_log(
            "attach_hidden_state",
            kind=kind,
            mode=mode or "open",
            page=page,
            todo_ids_count=len(todo_ids or []),
            hidden_state_length=len(hidden_state),
            visible_limit=visible_limit,
            content_length=len(content),
        )
        return f"{_truncate(content, limit=visible_limit)}\n{hidden_state}"

    def _parse_todo_message_state(self, content: str) -> Dict[str, object]:
        start = content.rfind(TODO_STATE_PREFIX)
        end = content.rfind(TODO_STATE_SUFFIX)
        if start != -1 and end != -1 and end > start:
            raw = self._decode_hidden_state(content[start + len(TODO_STATE_PREFIX):end])
            if raw:
                parts = raw.split("|", 3)
                if len(parts) == 4:
                    kind, mode, raw_page, raw_ids = parts
                    try:
                        page = max(int(raw_page), 0)
                    except ValueError:
                        page = 0
                    todo_ids = self._resolve_todo_ids_state(raw_ids)
                    return {"kind": kind, "mode": mode, "page": page, "todo_ids": todo_ids}
        return {"kind": "dashboard", "mode": "open", "page": 0, "todo_ids": []}

    def _get_todos_for_mode(self, mode: str) -> List[Dict[str, object]]:
        cached_calendar_todos = self._get_cached_calendar_todos()
        _todo_log("get_todos_for_mode_start", mode=mode, cached_calendar_items=len(cached_calendar_todos))
        if mode == "due":
            due_today = [
                todo
                for todo in self._todo_manager.list_todos(
                    user=BRIEFING_USER,
                    include_completed=False,
                    limit=100,
                    only_due_today=True,
                )
                if str(todo.get("source_type") or "") != "calendar"
            ]
            today = datetime.now(self._todo_manager._tz).date()
            due_today.extend(
                todo
                for todo in cached_calendar_todos
                if (self._todo_manager._parse_due_from_row(todo) and self._todo_manager._parse_due_from_row(todo).astimezone(self._todo_manager._tz).date() == today)
            )
            sorted_todos = self._sort_todos_for_ui(due_today)
            _todo_log("get_todos_for_mode_done", mode=mode, count=len(sorted_todos))
            return sorted_todos
        if mode == "completed":
            todos = [
                todo
                for todo in self._todo_manager.list_todos(
                    user=BRIEFING_USER,
                    include_completed=True,
                    limit=100,
                )
                if todo.get("completed")
            ]
            _todo_log("get_todos_for_mode_done", mode=mode, count=len(todos))
            return todos
        if mode == "all":
            all_todos = [
                todo
                for todo in self._todo_manager.list_todos(
                    user=BRIEFING_USER,
                    include_completed=True,
                    limit=100,
                )
                if str(todo.get("source_type") or "") != "calendar"
            ]
            all_todos.extend(cached_calendar_todos)
            sorted_todos = self._sort_todos_for_ui(all_todos)
            _todo_log("get_todos_for_mode_done", mode=mode, count=len(sorted_todos))
            return sorted_todos
        regular_open = [
            todo
            for todo in self._todo_manager.list_todos(
                user=BRIEFING_USER,
                include_completed=False,
                limit=100,
            )
            if str(todo.get("source_type") or "") != "calendar"
        ]
        regular_open.extend(cached_calendar_todos)
        sorted_todos = self._sort_todos_for_ui(regular_open)
        _todo_log("get_todos_for_mode_done", mode=mode, count=len(sorted_todos))
        return sorted_todos

    def _is_calendar_todo_in_window(self, todo: Dict[str, object], *, days: int) -> bool:
        if str(todo.get("source_type") or "") != "calendar":
            return False

        due_dt = self._todo_manager._parse_due_from_row(todo)
        if due_dt is None:
            return False

        due_local = due_dt.astimezone(self._todo_manager._tz)
        now_local = datetime.now(self._todo_manager._tz)
        latest_local = now_local + timedelta(days=days)
        return now_local <= due_local <= latest_local

    def _merge_discord_open_todos(self, todos: List[Dict[str, object]]) -> List[Dict[str, object]]:
        merged: List[Dict[str, object]] = []
        for todo in todos:
            if str(todo.get("source_type") or "") == "calendar":
                if self._is_calendar_todo_in_window(todo, days=TODO_CALENDAR_WINDOW_DAYS):
                    merged.append(todo)
                continue
            merged.append(todo)
        return merged

    def _sort_todos_for_ui(self, todos: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return sorted(
            todos,
            key=lambda todo: (
                1 if todo.get("completed") else 0,
                todo.get("due_at") or "9999-12-31T23:59:59+00:00",
                str(todo.get("title") or "").lower(),
            ),
        )

    def _paginate_todos(self, todos: List[Dict[str, object]], requested_page: int) -> tuple[List[Dict[str, object]], int, int]:
        total_pages = max((len(todos) - 1) // TODO_PAGE_SIZE + 1, 1)
        page = min(max(requested_page, 0), total_pages - 1)
        start = page * TODO_PAGE_SIZE
        end = start + TODO_PAGE_SIZE
        return todos[start:end], page, total_pages

    def _format_todo_table_page(self, todos: List[Dict[str, object]], mode: str) -> str:
        include_status = mode in {"completed", "all"}
        num_width = 3
        source_width = 10
        due_width = 16
        title_width = 40 if include_status else 47
        headers = [
            self._clip_table_cell("#", num_width),
            self._clip_table_cell("Title", title_width),
            self._clip_table_cell("Source", source_width),
            self._clip_table_cell("Due", due_width),
        ]
        divider_parts = [
            "-" * num_width,
            "-" * title_width,
            "-" * source_width,
            "-" * due_width,
        ]
        if include_status:
            headers.insert(1, self._clip_table_cell("Done", 6))
            divider_parts.insert(1, "-" * 6)

        header = " | ".join(headers)
        divider = "-+-".join(divider_parts)

        rows = [header, divider]
        for slot_index, todo in enumerate(todos, start=1):
            row = [
                self._clip_table_cell(slot_index, num_width),
                self._clip_table_cell(todo.get("title") or "Untitled todo", title_width),
                self._clip_table_cell(self._todo_source_label(todo), source_width),
                self._format_due_table_cell(todo.get("due_display"), due_width),
            ]
            if include_status:
                row.insert(1, self._clip_table_cell("[x]" if todo.get("completed") else "[ ]", 6))
            rows.append(" | ".join(row))

        for slot_index in range(len(todos) + 1, TODO_PAGE_SIZE + 1):
            row = [
                self._clip_table_cell(slot_index, num_width),
                self._clip_table_cell("-", title_width),
                self._clip_table_cell("-", source_width),
                self._clip_table_cell("-", due_width),
            ]
            if include_status:
                row.insert(1, self._clip_table_cell("-", 6))
            rows.append(" | ".join(row))

        return "```\n" + "\n".join(rows) + "\n```"

    def _render_dashboard_content(self, *, status_message: Optional[str] = None) -> str:
        if not self._todo_manager.is_available():
            lines = [
                "Todo dashboard",
                "Supabase is not configured for todos, so the dashboard cannot load tasks yet.",
                "",
                _format_todo_help(),
            ]
            return self._attach_hidden_state("\n".join(lines), kind="dashboard")

        open_todos = self._get_todos_for_mode("open")
        due_today = self._get_todos_for_mode("due")
        completed = self._get_todos_for_mode("completed")
        lines = [
            f"Todo dashboard for `{BRIEFING_USER}`",
            "Use the buttons below for the common actions. Typed commands still work if you prefer:",
            "- `!` or `!help` refreshes this dashboard",
            "- `/todo add` opens the add flow from text",
            "- `/todo list` opens the paged todo list",
            "- `/todo due-today` opens tasks due today",
            "- `/todo done`, `/todo reopen`, `/todo delete` still work by title or id",
            "",
            f"Open: {len(open_todos)}",
            f"Due today: {len(due_today)}",
            f"Completed: {len(completed)}",
        ]
        if status_message:
            lines.insert(1, status_message)
        return self._attach_hidden_state("\n".join(lines), kind="dashboard")

    def _render_list_content(self, mode: str, page: int, *, status_message: Optional[str] = None) -> tuple[str, List[Dict[str, object]], int, int]:
        todos = self._get_todos_for_mode(mode)
        page_todos, page, total_pages = self._paginate_todos(todos, page)
        _todo_log(
            "render_list_content",
            mode=mode,
            page=page,
            total_todos=len(todos),
            page_todos=len(page_todos),
            total_pages=total_pages,
            cache_items=len(self._calendar_cache),
            cache_last_pull_at=self._calendar_cache_last_pull_at,
        )

        lines = [
            self._todo_heading(mode),
            "Use the numbered buttons below to update the matching row.",
        ]
        if mode == "open":
            lines.append(f"Including calendar items from the next {TODO_CALENDAR_WINDOW_DAYS} days.")

        if status_message:
            lines.append(status_message)

        if not page_todos:
            lines.append("No matching todos.")
        else:
            lines.append(self._format_todo_table_page(page_todos, mode))

        lines.extend(
            [
                "",
                f"Page {page + 1} of {total_pages}",
            ]
        )
        content = self._attach_hidden_state(
            "\n".join(lines),
            kind="list",
            mode=mode,
            page=page,
            todo_ids=[str(todo["id"]) for todo in page_todos],
        )
        return content, page_todos, page, total_pages

    def _build_dashboard_view(self) -> TodoDashboardView:
        return TodoDashboardView(self)

    def _build_list_view(self, *, mode: str, page_todos: List[Dict[str, object]], page: int, total_pages: int) -> TodoListView:
        return TodoListView(self, mode=mode, page_todos=page_todos, page=page, total_pages=total_pages)

    def _load_calendar_cache_from_disk(self) -> None:
        if not TODO_CALENDAR_CACHE_PATH.exists():
            print("ℹ️  Calendar todo cache file not found on boot")
            return

        try:
            with open(TODO_CALENDAR_CACHE_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._calendar_cache = payload.get("calendar_todos", [])
            self._calendar_cache_last_pull_at = payload.get("last_pull_at")
            print(
                f"✅ Loaded calendar todo cache: {len(self._calendar_cache)} item(s)"
                + (f" from {self._calendar_cache_last_pull_at}" if self._calendar_cache_last_pull_at else "")
            )
            _todo_log("calendar_cache_loaded", items=len(self._calendar_cache), last_pull_at=self._calendar_cache_last_pull_at)
        except Exception as exc:
            print(f"⚠️  Failed to load calendar todo cache: {exc}")
            _todo_log("calendar_cache_load_failed", error=str(exc))

    def _write_calendar_cache_to_disk(self, *, calendar_todos: List[Dict[str, object]], sync_summary: str) -> None:
        try:
            TODO_CALENDAR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_pull_at": datetime.now().isoformat(),
                "sync_summary": sync_summary,
                "calendar_todos": calendar_todos,
            }
            with open(TODO_CALENDAR_CACHE_PATH, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self._calendar_cache = calendar_todos
            self._calendar_cache_last_pull_at = payload["last_pull_at"]
            print(
                f"✅ Cached {len(calendar_todos)} calendar todo item(s)"
                f" at {self._calendar_cache_last_pull_at}"
            )
            _todo_log("calendar_cache_written", items=len(calendar_todos), last_pull_at=self._calendar_cache_last_pull_at)
        except Exception as exc:
            print(f"⚠️  Failed to write calendar todo cache: {exc}")
            _todo_log("calendar_cache_write_failed", error=str(exc))

    def _get_cached_calendar_todos(self) -> List[Dict[str, object]]:
        return [dict(todo) for todo in self._calendar_cache]

    def _calendar_cache_last_pull_datetime(self) -> Optional[datetime]:
        if not self._calendar_cache_last_pull_at:
            return None
        try:
            return datetime.fromisoformat(self._calendar_cache_last_pull_at)
        except ValueError:
            return None

    def _sync_calendar_todos_blocking(self) -> str:
        if not CALENDAR_SYNC_AVAILABLE:
            return "Calendar sync unavailable."
        if not self._todo_manager.is_available():
            return "Supabase unavailable."

        users = list(CALENDAR_USERS.keys())
        if not users:
            return "No configured calendars."

        total_synced = 0
        total_removed = 0
        total_failed = 0
        processed_users = 0

        for user in users:
            try:
                calendar = CalendarComponent(user=user)
                user_config = CALENDAR_USERS.get(user, {})
                calendar_id = user_config.get("calendar_id", "primary")
                now = datetime.now(self._todo_manager._tz).astimezone()
                end_time = now + timedelta(days=TODO_CALENDAR_WINDOW_DAYS)
                events = calendar.get_events(
                    num_events=50,
                    time_min=now.isoformat(),
                    time_max=end_time.isoformat(),
                    calendar_id=calendar_id,
                )
                formatted_events = [calendar.format_event(event) for event in events]
                result = self._todo_manager.sync_calendar_events(calendar_user=user, events=formatted_events)
                total_synced += result.get("synced", 0)
                total_removed += result.get("removed", 0)
                total_failed += result.get("failed", 0)
                processed_users += 1
            except Exception as exc:
                total_failed += 1
                print(f"⚠️  Calendar sync failed for {user}: {exc}")

        sync_summary = f"Calendar sync: {total_synced} events from {processed_users} calendar(s)."
        if total_removed:
            sync_summary += f" Removed {total_removed} stale event(s)."
        if total_failed:
            sync_summary += f" {total_failed} failure(s)."
        cached_calendar_todos = [
            todo
            for todo in self._todo_manager.list_todos(
                user=BRIEFING_USER,
                include_completed=False,
                limit=250,
                source_type="calendar",
            )
            if self._is_calendar_todo_in_window(todo, days=TODO_CALENDAR_WINDOW_DAYS)
        ]
        self._write_calendar_cache_to_disk(
            calendar_todos=cached_calendar_todos,
            sync_summary=sync_summary,
        )
        return sync_summary

    async def _refresh_current_todo_surface(self) -> None:
        if not DISCORD_TODO_CHANNEL_ID or not self._todo_dashboard_message_id:
            _todo_log("refresh_current_surface_skipped", reason="missing_channel_or_surface", message_id=self._todo_dashboard_message_id)
            return

        channel = self.get_channel(DISCORD_TODO_CHANNEL_ID)
        if channel is None:
            _todo_log("refresh_current_surface_skipped", reason="channel_not_found", channel_id=DISCORD_TODO_CHANNEL_ID)
            return

        try:
            message = await channel.fetch_message(self._todo_dashboard_message_id)
        except Exception as exc:
            _todo_log("refresh_current_surface_fetch_failed", message_id=self._todo_dashboard_message_id, error=str(exc))
            self._todo_dashboard_message_id = None
            try:
                await self._post_todo_surface(
                    channel=channel,
                    content=self._render_dashboard_content(),
                    view=self._build_dashboard_view(),
                )
                _todo_log("refresh_current_surface_reposted_dashboard")
            except Exception as repost_exc:
                _todo_log("refresh_current_surface_repost_failed", error=str(repost_exc))
            return

        state = self._parse_todo_message_state(message.content)
        _todo_log("refresh_current_surface", message_id=message.id, state=state)
        if state.get("kind") == "list":
            mode = str(state.get("mode") or "open")
            page = int(state.get("page") or 0)
            content, page_todos, page, total_pages = self._render_list_content(mode, page)
            try:
                await message.edit(
                    content=content,
                    view=self._build_list_view(mode=mode, page_todos=page_todos, page=page, total_pages=total_pages),
                )
                _todo_log("refresh_current_surface_list_edited", message_id=message.id, mode=mode, page=page, page_todos=len(page_todos))
            except Exception as exc:
                _todo_log("refresh_current_surface_list_edit_failed", message_id=message.id, mode=mode, page=page, error=str(exc))
            return

        try:
            await message.edit(
                content=self._render_dashboard_content(),
                view=self._build_dashboard_view(),
            )
            _todo_log("refresh_current_surface_dashboard_edited", message_id=message.id)
        except Exception as exc:
            _todo_log("refresh_current_surface_dashboard_edit_failed", message_id=message.id, error=str(exc))

    async def _run_calendar_sync(self, *, refresh_surface: bool) -> None:
        _todo_log("calendar_sync_started", refresh_surface=refresh_surface)
        try:
            async with self._calendar_sync_lock:
                await asyncio.to_thread(self._sync_calendar_todos_blocking)
        except Exception as exc:
            print(f"⚠️  Background calendar sync failed: {exc}")
            _todo_log("calendar_sync_failed", error=str(exc))
        else:
            _todo_log("calendar_sync_finished", refresh_surface=refresh_surface)
            if refresh_surface:
                await self._refresh_current_todo_surface()
        finally:
            self._calendar_sync_task = None

    def _schedule_calendar_sync(self, *, force: bool = False, refresh_surface: bool = True) -> bool:
        if not CALENDAR_SYNC_AVAILABLE or not self._todo_manager.is_available():
            _todo_log("calendar_sync_not_scheduled", reason="unavailable")
            return False

        if self._calendar_sync_task and not self._calendar_sync_task.done():
            _todo_log("calendar_sync_not_scheduled", reason="already_running")
            return False

        now = datetime.now()
        last_pull = self._calendar_cache_last_pull_datetime()
        if not force and last_pull and (now - last_pull) < timedelta(seconds=TODO_CALENDAR_SYNC_INTERVAL_SECONDS):
            _todo_log("calendar_sync_not_scheduled", reason="fresh_cache", last_pull_at=self._calendar_cache_last_pull_at)
            return False

        self._calendar_sync_task = asyncio.create_task(
            self._run_calendar_sync(refresh_surface=refresh_surface)
        )
        _todo_log("calendar_sync_scheduled", force=force, refresh_surface=refresh_surface)
        return True

    async def _calendar_sync_loop(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed():
            self._schedule_calendar_sync(force=True, refresh_surface=True)
            await asyncio.sleep(TODO_CALENDAR_SYNC_INTERVAL_SECONDS)

    async def _delete_todo_surface_message(self, channel, message_id: Optional[int]) -> None:
        if channel is None or not message_id:
            return
        try:
            message = await channel.fetch_message(message_id)
            await message.delete()
            _todo_log("delete_surface_success", message_id=message_id)
        except Exception as exc:
            _todo_log("delete_surface_failed", message_id=message_id, error=str(exc))

    def _is_managed_todo_surface(self, message: discord.Message) -> bool:
        if message.author != self.user:
            return False
        kind = self._parse_todo_message_state(message.content).get("kind")
        return kind in {"dashboard", "list"}

    async def _collect_stale_todo_surface_ids(
        self,
        channel,
        *,
        keep_message_id: int,
        extra_keep_ids: Optional[set[int]] = None,
        history_limit: int = 25,
    ) -> set[int]:
        keep_ids = {keep_message_id}
        if extra_keep_ids:
            keep_ids.update(extra_keep_ids)

        stale_ids: set[int] = set()
        async for candidate in channel.history(limit=history_limit):
            if candidate.id in keep_ids:
                continue
            if self._is_managed_todo_surface(candidate):
                stale_ids.add(candidate.id)
        return stale_ids

    async def _cleanup_todo_channel_surfaces(self, channel, *, history_limit: int = 50) -> int:
        if channel is None:
            _todo_log("cleanup_surfaces_skipped", reason="no_channel")
            return 0

        removed = 0
        async for candidate in channel.history(limit=history_limit):
            if self._is_managed_todo_surface(candidate):
                try:
                    await candidate.delete()
                    removed += 1
                except Exception as exc:
                    _todo_log("cleanup_surface_delete_failed", message_id=candidate.id, error=str(exc))

        self._todo_dashboard_message_id = None
        _todo_log("cleanup_surfaces_finished", removed=removed, history_limit=history_limit)
        return removed

    async def _post_todo_surface(
        self,
        *,
        channel,
        content: str,
        view: discord.ui.View,
        replace_message_id: Optional[int] = None,
    ) -> Optional[discord.Message]:
        if channel is None:
            _todo_log("post_surface_skipped", reason="no_channel")
            return None

        previous_message_id = self._todo_dashboard_message_id
        try:
            message = await channel.send(content, view=view, silent=True)
        except Exception as exc:
            _todo_log("post_surface_failed", replace_message_id=replace_message_id, previous_message_id=previous_message_id, error=str(exc))
            return None

        self._todo_dashboard_message_id = message.id
        _todo_log("post_surface_sent", message_id=message.id, replace_message_id=replace_message_id, previous_message_id=previous_message_id)

        stale_ids = {
            old_id
            for old_id in (previous_message_id, replace_message_id)
            if old_id and old_id != message.id
        }
        stale_ids.update(
            await self._collect_stale_todo_surface_ids(
                channel,
                keep_message_id=message.id,
            )
        )
        for stale_id in stale_ids:
            await self._delete_todo_surface_message(channel, stale_id)

        return message

    async def _send_dashboard_response(self, interaction: discord.Interaction, *, status_message: Optional[str] = None) -> None:
        _todo_log(
            "send_dashboard_response",
            interaction_id=getattr(interaction, "id", None),
            channel_id=getattr(interaction.channel, "id", None),
            status_message=bool(status_message),
        )
        await interaction.response.defer()
        await self._post_todo_surface(
            channel=interaction.channel,
            content=self._render_dashboard_content(status_message=status_message),
            view=self._build_dashboard_view(),
        )

    async def _send_list_response(self, interaction: discord.Interaction, *, mode: str, page: int = 0, status_message: Optional[str] = None, sync_calendar: bool = False) -> None:
        _todo_log(
            "send_list_response_start",
            interaction_id=getattr(interaction, "id", None),
            channel_id=getattr(interaction.channel, "id", None),
            mode=mode,
            page=page,
            sync_calendar=sync_calendar,
            status_message=bool(status_message),
        )
        if sync_calendar and mode in {"open", "due", "all"}:
            self._schedule_calendar_sync(refresh_surface=True)
        await interaction.response.defer()
        content, page_todos, page, total_pages = self._render_list_content(mode, page, status_message=status_message)
        await self._post_todo_surface(
            channel=interaction.channel,
            content=content,
            view=self._build_list_view(mode=mode, page_todos=page_todos, page=page, total_pages=total_pages),
        )

    async def _edit_interaction_to_dashboard(self, interaction: discord.Interaction, *, status_message: Optional[str] = None, sync_calendar: bool = False) -> None:
        _todo_log(
            "edit_to_dashboard_start",
            interaction_id=getattr(interaction, "id", None),
            message_id=getattr(interaction.message, "id", None) if interaction.message else None,
            sync_calendar=sync_calendar,
            status_message=bool(status_message),
        )
        if sync_calendar:
            self._schedule_calendar_sync(refresh_surface=True)
        await interaction.response.defer()
        if interaction.message is not None:
            try:
                await interaction.message.edit(
                    content=self._render_dashboard_content(status_message=status_message),
                    view=self._build_dashboard_view(),
                )
                self._todo_dashboard_message_id = interaction.message.id
                _todo_log("edit_to_dashboard_in_place", message_id=interaction.message.id)
                return
            except Exception as exc:
                _todo_log("edit_to_dashboard_in_place_failed", message_id=interaction.message.id, error=str(exc))
        await self._post_todo_surface(
            channel=interaction.channel,
            content=self._render_dashboard_content(status_message=status_message),
            view=self._build_dashboard_view(),
            replace_message_id=interaction.message.id if interaction.message else None,
        )

    async def _edit_interaction_to_list(self, interaction: discord.Interaction, *, mode: str, page: int = 0, status_message: Optional[str] = None, sync_calendar: bool = False) -> None:
        _todo_log(
            "edit_to_list_start",
            interaction_id=getattr(interaction, "id", None),
            message_id=getattr(interaction.message, "id", None) if interaction.message else None,
            mode=mode,
            page=page,
            sync_calendar=sync_calendar,
            status_message=bool(status_message),
        )
        if sync_calendar and mode in {"open", "due", "all"}:
            self._schedule_calendar_sync(refresh_surface=True)
        await interaction.response.defer()
        content, page_todos, page, total_pages = self._render_list_content(mode, page, status_message=status_message)
        if interaction.message is not None:
            try:
                await interaction.message.edit(
                    content=content,
                    view=self._build_list_view(mode=mode, page_todos=page_todos, page=page, total_pages=total_pages),
                )
                self._todo_dashboard_message_id = interaction.message.id
                _todo_log("edit_to_list_in_place", message_id=interaction.message.id, mode=mode, page=page, page_todos=len(page_todos))
                return
            except Exception as exc:
                _todo_log("edit_to_list_in_place_failed", message_id=interaction.message.id, mode=mode, page=page, error=str(exc))
        await self._post_todo_surface(
            channel=interaction.channel,
            content=content,
            view=self._build_list_view(mode=mode, page_todos=page_todos, page=page, total_pages=total_pages),
            replace_message_id=interaction.message.id if interaction.message else None,
        )

    async def _refresh_todo_surface(
        self,
        *,
        channel,
        source_kind: str,
        source_mode: str,
        source_page: int,
        source_message_id: Optional[int],
        status_message: Optional[str] = None,
    ) -> None:
        if channel is None:
            return

        if source_kind == "dashboard":
            if source_message_id:
                try:
                    message = await channel.fetch_message(source_message_id)
                    await message.edit(
                        content=self._render_dashboard_content(status_message=status_message),
                        view=self._build_dashboard_view(),
                    )
                    self._todo_dashboard_message_id = message.id
                    return
                except Exception:
                    pass
            await self._ensure_todo_dashboard_message(status_message=status_message)
            return

        content, page_todos, source_page, total_pages = self._render_list_content(
            source_mode,
            source_page,
            status_message=status_message,
        )
        if source_message_id:
            try:
                message = await channel.fetch_message(source_message_id)
                await message.edit(
                    content=content,
                    view=self._build_list_view(
                        mode=source_mode,
                        page_todos=page_todos,
                        page=source_page,
                        total_pages=total_pages,
                    ),
                )
                self._todo_dashboard_message_id = message.id
                return
            except Exception:
                pass
        await self._post_todo_surface(
            channel=channel,
            content=content,
            view=self._build_list_view(
                mode=source_mode,
                page_todos=page_todos,
                page=source_page,
                total_pages=total_pages,
            ),
            replace_message_id=source_message_id,
        )

    async def _find_todo_dashboard_message(self, channel) -> Optional[discord.Message]:
        if self._todo_dashboard_message_id:
            try:
                message = await channel.fetch_message(self._todo_dashboard_message_id)
                if message.author == self.user and self._parse_todo_message_state(message.content).get("kind") == "dashboard":
                    return message
            except Exception:
                self._todo_dashboard_message_id = None

        async for candidate in channel.history(limit=50):
            if candidate.author == self.user and self._parse_todo_message_state(candidate.content).get("kind") == "dashboard":
                self._todo_dashboard_message_id = candidate.id
                return candidate
        return None

    async def _ensure_todo_dashboard_message(self, *, status_message: Optional[str] = None, sync_calendar: bool = False) -> Optional[discord.Message]:
        if not DISCORD_TODO_CHANNEL_ID:
            return None

        channel = self.get_channel(DISCORD_TODO_CHANNEL_ID)
        if channel is None:
            return None

        if sync_calendar:
            self._schedule_calendar_sync(refresh_surface=True)

        existing = await self._find_todo_dashboard_message(channel)
        if existing is not None:
            try:
                await existing.edit(
                    content=self._render_dashboard_content(status_message=status_message),
                    view=self._build_dashboard_view(),
                )
                self._todo_dashboard_message_id = existing.id
                _todo_log("ensure_dashboard_edited_existing", message_id=existing.id, status_message=bool(status_message))
                return existing
            except Exception as exc:
                _todo_log("ensure_dashboard_edit_existing_failed", message_id=existing.id, error=str(exc))
        return await self._post_todo_surface(
            channel=channel,
            content=self._render_dashboard_content(status_message=status_message),
            view=self._build_dashboard_view(),
            replace_message_id=existing.id if existing else None,
        )

    async def _handle_dashboard_button(self, interaction: discord.Interaction, action: str) -> None:
        _todo_log(
            "dashboard_button_clicked",
            action=action,
            interaction_id=getattr(interaction, "id", None),
            message_id=getattr(interaction.message, "id", None) if interaction.message else None,
        )
        if action == "add":
            state = self._parse_todo_message_state(interaction.message.content if interaction.message else "")
            _todo_log("dashboard_button_add_state", state=state)
            await interaction.response.send_modal(
                TodoCreateModal(
                    self,
                    source_kind=str(state.get("kind") or "dashboard"),
                    source_mode=str(state.get("mode") or "open"),
                    source_page=int(state.get("page") or 0),
                    source_message_id=interaction.message.id if interaction.message else None,
                )
            )
            return

        if not self._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        if action == "open":
            await self._edit_interaction_to_list(interaction, mode="open", page=0)
            return
        if action == "due":
            await self._edit_interaction_to_list(interaction, mode="due", page=0)
            return
        if action == "completed":
            await self._edit_interaction_to_list(interaction, mode="completed", page=0)
            return
        await self._edit_interaction_to_dashboard(interaction)

    async def _handle_todo_list_navigation(self, interaction: discord.Interaction, action: str) -> None:
        state = self._parse_todo_message_state(interaction.message.content if interaction.message else "")
        mode = str(state.get("mode") or "open")
        page = int(state.get("page") or 0)
        _todo_log(
            "list_navigation_clicked",
            action=action,
            interaction_id=getattr(interaction, "id", None),
            message_id=getattr(interaction.message, "id", None) if interaction.message else None,
            state=state,
        )

        if action == "back":
            await self._edit_interaction_to_dashboard(interaction)
            return
        if action == "add":
            await interaction.response.send_modal(
                TodoCreateModal(
                    self,
                    source_kind="list",
                    source_mode=mode,
                    source_page=page,
                    source_message_id=interaction.message.id if interaction.message else None,
                )
            )
            return

        if not self._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        if action == "prev":
            page -= 1
        elif action == "next":
            page += 1
        await self._edit_interaction_to_list(
            interaction,
            mode=mode,
            page=page,
        )

    async def _handle_todo_list_item(self, interaction: discord.Interaction, slot_index: int) -> None:
        if not self._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        state = self._parse_todo_message_state(interaction.message.content if interaction.message else "")
        mode = str(state.get("mode") or "open")
        page = int(state.get("page") or 0)
        todo_ids = list(state.get("todo_ids") or [])
        _todo_log(
            "list_item_clicked",
            slot_index=slot_index,
            interaction_id=getattr(interaction, "id", None),
            message_id=getattr(interaction.message, "id", None) if interaction.message else None,
            state=state,
            todo_ids_count=len(todo_ids),
        )
        if slot_index >= len(todo_ids):
            await interaction.response.send_message("That todo is no longer available on this page.", ephemeral=True)
            return

        todo_id = str(todo_ids[slot_index])
        current_todos = {str(todo["id"]): todo for todo in self._get_todos_for_mode(mode)}
        current_todo = current_todos.get(todo_id)
        if current_todo is None:
            await interaction.response.send_message("That todo no longer matches this view. Try Refresh.", ephemeral=True)
            return

        try:
            if self._todo_item_action_for(current_todo, mode) == "Reopen":
                todo = self._todo_manager.reopen_todo(user=BRIEFING_USER, todo_id=todo_id)
                status_message = self._format_todo_confirmation("Reopened", todo)
            else:
                todo = self._todo_manager.complete_todo(user=BRIEFING_USER, todo_id=todo_id)
                status_message = self._format_todo_confirmation("Completed", todo)
        except Exception as exc:
            await interaction.response.send_message(f"Todo action failed: {exc}", ephemeral=True)
            return

        await self._edit_interaction_to_list(interaction, mode=mode, page=page, status_message=status_message)

    async def _handle_todo_channel_message(self, message: discord.Message, user_text: str) -> bool:
        """Handle lightweight text commands in the dedicated todo channel."""
        if not DISCORD_TODO_CHANNEL_ID or message.channel.id != DISCORD_TODO_CHANNEL_ID:
            return False

        if user_text in {"!", "!help"}:
            dashboard_message = await self._ensure_todo_dashboard_message()
            if dashboard_message is None:
                await message.reply("Todo dashboard is unavailable right now.")
            return True

        return False

    async def _send_todo_list(
        self,
        interaction: discord.Interaction,
        *,
        include_completed: bool = False,
        only_due_today: bool = False,
    ) -> None:
        if not self._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        mode = "all" if include_completed else "open"
        if only_due_today:
            mode = "due"
        await self._send_list_response(
            interaction,
            mode=mode,
        )

    async def _mutate_todo(self, interaction: discord.Interaction, *, action: str, item: str) -> None:
        if not self._todo_manager.is_available():
            await interaction.response.send_message("Supabase is not configured for todos.", ephemeral=True)
            return

        try:
            kwargs = {"user": BRIEFING_USER, "todo_id": item if item.startswith("todo_") else None, "match": None if item.startswith("todo_") else item}
            if action == "complete":
                todo = self._todo_manager.complete_todo(**kwargs)
                message = self._format_todo_confirmation("Completed", todo)
            elif action == "reopen":
                todo = self._todo_manager.reopen_todo(**kwargs)
                message = self._format_todo_confirmation("Reopened", todo)
            elif action == "delete":
                todo = self._todo_manager.delete_todo(**kwargs)
                message = f"Deleted todo: `{todo['title']}`"
            else:
                message = "Unknown todo action."
            await self._send_dashboard_response(interaction, status_message=message)
        except Exception as exc:
            await interaction.response.send_message(f"Todo action failed: {exc}", ephemeral=True)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message):
        if message.author == self.user or message.author.bot:
            return

        user_text = message.content.strip()
        if not user_text:
            return

        if await self._handle_todo_channel_message(message, user_text):
            return

        if not DISCORD_CHANNEL_ID or message.channel.id != DISCORD_CHANNEL_ID:
            return

        # Serialize responses so concurrent messages don't interleave context
        async with self._response_lock:
            async with message.channel.typing():
                response_text, tool_calls = await self.orchestrator.run_response(user_text)

        if response_text is None:
            await message.reply("Sorry, I wasn't able to generate a response.")
            return

        # Build reply with tool details as subtext
        reply = response_text
        if tool_calls:
            tool_note = _format_tool_calls(tool_calls)
            if tool_note:
                reply = f"{response_text}\n{tool_note}"

        await message.reply(_truncate(reply))

    # ------------------------------------------------------------------
    # Proactive briefing delivery
    # ------------------------------------------------------------------

    async def _briefing_delivery_loop(self):
        """Catch up on missed briefings, then subscribe to live Supabase changes."""
        await self.wait_until_ready()
        channel = self.get_channel(DISCORD_BRIEFING_CHANNEL_ID)
        if channel is None:
            print(f"⚠️  Could not find channel {DISCORD_BRIEFING_CHANNEL_ID} -- briefing delivery disabled")
            return

        await self._send_pending_discord_briefings(channel, limit=BRIEFING_BOOT_LIMIT)
        poll_task = asyncio.create_task(self._briefing_poll_loop(channel))

        if not ASYNC_SUPABASE_AVAILABLE:
            print("⚠️  Async Supabase client unavailable -- realtime briefing subscription disabled; polling fallback active")
            await poll_task
            return

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            print("⚠️  SUPABASE_URL or SUPABASE_KEY not set -- realtime briefing subscription disabled; polling fallback active")
            await poll_task
            return

        try:
            supabase = await acreate_client(supabase_url, supabase_key)
            await supabase.realtime.connect()
            try:
                await supabase.realtime.set_auth(supabase_key)
            except Exception:
                pass
            loop = asyncio.get_running_loop()

            def schedule_refresh(payload: dict) -> None:
                payload_dict = payload if isinstance(payload, dict) else {}
                record = getattr(payload, "record", None) or payload_dict.get("record") or {}
                old_record = getattr(payload, "old_record", None) or payload_dict.get("old_record") or {}
                candidate = record or old_record
                if candidate.get("user_id") != BRIEFING_USER:
                    return

                briefing_id = candidate.get("id")
                if briefing_id:
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        self._send_pending_discord_briefings(
                            channel,
                            briefing_ids=[briefing_id],
                            limit=1,
                        )
                    )
                else:
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        self._send_pending_discord_briefings(channel, limit=BRIEFING_BOOT_LIMIT),
                    )

            realtime_channel = (
                supabase.channel(f"briefing-announcements-{BRIEFING_USER.lower()}")
                .on_postgres_changes(
                    "INSERT",
                    schema="public",
                    table=self._briefing_manager.TABLE_NAME,
                    callback=schedule_refresh,
                )
                .on_postgres_changes(
                    "UPDATE",
                    schema="public",
                    table=self._briefing_manager.TABLE_NAME,
                    callback=schedule_refresh,
                )
            )

            await realtime_channel.subscribe()
            print(f"📋 Briefing realtime subscription started for {BRIEFING_USER}")
            await supabase.realtime.listen()
        except Exception as e:
            print(f"⚠️  Briefing realtime subscription error: {e}")
            await poll_task
        finally:
            if not poll_task.done():
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poll_task

    async def _briefing_poll_loop(self, channel: discord.abc.Messageable) -> None:
        """Periodic fallback so pending briefings still get delivered without realtime events."""
        while True:
            await asyncio.sleep(BRIEFING_POLL_INTERVAL_SECONDS)
            try:
                await self._send_pending_discord_briefings(channel, limit=BRIEFING_BOOT_LIMIT)
            except Exception as e:
                print(f"⚠️  Briefing polling error: {e}")

    async def _send_pending_discord_briefings(
        self,
        channel: discord.abc.Messageable,
        *,
        briefing_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> None:
        """Fetch Discord-pending briefings and send them to the dedicated channel."""
        while True:
            try:
                briefings = await self._briefing_manager.get_pending_briefings_for_delivery(
                    user=BRIEFING_USER,
                    delivery_target="discord",
                    limit=limit,
                    briefing_ids=briefing_ids,
                )
            except Exception as e:
                print(f"⚠️  Briefing fetch error: {e}")
                return

            if not briefings:
                return

            for briefing in briefings:
                bid = briefing.get("id")
                if not bid or bid in self._briefings_in_flight:
                    continue

                opener, source = _extract_briefing_text(briefing)
                if not opener:
                    print(
                        "⚠️  Skipping Discord briefing "
                        f"{bid}: no opener_text or message content available"
                        + (f" (source={source})" if source else "")
                    )
                    continue

                if self._briefing_manager:
                    resolved = self._briefing_manager.get_combined_opener([briefing])
                    if resolved:
                        opener = resolved
                    else:
                        opener = _substitute_time_placeholder(opener, briefing)

                self._briefings_in_flight.add(bid)
                try:
                    await channel.send(_truncate(f"📢 {opener}"))
                    await self._briefing_manager.mark_discord_sent([bid])
                except Exception as e:
                    print(f"⚠️  Failed to send Discord briefing {bid}: {e}")
                finally:
                    self._briefings_in_flight.discard(bid)

            if briefing_ids or len(briefings) < limit:
                return
