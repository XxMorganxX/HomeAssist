"""
Discord bot that bridges a text channel to the HomeAssist orchestrator.

Features:
  - Listens in a single configured channel
  - Passes messages through TextOrchestrator (LLM + MCP tools)
  - Shows which tools the assistant invoked (replies to own message with schema)
  - Polls for pending briefings and posts them proactively
"""

import asyncio
import json
import os
from typing import Optional

import discord

from assistant_framework.utils.briefing.briefing_manager import BriefingManager
from discord_bot.text_orchestrator import TextOrchestrator

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
BRIEFING_POLL_INTERVAL = int(os.getenv("DISCORD_BRIEFING_POLL_SECONDS", "60"))
BRIEFING_USER = os.getenv("EMAIL_NOTIFICATION_RECIPIENT", "Morgan")

# Discord limits messages to 2000 chars
MAX_MESSAGE_LENGTH = 2000


def _truncate(text: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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


class HomeAssistBot(discord.Client):
    """Thin discord.Client subclass wired to the TextOrchestrator."""

    def __init__(self, orchestrator: TextOrchestrator, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.orchestrator = orchestrator
        self._briefing_manager: Optional[BriefingManager] = None
        self._briefing_task: Optional[asyncio.Task] = None
        self._response_lock = asyncio.Lock()

        try:
            self._briefing_manager = BriefingManager()
        except Exception as e:
            print(f"⚠️  Briefing manager unavailable: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self):
        print(f"✅ Discord bot logged in as {self.user}")
        if DISCORD_CHANNEL_ID:
            print(f"📡 Listening in channel {DISCORD_CHANNEL_ID}")
            channel = self.get_channel(DISCORD_CHANNEL_ID)
            if channel:
                await channel.send("HomeAssist online.")
        else:
            print("⚠️  DISCORD_CHANNEL_ID not set -- bot will not respond to any channel")

        if self._briefing_manager and DISCORD_CHANNEL_ID:
            self._briefing_task = asyncio.create_task(self._briefing_loop())

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message):
        if message.author == self.user or message.author.bot:
            return
        if not DISCORD_CHANNEL_ID or message.channel.id != DISCORD_CHANNEL_ID:
            return

        user_text = message.content.strip()
        if not user_text:
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

    async def _briefing_loop(self):
        """Poll Supabase for pending briefings and post them to the channel."""
        await self.wait_until_ready()
        channel = self.get_channel(DISCORD_CHANNEL_ID)
        if channel is None:
            print(f"⚠️  Could not find channel {DISCORD_CHANNEL_ID} -- briefing poller disabled")
            return

        print(f"📋 Briefing poller started (every {BRIEFING_POLL_INTERVAL}s)")
        while not self.is_closed():
            try:
                briefings = await self._briefing_manager.get_pending_briefings_with_opener(
                    user=BRIEFING_USER
                )
                for briefing in briefings:
                    opener = (
                        briefing.get("opener_text")
                        or (briefing.get("content") or {}).get("message", "")
                    )
                    if opener:
                        await channel.send(_truncate(f"📢 {opener}"))

                    bid = briefing.get("id")
                    if bid:
                        try:
                            await self._briefing_manager.mark_delivered([bid])
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️  Briefing poll error: {e}")

            await asyncio.sleep(BRIEFING_POLL_INTERVAL)
