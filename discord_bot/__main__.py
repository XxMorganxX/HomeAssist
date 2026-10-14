"""
Entry point: python -m discord_bot
"""

import asyncio
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

# Suppress the verbose config summary that prints on assistant_framework import
os.environ.setdefault("QUIET_IMPORT", "1")

from discord_bot.bot import HomeAssistBot, DISCORD_BOT_TOKEN
from discord_bot.text_orchestrator import TextOrchestrator


async def async_main():
    if not DISCORD_BOT_TOKEN:
        print("❌ DISCORD_BOT_TOKEN not set in .env -- cannot start bot")
        sys.exit(1)

    orchestrator = TextOrchestrator()
    if not await orchestrator.initialize():
        print("❌ Orchestrator initialization failed")
        sys.exit(1)

    bot = HomeAssistBot(orchestrator)

    try:
        await bot.start(DISCORD_BOT_TOKEN)
    except KeyboardInterrupt:
        pass
    finally:
        if not bot.is_closed():
            await bot.close()
        await orchestrator.cleanup()


def main():
    signal.signal(signal.SIGINT, lambda *_: None)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")


if __name__ == "__main__":
    main()
