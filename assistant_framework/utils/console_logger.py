"""
Remote console logger for the HomeAssist dashboard.

Sends log entries asynchronously to the dashboard API without blocking
the main program flow. Uses fire-and-forget pattern with aiohttp.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional
import threading

# Dashboard API endpoint
DASHBOARD_URL = "https://home-assist-web-dashboard.vercel.app/api/console/log"

# Token from environment
_CONSOLE_TOKEN: Optional[str] = None


def _get_token() -> Optional[str]:
    """Get the console token from environment (cached)."""
    global _CONSOLE_TOKEN
    if _CONSOLE_TOKEN is None:
        _CONSOLE_TOKEN = os.getenv("CONSOLE_TOKEN", "").strip()
    return _CONSOLE_TOKEN if _CONSOLE_TOKEN else None


def _send_log_sync(text: str, is_positive: bool = True) -> None:
    """
    Synchronous log sender (runs in background thread).
    Uses requests for simplicity in thread context.
    """
    token = _get_token()
    if not token:
        return
    
    try:
        import requests
        requests.post(
            DASHBOARD_URL,
            json={
                "token": token,
                "text": text,
                "is_positive": is_positive,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            timeout=5
        )
    except Exception:
        # Never let logging failures affect the main program
        pass


def console_log(text: str, is_positive: bool = True) -> None:
    """
    Fire-and-forget log to the remote console.
    
    Runs in a background thread to never block the main program.
    Silently fails if token not set or network issues occur.
    
    Args:
        text: Log message to display
        is_positive: True for success/green, False for error/red
    """
    if not _get_token():
        return
    
    # Fire and forget in background thread
    thread = threading.Thread(
        target=_send_log_sync,
        args=(text, is_positive),
        daemon=True
    )
    thread.start()


async def console_log_async(text: str, is_positive: bool = True) -> None:
    """
    Async version of console_log for use in async contexts.
    Still non-blocking - creates a task that runs independently.
    
    Args:
        text: Log message to display
        is_positive: True for success/green, False for error/red
    """
    if not _get_token():
        return
    
    # Schedule in background without awaiting
    asyncio.get_event_loop().run_in_executor(
        None,
        _send_log_sync,
        text,
        is_positive
    )


# Convenience functions for specific event types
def log_boot() -> None:
    """Log program boot."""
    console_log("🚀 HomeAssist started", is_positive=True)


def log_shutdown() -> None:
    """Log program force shutdown (Ctrl+C)."""
    console_log("🛑 HomeAssist force shutdown", is_positive=False)


def log_conversation_end() -> None:
    """Log graceful conversation end (termination phrase like 'over out')."""
    console_log("👋 Conversation ended (user said goodbye)", is_positive=True)


def log_wake_word(model_name: str, score: float) -> None:
    """Log wake word detection."""
    console_log(f"👂 Wake word detected: {model_name} (score: {score:.2f})", is_positive=True)


def log_user_message(text: str) -> None:
    """Log user message sent to provider."""
    # Truncate long messages
    display_text = text[:100] + "..." if len(text) > 100 else text
    console_log(f"🎤 User: {display_text}", is_positive=True)


def log_assistant_response(text: str) -> None:
    """Log assistant response received."""
    # Truncate long responses
    display_text = text[:100] + "..." if len(text) > 100 else text
    console_log(f"🤖 Assistant: {display_text}", is_positive=True)


def log_tool_call(tool_name: str, success: bool = True) -> None:
    """Log tool execution."""
    if success:
        console_log(f"🔧 Tool called: {tool_name}", is_positive=True)
    else:
        console_log(f"❌ Tool failed: {tool_name}", is_positive=False)


def log_error(message: str) -> None:
    """Log an error."""
    console_log(f"❌ Error: {message}", is_positive=False)


def log_info(message: str) -> None:
    """Log general info."""
    console_log(f"ℹ️ {message}", is_positive=True)

