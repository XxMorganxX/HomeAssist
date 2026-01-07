"""
Remote console logger for the HomeAssist dashboard.

Sends log entries asynchronously to the dashboard API without blocking
the main program flow. Uses fire-and-forget pattern with aiohttp.
"""

import os
import asyncio
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


def _send_log_sync(text: str, message_type: str, is_positive: Optional[bool] = None) -> None:
    """
    Synchronous log sender (runs in background thread).
    Uses requests for simplicity in thread context.
    
    Args:
        text: Message content to display
        message_type: "user", "agent", or "command"
        is_positive: Optional - only used for "command" type messages
    """
    token = _get_token()
    if not token:
        return
    
    try:
        import requests
        payload = {
            "token": token,
            "text": text,
            "type": message_type
        }
        # Only include is_positive for command type messages
        if message_type == "command" and is_positive is not None:
            payload["is_positive"] = is_positive
        
        requests.post(
            DASHBOARD_URL,
            json=payload,
            timeout=5
        )
    except Exception:
        # Never let logging failures affect the main program
        pass


def console_log(text: str, message_type: str, is_positive: Optional[bool] = None) -> None:
    """
    Fire-and-forget log to the remote console.
    
    Runs in a background thread to never block the main program.
    Silently fails if token not set or network issues occur.
    
    Args:
        text: Log message to display
        message_type: "user", "agent", or "command"
        is_positive: Optional - True for success (✓), False for error (✗). Only for "command" type.
    """
    if not _get_token():
        return
    
    # Fire and forget in background thread
    thread = threading.Thread(
        target=_send_log_sync,
        args=(text, message_type, is_positive),
        daemon=True
    )
    thread.start()


async def console_log_async(text: str, message_type: str, is_positive: Optional[bool] = None) -> None:
    """
    Async version of console_log for use in async contexts.
    Still non-blocking - creates a task that runs independently.
    
    Args:
        text: Log message to display
        message_type: "user", "agent", or "command"
        is_positive: Optional - True for success (✓), False for error (✗). Only for "command" type.
    """
    if not _get_token():
        return
    
    # Schedule in background without awaiting
    asyncio.get_event_loop().run_in_executor(
        None,
        _send_log_sync,
        text,
        message_type,
        is_positive
    )


# Convenience functions for specific event types
def log_boot() -> None:
    """Log program boot."""
    console_log("🚀 HomeAssist started", message_type="command", is_positive=True)


def log_shutdown() -> None:
    """Log program force shutdown (Ctrl+C)."""
    console_log("🛑 HomeAssist force shutdown", message_type="command", is_positive=False)


def log_conversation_end() -> None:
    """Log graceful conversation end (termination phrase like 'over out')."""
    console_log("👋 Conversation ended (user said goodbye)", message_type="command", is_positive=True)


def log_termination_detected(phrase_name: str, interrupted_state: str) -> None:
    """Log parallel termination phrase detection."""
    console_log(
        f"🛑 Termination phrase '{phrase_name}' detected - interrupting {interrupted_state}",
        message_type="command",
        is_positive=True
    )


def log_wake_word(model_name: str, score: float) -> None:
    """Log wake word detection."""
    console_log(f"👂 Wake word detected: {model_name} (score: {score:.2f})", message_type="command", is_positive=True)


def log_user_message(text: str) -> None:
    """Log user message sent to provider."""
    console_log(text, message_type="user")


def log_assistant_response(text: str) -> None:
    """Log assistant response received."""
    console_log(text, message_type="agent")


def log_tool_call(tool_name: str, success: bool = True) -> None:
    """Log tool execution."""
    if success:
        console_log(f"🔧 Tool called: {tool_name}", message_type="command", is_positive=True)
    else:
        console_log(f"❌ Tool failed: {tool_name}", message_type="command", is_positive=False)


def log_error(message: str) -> None:
    """Log an error."""
    console_log(f"❌ Error: {message}", message_type="command", is_positive=False)


def log_info(message: str) -> None:
    """Log general info."""
    console_log(f"ℹ️ {message}", message_type="command", is_positive=True)

