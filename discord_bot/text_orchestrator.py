"""
Text-only orchestrator for non-voice channels (Discord, webhooks, etc.).

Reuses the assistant_framework's response provider, context manager, and vector
memory without loading any audio infrastructure (wake word, transcription, TTS,
barge-in, shared audio bus).
"""

import asyncio
import os
import time
from typing import Optional, List, Tuple, Dict, Any

# Suppress the verbose config banner on import (we print our own status)
os.environ.setdefault("QUIET_IMPORT", "1")

from assistant_framework.config import (
    RESPONSE_PROVIDER,
    OPENAI_WS_CONFIG,
    UNIFIED_CONTEXT_CONFIG,
)
from assistant_framework.factory import ProviderFactory
from assistant_framework.providers.context import UnifiedContextProvider


class TextOrchestrator:
    """
    Lightweight orchestrator that only initialises the response provider
    (which starts an MCP server), a conversation context manager, and
    vector memory.  No audio providers are loaded.
    """

    def __init__(self):
        self._response = None
        self._context: Optional[UnifiedContextProvider] = None
        self._last_tool_calls: list = []
        self.is_initialized = False

    async def initialize(self) -> bool:
        boot_start = time.perf_counter()

        def elapsed() -> str:
            return f"[{time.perf_counter() - boot_start:.2f}s]"

        print(f"🚀 {elapsed()} Initializing TextOrchestrator (text-only, no audio)...")

        try:
            # Build only the configs we need — skip get_framework_config() which
            # runs audio device detection, Bluetooth scanning, etc.
            response_config = {
                "response": {
                    "provider": RESPONSE_PROVIDER,
                    "config": OPENAI_WS_CONFIG,
                }
            }

            # --- Context provider (instant, no async work) ---
            self._context = UnifiedContextProvider(UNIFIED_CONTEXT_CONFIG)

            # --- Parallelise the two slow init paths ---
            async def init_response():
                providers = ProviderFactory.create_all_providers(response_config)
                self._response = providers["response"]
                await self._response.initialize()

            async def init_vector_memory():
                if self._context and hasattr(self._context, "initialize_vector_memory"):
                    return await self._context.initialize_vector_memory()
                return False

            print(f"🔧 {elapsed()} Starting parallel init (response + vector memory)...")
            results = await asyncio.gather(
                init_response(),
                init_vector_memory(),
                return_exceptions=True,
            )

            if isinstance(results[0], Exception):
                raise results[0]
            print(f"✅ {elapsed()} Response provider ready")

            if isinstance(results[1], Exception):
                print(f"⚠️  {elapsed()} Vector memory error: {results[1]}")
            elif results[1]:
                print(f"✅ {elapsed()} Vector memory initialized")
            else:
                print(f"⚠️  {elapsed()} Vector memory unavailable (continuing without)")

            self.is_initialized = True
            print(f"✅ {elapsed()} TextOrchestrator ready")
            return True

        except Exception as e:
            print(f"❌ TextOrchestrator initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def run_response(self, user_message: str) -> Tuple[Optional[str], list]:
        """
        Generate a response for *user_message*.

        Returns:
            (response_text, tool_calls)
            response_text is None on failure.
            tool_calls is a list of ToolCall dataclasses (name, arguments, result).
        """
        self._last_tool_calls = []

        if not self._response:
            return None, []

        try:
            # Add user message to context
            if self._context:
                self._context.add_message("user", user_message)
                self._context.auto_trim_if_needed()

            # Vector memory query (parallel with context prep)
            vector_task = None
            if self._context and hasattr(self._context, "get_vector_memory_context"):
                vector_task = asyncio.create_task(
                    self._context.get_vector_memory_context(user_message)
                )

            # Build context bundle
            context = None
            tool_context = None
            if self._context:
                if hasattr(self._context, "get_context_bundle"):
                    bundle = self._context.get_context_bundle()
                    context = bundle.response_context
                    tool_context = bundle.tool_context
                else:
                    context = self._context.get_recent_for_response()
                    tool_context = self._context.get_tool_context()

            # Await vector memory
            if vector_task:
                vector_context = await vector_task or ""
                if vector_context and context:
                    context.insert(1, {"role": "system", "content": vector_context})

            # Stream response
            print(f"💭 Generating response for: {user_message[:80]}")
            full_response = ""
            streamed_deltas = False

            async for chunk in self._response.stream_response(
                user_message, context=context, tool_context=tool_context
            ):
                if chunk.content:
                    if chunk.is_complete:
                        full_response = chunk.content
                        if chunk.tool_calls:
                            self._last_tool_calls = chunk.tool_calls
                    else:
                        full_response += chunk.content
                        streamed_deltas = True

            # Add assistant response to context
            if self._context and full_response:
                self._context.add_message("assistant", full_response)

            return full_response or None, list(self._last_tool_calls)

        except Exception as e:
            print(f"❌ Response generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    def reset_context(self):
        """Reset conversation context (start a fresh session)."""
        if self._context:
            self._context.reset()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Shut down providers."""
        if self._response and hasattr(self._response, "cleanup"):
            try:
                await self._response.cleanup()
            except Exception as e:
                print(f"⚠️  Response provider cleanup error: {e}")
        print("🛑 TextOrchestrator shut down")
