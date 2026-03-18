"""
Tool-Calling Mini routing provider.

Routes tool calls through the Tool-Calling Mini inference API — a fine-tuned
Qwen3 model that handles tool selection and argument extraction on-device.
See TOOL_CALLING_MINI_API.md for the full spec.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

try:
    from ...interfaces.tool_routing import ToolRoutingInterface
    from ...models.data_models import ToolCall, HandoffContext
except ImportError:
    from assistant_framework.interfaces.tool_routing import ToolRoutingInterface
    from assistant_framework.models.data_models import ToolCall, HandoffContext


class ToolCallingMiniProvider(ToolRoutingInterface):
    """Routes tool calls via the Tool-Calling Mini inference API."""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url", "https://inference.stuart-labs.com")
        self.refresh_token = config.get("refresh_token", "")
        self.enable_thinking = config.get("enable_thinking", False)
        self.execute_tools = config.get("execute_tools", False)
        self.generation = config.get("generation", {})
        self.timeout = config.get("timeout", 120)

        self._api_key: Optional[str] = None
        self._key_expires_at: Optional[datetime] = None
        self._session = None

    async def initialize(self) -> bool:
        """Validate connectivity via GET /health and obtain an initial API key."""
        try:
            if not self.refresh_token:
                print("❌ [ToolRouting:Mini] INFERENCE_REFRESH_TOKEN not set — cannot authenticate")
                return False

            import httpx
            self._session = httpx.AsyncClient(timeout=self.timeout)

            resp = await self._session.get(f"{self.base_url}/health")
            if resp.status_code != 200:
                print(f"⚠️ [ToolRouting:Mini] Health check failed: HTTP {resp.status_code}")
                return False
            print(f"✅ [ToolRouting:Mini] Server healthy at {self.base_url}")

            await self._ensure_api_key()
            return True
        except Exception as e:
            print(f"❌ [ToolRouting:Mini] Initialization failed: {e}")
            return False

    # ------------------------------------------------------------------
    # ToolRoutingInterface
    # ------------------------------------------------------------------

    async def route(
        self,
        user_message: str,
        context: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        handoff: Optional[HandoffContext] = None,
    ) -> List[ToolCall]:
        """Send the message to the Mini API and parse returned tool_calls."""
        import json as _json
        try:
            messages = self._build_messages(user_message, context)

            body: Dict[str, Any] = {
                "messages": messages,
                "enable_thinking": self.enable_thinking,
                "execute_tools": self.execute_tools,
            }
            if self.generation:
                body["generation"] = self.generation

            print(f"🔄 [ToolRouting:Mini] POST /v1/chat/completions ({len(messages)} messages, execute_tools={self.execute_tools})")
            api_start = time.time()
            data = await self._request("POST", "/v1/chat/completions", json=body)
            api_ms = int((time.time() - api_start) * 1000)
            print(f"⏱️ [ToolRouting:Mini] route api_ms={api_ms}")

            if data is None:
                print("❌ [ToolRouting:Mini] API returned no data (request failed)")
                return []

            # Diagnostic: log response shape
            print(f"📦 [ToolRouting:Mini] Response keys: {list(data.keys())}")
            raw_calls = data.get("tool_calls")
            print(f"   tool_calls type={type(raw_calls).__name__}, value={_json.dumps(raw_calls)[:500] if raw_calls else 'null'}")
            if data.get("content"):
                print(f"   content: {data['content'][:200]}")
            if data.get("usage"):
                print(f"   usage: {data['usage']}")

            if not raw_calls:
                return []

            tool_calls: List[ToolCall] = []
            for tc in raw_calls:
                name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = _json.loads(arguments)
                    except Exception:
                        arguments = {}
                print(f"🔧 [ToolRouting:Mini] → {name}({_json.dumps(arguments)[:200]})")
                tool_calls.append(ToolCall(name=name, arguments=arguments))

            return tool_calls

        except Exception as e:
            print(f"❌ [ToolRouting:Mini] route failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def check_additional_tools(
        self,
        user_message: str,
        tool_calls_so_far: List[ToolCall],
        context: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        handoff: Optional[HandoffContext] = None,
    ) -> List[ToolCall]:
        """
        Re-query the Mini API with tool results appended as messages.

        The API is stateless, so we send the full conversation including
        tool-call results so the model can decide if more tools are needed.
        """
        try:
            messages = self._build_messages(user_message, context)

            for tc in tool_calls_so_far:
                if tc and tc.result is not None:
                    result_preview = tc.result[:500] if len(tc.result) > 500 else tc.result
                    messages.append({
                        "role": "tool",
                        "content": f"[{tc.name}] {result_preview}",
                    })

            messages.append({
                "role": "user",
                "content": "Based on the tool results above, are additional tools needed to fully answer the original request? If not, just respond conversationally.",
            })

            body: Dict[str, Any] = {
                "messages": messages,
                "enable_thinking": self.enable_thinking,
                "execute_tools": self.execute_tools,
            }
            if self.generation:
                body["generation"] = self.generation

            data = await self._request("POST", "/v1/chat/completions", json=body)
            if data is None:
                return []

            raw_calls = data.get("tool_calls")
            if not raw_calls:
                return []

            already_executed = {tc.name for tc in tool_calls_so_far if tc}
            additional: List[ToolCall] = []
            for tc in raw_calls:
                name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                if isinstance(arguments, str):
                    import json
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                if name in already_executed:
                    continue
                additional.append(ToolCall(name=name, arguments=arguments))

            return additional

        except Exception as e:
            print(f"⚠️ [ToolRouting:Mini] check_additional_tools failed: {e}")
            return []

    @property
    def supports_iterative_routing(self) -> bool:
        return True

    async def cleanup(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None
        self._api_key = None
        self._key_expires_at = None

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def _ensure_api_key(self) -> str:
        """Return a valid API key, refreshing if expired."""
        now = datetime.now(timezone.utc)
        if self._api_key and self._key_expires_at and now < self._key_expires_at:
            return self._api_key

        session = await self._ensure_session()
        resp = await session.post(
            f"{self.base_url}/auth/token",
            json={"refresh_token": self.refresh_token},
        )
        resp.raise_for_status()
        data = resp.json()
        self._api_key = data["api_key"]
        self._key_expires_at = datetime.fromisoformat(data["expires_at"])
        print(f"🔑 [ToolRouting:Mini] API key refreshed (expires {self._key_expires_at.isoformat()})")
        return self._api_key

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self):
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=self.timeout)
        return self._session

    async def _request(
        self, method: str, path: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make an authenticated request, retrying once on 403."""
        session = await self._ensure_session()
        api_key = await self._ensure_api_key()
        headers = {"Content-Type": "application/json", "X-API-Key": api_key}

        url = f"{self.base_url}{path}"
        resp = await session.request(method, url, headers=headers, **kwargs)

        if resp.status_code == 403:
            print("🔑 [ToolRouting:Mini] 403 — refreshing API key and retrying")
            self._api_key = None
            api_key = await self._ensure_api_key()
            headers["X-API-Key"] = api_key
            resp = await session.request(method, url, headers=headers, **kwargs)

        if resp.status_code >= 400:
            print(f"❌ [ToolRouting:Mini] {method} {path} → HTTP {resp.status_code}")
            print(f"   Response: {resp.text[:500]}")
            return None

        return resp.json()

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        user_message: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages array for the Mini API (stateless)."""
        messages: List[Dict[str, str]] = []
        for msg in (context or [])[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content and role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages
