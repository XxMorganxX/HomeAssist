"""
OpenAI-based tool routing provider.

Uses OpenAI chat completions with function calling (tool_choice="required")
to determine which tools to call and with what arguments. Extracted from
the inline routing logic previously in openai_websocket.py.
"""

import json
import time
import hashlib
from typing import Optional, List, Dict, Any

try:
    from ...interfaces.tool_routing import ToolRoutingInterface
    from ...models.data_models import ToolCall, HandoffContext
    from ...config import TOOL_SUBAGENT_CONFIG
except ImportError:
    from assistant_framework.interfaces.tool_routing import ToolRoutingInterface
    from assistant_framework.models.data_models import ToolCall, HandoffContext
    from assistant_framework.config import TOOL_SUBAGENT_CONFIG


class OpenAIToolRoutingProvider(ToolRoutingInterface):
    """Routes tool calls via OpenAI chat completions with function calling."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("tool_subagent_model", "gpt-4o-mini")
        self.max_tool_iterations = config.get("max_tool_iterations", 5)
        self._client = None

        self._input_tokens = 0
        self._output_tokens = 0

    async def initialize(self) -> bool:
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Failed to initialize OpenAI tool routing provider: {e}")
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
        """Select tools via OpenAI with tool_choice='required'."""
        try:
            client = await self._ensure_client()

            orchestration_prompt = TOOL_SUBAGENT_CONFIG.get("tool_orchestration_prompt", "")
            formatted_prompt = orchestration_prompt.format(user_message=user_message)

            tools = self._build_tools_payload(available_tools)

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": formatted_prompt},
            ]
            for msg in (context or [])[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_message})

            print(f"🔄 [ToolRouting:OpenAI] Calling {self.model} with {len(messages)} messages, tool_choice=required")
            api_start = time.time()
            result = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="required",
                max_completion_tokens=1000,
            )
            api_ms = int((time.time() - api_start) * 1000)
            print(f"⏱️ [ToolRouting:OpenAI] route api_ms={api_ms}")

            self._track_usage(result)

            choice = result.choices[0] if result and result.choices else None
            if not choice or not choice.message.tool_calls:
                content = (choice.message.content if choice else "") or ""
                print(f"⚠️ [ToolRouting:OpenAI] No tool calls returned: {content[:200]}")
                return []

            tool_calls: List[ToolCall] = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                print(f"🔧 [ToolRouting:OpenAI] → {tc.function.name} {json.dumps(args, indent=2)[:300]}")
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

            return tool_calls

        except Exception as e:
            print(f"❌ [ToolRouting:OpenAI] route failed: {e}")
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
        """Decide whether more tools are needed after the first execution round."""
        try:
            client = await self._ensure_client()

            executed_signatures = {
                self._get_tool_signature(tc.name, tc.arguments)
                for tc in tool_calls_so_far if tc
            }

            # ---- build tool-results summary ----
            tool_summaries = []
            successful_tools = []
            for tc in tool_calls_so_far:
                if tc and tc.result:
                    is_success = (
                        '"success":true' in tc.result.lower()
                        or '"success": true' in tc.result.lower()
                    )
                    status = "✓ SUCCESS" if is_success else "Result"
                    if is_success:
                        successful_tools.append(tc.name)
                    result_preview = tc.result[:500] + "..." if len(tc.result) > 500 else tc.result
                    tool_summaries.append(f"[{status}] Tool '{tc.name}':\n{result_preview}")
            tools_summary = "\n\n".join(tool_summaries)

            # ---- success / stickies notes ----
            stickies_read_done = any(
                tc and tc.name == "stickies" and tc.arguments and tc.arguments.get("action") == "read"
                for tc in tool_calls_so_far
            )
            non_stickies_successful = [t for t in successful_tools if t != "stickies"]
            success_note = ""
            if non_stickies_successful:
                success_note = f"\n\n⚠️ ALREADY COMPLETED: {', '.join(non_stickies_successful)} executed successfully. Do NOT call these again."
            if stickies_read_done:
                success_note += "\n\n📝 STICKIES READ COMPLETED - if user wanted to add/remove/edit, you MUST now call stickies with action='write'."

            # ---- iteration note ----
            iteration = len([tc for tc in tool_calls_so_far if tc]) - 1
            iteration = max(1, iteration)
            iteration_note = f"ITERATION: {iteration}/{self.max_tool_iterations}"
            if iteration >= self.max_tool_iterations - 1:
                iteration_note += " - APPROACHING LIMIT, prioritize completing the task now"

            # ---- format prompt ----
            prompt_template = TOOL_SUBAGENT_CONFIG.get("tool_decision_prompt", "")
            user_intent = handoff.user_intent if handoff else user_message[:100]
            relevant_context = (
                "\n".join(handoff.relevant_context)
                if handoff and handoff.relevant_context
                else "None"
            )
            system_content = prompt_template.format(
                user_intent=user_intent,
                relevant_context=relevant_context,
                iteration_note=iteration_note,
                success_note=success_note,
                tools_summary=tools_summary,
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_content},
            ]
            for msg in (context or [])[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({
                        "role": role if role in ("user", "assistant") else "assistant",
                        "content": content,
                    })
            messages.append({"role": "user", "content": user_message})

            tools = self._build_tools_payload(available_tools)

            result = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto",
                max_completion_tokens=2480,
            )
            self._track_usage(result)

            choice = result.choices[0] if result and result.choices else None
            if not choice or not choice.message.tool_calls:
                content = (choice.message.content if choice else "") or ""
                if "DONE" in content.upper() or not content.strip():
                    return []
                return []

            # ---- duplicate / one-shot filtering ----
            calendar_already_done = any(tc and tc.name == "calendar_data" for tc in tool_calls_so_far)
            search_already_done = any(tc and tc.name == "google_search" for tc in tool_calls_so_far)
            stickies_already_done = any(tc and tc.name == "stickies" for tc in tool_calls_so_far)

            additional: List[ToolCall] = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                sig = self._get_tool_signature(tc.function.name, args)
                if sig in executed_signatures:
                    print(f"🚫 Skipping duplicate tool call: {tc.function.name}")
                    continue

                if tc.function.name == "calendar_data" and calendar_already_done:
                    print("🚫 Blocking calendar_data — only one call per request")
                    continue

                if tc.function.name == "google_search" and search_already_done:
                    print("🚫 Blocking google_search — already completed in this request")
                    continue

                if tc.function.name == "stickies":
                    stickies_in_batch = sum(1 for t in additional if t.name == "stickies")
                    if stickies_already_done:
                        stickies_calls = [t for t in tool_calls_so_far if t and t.name == "stickies"]
                        had_write = any(t.arguments and t.arguments.get("action") == "write" for t in stickies_calls)
                        had_read = any(t.arguments and t.arguments.get("action") == "read" for t in stickies_calls)
                        this_is_write = args.get("action") == "write"
                        if had_write:
                            print("🚫 Blocking stickies — write already done, max calls reached")
                            continue
                        if stickies_in_batch > 0:
                            print("🚫 Blocking stickies — one already in batch")
                            continue
                        if had_read and this_is_write:
                            print("✅ Allowing stickies write after read (read→write flow)")
                        else:
                            print("🚫 Blocking stickies — invalid sequence")
                            continue
                    elif stickies_in_batch > 0:
                        print("🚫 Blocking stickies — one already in batch")
                        continue

                additional.append(ToolCall(name=tc.function.name, arguments=args))

            if not additional:
                print("✅ All requested tools were duplicates — task complete")
            return additional

        except Exception as e:
            print(f"⚠️ [ToolRouting:OpenAI] check_additional_tools failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def cleanup(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Token usage tracking (exposed so the response provider can aggregate)
    # ------------------------------------------------------------------

    def get_token_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total": self._input_tokens + self._output_tokens,
        }

    def reset_token_usage(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    @staticmethod
    def _build_tools_payload(
        available_tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Convert openai_functions list to the tools payload format."""
        if not available_tools:
            return []
        tools = []
        for func in available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return tools

    @staticmethod
    def _get_tool_signature(name: str, arguments: Dict[str, Any]) -> str:
        args_str = json.dumps(arguments, sort_keys=True) if arguments else "{}"
        return hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()

    def _track_usage(self, result) -> None:
        if result and result.usage:
            self._input_tokens += result.usage.prompt_tokens or 0
            self._output_tokens += result.usage.completion_tokens or 0
