# Tool Response Flow: What Gets Stored in Context/Conversation History

## Complete Flow

### 1. Tool Execution
```python
# Tool returns raw JSON dict
tool_result = {
    "success": True,
    "message": "Playback paused",
    "user": "morgan",
    "action": "pause",
    "timestamp": "2025-10-14T19:06:29.034310"
}
```

### 2. Result Storage in ToolCall Object
**Location**: `assistant_framework/providers/response/openai_websocket.py:341-342`
```python
tool_call = ToolCall(name=fc["name"], arguments=args)
result = await self.execute_tool(fc["name"], args)
tool_call.result = result  # Raw JSON is converted to STRING here
```

### 3. Composition Step (GPT-4o-mini generates natural language)
**Location**: `assistant_framework/providers/response/openai_websocket.py:346-351`
```python
# Tool results are passed to GPT-4o-mini for composition
final_text = await self._compose_final_answer(
    user_message=next((m.get("content", "") for m in messages...),
    context=[m for m in messages if m.get("role") in ("user", "assistant")],
    tool_calls=tool_calls,  # Contains raw results
    pre_text=""
)
```

**What happens inside `_compose_final_answer()`** (lines 464-489):
```python
# Tool summaries are created
for tc in tool_calls:
    if tc and tc.result:
        tool_summaries.append(f"{tc.name} result:\n{tc.result}")

guidance = (
    "You have executed tools for the user's request. "
    "Use the tool results below to produce a concise, direct answer for the user. "
    "Do not include raw JSON unless it improves clarity.\n\n" + tools_block
)

# GPT-4o-mini generates natural language response
result = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.6,
    max_tokens=800,
)
content = result.choices[0].message.content
# Returns: "I've paused your Spotify playback, Mr. Stuart."
```

### 4. ResponseChunk Creation
**Location**: `assistant_framework/providers/response/openai_websocket.py:353-358`
```python
yield ResponseChunk(
    content=final_text,           # Natural language: "I've paused your Spotify..."
    is_complete=True,
    tool_calls=tool_calls,        # Raw tool data is AVAILABLE HERE
    finish_reason="stop"
)
```

### 5. ResponseChunk Processing in Orchestrator
**Location**: `assistant_framework/orchestrator.py:338-348`
```python
full_response = ""
async for response_chunk in self.run_response_only(...):
    # Accumulate ONLY the content field
    full_response += response_chunk.content
    
    if response_chunk.is_complete:
        break
```

**⚠️ Note**: The `tool_calls` in the ResponseChunk are **NOT** accumulated here!

### 6. Storage in Conversation History
**Location**: `assistant_framework/orchestrator.py:350-352`
```python
# Add assistant response to context
if self.context:
    self.context.add_message("assistant", full_response)
```

**Location**: `assistant_framework/providers/context/unified_context.py:86-103`
```python
def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    message = ConversationMessage(
        role=message_role,
        content=content,        # ONLY natural language text
        metadata=metadata or {}
    )
    self.conversation_history.append(message)
```

---

## What Actually Gets Stored

### ❌ **NOT Stored in Conversation History:**
- Raw tool results (JSON dict)
- Tool call arguments
- Tool execution metadata
- The ToolCall objects themselves

### ✅ **STORED in Conversation History:**
- **Only the composed natural language response** from GPT-4o-mini
- Example: "I've paused your Spotify playback, Mr. Stuart." 
- NOT: `{"success": true, "message": "Playback paused", ...}`

---

## Data Models

### ConversationMessage Structure
```python
@dataclass
class ConversationMessage:
    role: MessageRole              # user, assistant, system
    content: str                   # Natural language text ONLY
    timestamp: float
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None  # ⚠️ Not populated by orchestrator!
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Note**: While `ConversationMessage` HAS a `tool_calls` field, the orchestrator's `add_message()` call at line 352 **does not pass tool calls**, so they are never stored.

---

## Summary

**What the user sees in conversation history:**
```
User: "Hit pause on Spotify"
Assistant: "I've paused your Spotify playback, Mr. Stuart."
```

**What's NOT in conversation history (but was available during processing):**
```json
{
  "tool_name": "improved_spotify_playback",
  "arguments": {"action": "pause", "user": "morgan"},
  "result": {
    "success": true,
    "message": "Playback paused",
    "user": "morgan",
    "action": "pause",
    "timestamp": "2025-10-14T19:06:29.034310"
  }
}
```

---

## Potential Improvement

If you want to store tool calls in history, you would need to modify:

**Option 1**: Update orchestrator to pass tool_calls
```python
# In orchestrator.py around line 350
if self.context and response_chunk.tool_calls:
    # Store with tool call information
    self.context.add_message_with_tools(
        "assistant", 
        full_response,
        tool_calls=response_chunk.tool_calls
    )
```

**Option 2**: Store in metadata
```python
if self.context:
    metadata = {}
    if response_chunk.tool_calls:
        metadata['tool_calls'] = [tc.to_dict() for tc in response_chunk.tool_calls]
    self.context.add_message("assistant", full_response, metadata=metadata)
```

This would preserve the raw tool execution data for debugging, analysis, or future context.

