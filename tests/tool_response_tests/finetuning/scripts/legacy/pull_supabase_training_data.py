#!/usr/bin/env python3
"""
Pull real conversation history from Supabase and convert to training data.

Queries:
  - conversation_sessions
  - conversation_messages (user prompts that triggered tool calls)
  - tool_calls (tool_name + arguments = ground truth labels)

Output: supabase_training_data.jsonl
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

SCRIPT_DIR = Path(__file__).parent

# Load system prompt for training format
with open(SCRIPT_DIR / "tool_schemas.json") as f:
    TOOL_SCHEMAS = json.load(f)

with open(SCRIPT_DIR / "system_prompt.txt") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{tools}",
    json.dumps(TOOL_SCHEMAS["tools"], indent=2)
)

VALID_TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS["tools"]}


def init_supabase():
    """Initialize Supabase client."""
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase package not installed. Run: pip install supabase")
        sys.exit(1)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    return create_client(url, key)


def fetch_tool_call_sessions(client, limit: int = 2000) -> list:
    """
    Fetch all tool_calls joined with their parent message and session.

    Returns a list of dicts:
      { session_id, message_id, user_content, tool_name, arguments, timestamp }
    """
    # Get all tool calls with their message content
    print("Fetching tool calls from Supabase...")
    tool_calls = (
        client.table("tool_calls")
        .select("id, message_id, tool_name, arguments, executed_at")
        .order("executed_at", desc=True)
        .limit(limit)
        .execute()
    )
    print(f"  Found {len(tool_calls.data)} tool call records")

    if not tool_calls.data:
        return []

    # Collect unique message IDs
    message_ids = list({tc["message_id"] for tc in tool_calls.data if tc["message_id"]})
    print(f"  Linked to {len(message_ids)} unique messages")

    # Fetch those messages (batch in chunks of 100 for Supabase limits)
    messages_by_id = {}
    for i in range(0, len(message_ids), 100):
        chunk = message_ids[i:i+100]
        msgs = (
            client.table("conversation_messages")
            .select("id, session_id, role, content, timestamp")
            .in_("id", chunk)
            .execute()
        )
        for m in msgs.data:
            messages_by_id[m["id"]] = m

    # For each assistant message with tool calls, find the preceding user message
    session_ids = list({m["session_id"] for m in messages_by_id.values() if m.get("session_id")})
    print(f"  Across {len(session_ids)} sessions")

    # Fetch all messages from those sessions to find user->assistant pairs
    all_session_messages = {}
    for i in range(0, len(session_ids), 50):
        chunk = session_ids[i:i+50]
        msgs = (
            client.table("conversation_messages")
            .select("id, session_id, role, content, timestamp")
            .in_("session_id", chunk)
            .order("timestamp")
            .execute()
        )
        for m in msgs.data:
            sid = m["session_id"]
            if sid not in all_session_messages:
                all_session_messages[sid] = []
            all_session_messages[sid].append(m)

    # Build training pairs: user_prompt -> [tool_calls]
    # Group tool calls by message_id
    tool_calls_by_msg = {}
    for tc in tool_calls.data:
        mid = tc["message_id"]
        if mid not in tool_calls_by_msg:
            tool_calls_by_msg[mid] = []
        tool_calls_by_msg[mid].append({
            "name": tc["tool_name"],
            "arguments": tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"] or "{}")
        })

    # For each message with tool calls, find the user message that triggered it
    training_pairs = []
    for msg_id, tools in tool_calls_by_msg.items():
        msg = messages_by_id.get(msg_id)
        if not msg:
            continue

        session_msgs = all_session_messages.get(msg.get("session_id"), [])

        # Find the user message right before this assistant message
        user_content = None
        for i, sm in enumerate(session_msgs):
            if sm["id"] == msg_id:
                # Look backward for the most recent user message
                for j in range(i - 1, -1, -1):
                    if session_msgs[j]["role"] == "user":
                        user_content = session_msgs[j]["content"]
                        break
                break

        if not user_content or not user_content.strip():
            continue

        # Filter to only known tools
        valid_tools = [t for t in tools if t["name"] in VALID_TOOL_NAMES]
        if not valid_tools:
            continue

        training_pairs.append({
            "user_prompt": user_content.strip(),
            "tool_calls": valid_tools,
            "session_id": msg.get("session_id"),
            "timestamp": msg.get("timestamp"),
        })

    return training_pairs


def deduplicate_pairs(pairs: list) -> list:
    """Remove exact duplicate user prompts, keeping the most recent."""
    seen = {}
    for pair in pairs:
        key = pair["user_prompt"].lower().strip()
        if key not in seen:
            seen[key] = pair
    return list(seen.values())


def convert_to_training_format(pairs: list) -> list:
    """Convert pairs to JSONL training format."""
    examples = []
    for pair in pairs:
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["user_prompt"]},
                {"role": "assistant", "content": json.dumps(pair["tool_calls"])}
            ]
        }
        examples.append(example)
    return examples


def main():
    client = init_supabase()

    # Fetch real tool call data
    pairs = fetch_tool_call_sessions(client, limit=5000)
    print(f"\nExtracted {len(pairs)} user→tool_call training pairs")

    if not pairs:
        print("No tool call data found in Supabase. Nothing to export.")
        return

    # Deduplicate
    unique_pairs = deduplicate_pairs(pairs)
    print(f"After deduplication: {unique_pairs} unique pairs")

    # Convert to training format
    examples = convert_to_training_format(unique_pairs)

    # Write output
    output_path = SCRIPT_DIR / "supabase_training_data.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} training examples to {output_path}")

    # Stats
    tool_counts = {}
    for pair in unique_pairs:
        for tc in pair["tool_calls"]:
            name = tc["name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

    print("\nTool distribution from real usage:")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # Also save raw pairs for inspection
    raw_path = SCRIPT_DIR / "supabase_raw_pairs.json"
    with open(raw_path, "w") as f:
        json.dump(unique_pairs, f, indent=2, default=str)
    print(f"\nRaw pairs saved to {raw_path} (for manual inspection)")


if __name__ == "__main__":
    main()
