#!/usr/bin/env python3
"""
Transform training_data.jsonl to include Qwen3-style <think> reasoning traces.
Reads each example, generates a contextual thinking trace based on the user query
and tool calls, and writes the updated JSONL.
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Vocabulary pools for varied language
# ---------------------------------------------------------------------------
USER_WANTS = [
    "The user wants", "The user is asking", "The user is requesting",
    "User wants", "They want", "They're asking", "The request is",
]
I_SHOULD = [
    "I should use", "I'll use", "I need to use", "This calls for",
    "The right tool is", "I'll call",
]
BECAUSE = [
    "because", "since", "as", "given that",
]

def pick(lst):
    return random.choice(lst)

# ---------------------------------------------------------------------------
# Tool-aware thinking generators
# ---------------------------------------------------------------------------

def think_weather(args, user_prompt):
    lines = []
    if "specific_date" in args:
        d = args["specific_date"]
        lines.append(f"{pick(USER_WANTS)} to know the weather for {d}.")
        lines.append(f"{pick(I_SHOULD)} the weather tool with specific_date=\"{d}\".")
    elif "hours" in args:
        h = args["hours"]
        lines.append(f"{pick(USER_WANTS)} an hourly weather forecast.")
        lines.append(f"{pick(I_SHOULD)} the weather tool with hours={h} for a detailed hourly breakdown.")
    elif "days" in args:
        d = args["days"]
        if d == 1:
            lines.append(f"{pick(USER_WANTS)} the current weather or today's forecast.")
            lines.append(f"{pick(I_SHOULD)} the weather tool with days=1.")
        else:
            lines.append(f"{pick(USER_WANTS)} a {d}-day weather forecast.")
            lines.append(f"{pick(I_SHOULD)} the weather tool with days={d}.")
    return " ".join(lines)


def think_spotify(args, user_prompt):
    action = args.get("action", "play")
    lines = []

    if action == "play" and "query" in args:
        query = args["query"]
        stype = args.get("search_type", "track")
        # Determine if it's a genre/mood vs specific artist/track
        lower_q = query.lower()
        genre_keywords = ["music", "jazz", "rock", "classical", "hip hop", "pop",
                          "lo-fi", "chill", "relax", "focus", "study", "workout",
                          "party", "sleep", "meditation", "indie", "soft", "country",
                          "electronic", "R&B", "beats", "ambient"]
        is_genre = any(kw in lower_q for kw in genre_keywords)
        if is_genre:
            lines.append(f"{pick(USER_WANTS)} to listen to {query}.")
            lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"play\", query=\"{query}\", and search_type=\"artist\" {pick(BECAUSE)} genre/mood searches work best as artist searches.")
        else:
            if stype == "artist":
                lines.append(f"{pick(USER_WANTS)} to hear music by {query}.")
                lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"play\", query=\"{query}\", search_type=\"artist\".")
            else:
                lines.append(f"{pick(USER_WANTS)} to play the song \"{query}\".")
                lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"play\", query=\"{query}\", search_type=\"track\".")
    elif action == "play":
        lines.append(f"{pick(USER_WANTS)} to resume playback.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"play\" (no query means resume).")
    elif action == "pause":
        lines.append(f"{pick(USER_WANTS)} to stop/pause the music.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"pause\".")
    elif action == "next":
        lines.append(f"{pick(USER_WANTS)} to skip to the next track.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"next\".")
    elif action == "previous":
        lines.append(f"{pick(USER_WANTS)} to go back to the previous track.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"previous\".")
    elif action == "volume":
        vol = args.get("volume_level", 50)
        lines.append(f"{pick(USER_WANTS)} to change the volume to {vol}%.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"volume\" and volume_level={vol}.")
    elif action == "status":
        lines.append(f"{pick(USER_WANTS)} to know what's currently playing.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"status\".")
    elif action == "devices":
        lines.append(f"{pick(USER_WANTS)} to see available Spotify devices.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"devices\".")
    elif action == "shuffle":
        state = args.get("shuffle_state")
        if state is True:
            lines.append(f"{pick(USER_WANTS)} to enable shuffle mode.")
        elif state is False:
            lines.append(f"{pick(USER_WANTS)} to disable shuffle mode.")
        else:
            lines.append(f"{pick(USER_WANTS)} to toggle shuffle mode.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"shuffle\"" +
                      (f" and shuffle_state={json.dumps(state)}" if state is not None else "") + ".")
    elif action == "repeat":
        mode = args.get("repeat_mode")
        if mode == "track":
            lines.append(f"{pick(USER_WANTS)} to loop the current song.")
        elif mode == "context":
            lines.append(f"{pick(USER_WANTS)} to loop the current playlist/album.")
        elif mode == "off":
            lines.append(f"{pick(USER_WANTS)} to turn off repeat.")
        else:
            lines.append(f"{pick(USER_WANTS)} to toggle repeat mode.")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"repeat\"" +
                      (f" and repeat_mode=\"{mode}\"" if mode else "") + ".")
    elif action in ("search_track", "search_artist"):
        query = args.get("query", "")
        lines.append(f"{pick(USER_WANTS)} to search for \"{query}\".")
        lines.append(f"{pick(I_SHOULD)} spotify_playback with action=\"{action}\".")
    else:
        lines.append(f"Spotify action: {action}.")

    return " ".join(lines)


def think_kasa(args, user_prompt):
    interaction = args.get("interaction", "direct")
    lines = []

    if interaction == "direct":
        light = args.get("light_name", "Light 1")
        action = args.get("action", "on")
        lines.append(f"{pick(USER_WANTS)} to turn {action} {light}.")
        lines.append(f"{pick(I_SHOULD)} kasa_lighting in direct mode with light_name=\"{light}\" and action=\"{action}\".")
    elif interaction == "scene":
        scene = args.get("scene_name", "mood")
        room = args.get("room", "living room")
        lines.append(f"{pick(USER_WANTS)} to set a {scene} lighting scene in the {room}.")
        lines.append(f"{pick(I_SHOULD)} kasa_lighting in scene mode with scene_name=\"{scene}\" and room=\"{room}\".")

    return " ".join(lines)


def think_calendar(args, user_prompt):
    commands = args.get("commands", [{}])
    cmd = commands[0] if commands else {}
    rw = cmd.get("read_or_write", "read")
    lines = []

    if rw == "read":
        rt = cmd.get("read_type", "next_events")
        date = cmd.get("date")
        limit = cmd.get("limit")
        if rt == "day_summary":
            lines.append(f"{pick(USER_WANTS)} to see today's calendar events.")
            lines.append(f"{pick(I_SHOULD)} calendar_data with read_type=\"day_summary\" across all calendars.")
        elif rt == "week_summary":
            lines.append(f"{pick(USER_WANTS)} a weekly schedule overview.")
            lines.append(f"{pick(I_SHOULD)} calendar_data with read_type=\"week_summary\".")
        elif rt == "specific_date" and date:
            lines.append(f"{pick(USER_WANTS)} to check the schedule for {date}.")
            lines.append(f"{pick(I_SHOULD)} calendar_data with read_type=\"specific_date\" and date=\"{date}\".")
        elif rt == "next_events":
            if limit:
                lines.append(f"{pick(USER_WANTS)} to see the next {limit} upcoming events.")
            else:
                lines.append(f"{pick(USER_WANTS)} to see upcoming calendar events.")
            lines.append(f"{pick(I_SHOULD)} calendar_data with read_type=\"next_events\".")
    elif rw == "create_event":
        title = cmd.get("event_title", "event")
        start = cmd.get("start_time", "")
        location = cmd.get("location")
        end = cmd.get("end_time")
        lines.append(f"{pick(USER_WANTS)} to create a calendar event: \"{title}\" at {start}.")
        detail_parts = [f"event_title=\"{title}\", start_time=\"{start}\""]
        if end:
            detail_parts.append(f"end_time=\"{end}\"")
        if location:
            detail_parts.append(f"location=\"{location}\"")
        lines.append(f"{pick(I_SHOULD)} calendar_data with create_event, {', '.join(detail_parts)}.")

    return " ".join(lines)


def think_stickies(args, user_prompt):
    action = args.get("action", "read")
    lines = []

    if action == "read":
        section = args.get("section", "both")
        if section == "todo":
            lines.append(f"{pick(USER_WANTS)} to see their to-do list.")
        elif section == "notes":
            lines.append(f"{pick(USER_WANTS)} to read their notes.")
        else:
            lines.append(f"{pick(USER_WANTS)} to check their sticky note (notes and to-dos).")
        lines.append(f"{pick(I_SHOULD)} stickies with action=\"read\" and section=\"{section}\".")
    elif action == "write":
        edits = args.get("edits", [])
        ops = [e.get("op", "") for e in edits]
        if len(edits) == 1:
            e = edits[0]
            op = e.get("op", "")
            if op == "add_todo":
                item = e.get("item", "")
                due = e.get("due")
                lines.append(f"{pick(USER_WANTS)} to add \"{item}\" to their to-do list" +
                             (f" with a due date of {due}" if due else "") + ".")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/add_todo edit.")
            elif op == "remove_todo":
                match = e.get("match", "")
                lines.append(f"{pick(USER_WANTS)} to remove \"{match}\" from their to-do list.")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/remove_todo edit matching \"{match}\".")
            elif op == "edit_todo":
                old = e.get("old", "")
                new = e.get("new", "")
                lines.append(f"{pick(USER_WANTS)} to change \"{old}\" to \"{new}\" in their to-do list.")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/edit_todo operation.")
            elif op == "add_note":
                sub = e.get("subheading", "")
                lines.append(f"{pick(USER_WANTS)} to save a note about \"{sub}\".")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/add_note edit.")
            elif op == "remove_note":
                match = e.get("match", "")
                lines.append(f"{pick(USER_WANTS)} to remove the note about \"{match}\".")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/remove_note edit.")
            elif op == "edit_note":
                old = e.get("old", "")
                lines.append(f"{pick(USER_WANTS)} to update a note (changing \"{old}\").")
                lines.append(f"{pick(I_SHOULD)} stickies with a write/edit_note operation.")
        else:
            if all(o == "add_todo" for o in ops):
                items = [e.get("item", "") for e in edits]
                lines.append(f"{pick(USER_WANTS)} to add multiple items to their to-do: {', '.join(items)}.")
                lines.append(f"{pick(I_SHOULD)} stickies with {len(edits)} add_todo edits in one call.")
            else:
                lines.append(f"{pick(USER_WANTS)} to make {len(edits)} edits to their sticky note.")
                lines.append(f"{pick(I_SHOULD)} stickies with multiple edit operations.")

    return " ".join(lines)


def think_sms(args, user_prompt):
    msg = args.get("message", "")
    lines = []
    lines.append(f"{pick(USER_WANTS)} a text message sent to their phone.")
    lines.append(f"{pick(I_SHOULD)} send_sms with a message about: \"{msg[:50]}\".")
    return " ".join(lines)


def think_search(args, user_prompt):
    query = args.get("query", "")
    qtype = args.get("query_type", "general")
    lines = []

    if qtype == "directions":
        lines.append(f"{pick(USER_WANTS)} directions/navigation.")
        lines.append(f"{pick(I_SHOULD)} google_search with query=\"{query}\" and query_type=\"directions\".")
    elif qtype == "link":
        lines.append(f"{pick(USER_WANTS)} a website link/URL.")
        lines.append(f"{pick(I_SHOULD)} google_search with query=\"{query}\" and query_type=\"link\".")
    else:
        lines.append(f"{pick(USER_WANTS)} to search the web for information.")
        lines.append(f"{pick(I_SHOULD)} google_search with query=\"{query}\".")

    return " ".join(lines)


def think_clipboard(args, user_prompt):
    lines = []
    lines.append(f"{pick(USER_WANTS)} to see what's on their clipboard.")
    lines.append(f"{pick(I_SHOULD)} read_clipboard. The clipboard content is unknown until read, so this must be a standalone call.")
    return " ".join(lines)


def think_briefing(args, user_prompt):
    action = args.get("action", "list")
    lines = []

    if action == "create":
        msg = args.get("message", "")
        remind_at = args.get("remind_at")
        event_time = args.get("event_time")
        before_min = args.get("remind_before_minutes")
        if remind_at:
            lines.append(f"{pick(USER_WANTS)} a reminder at {remind_at}: \"{msg}\".")
            lines.append(f"{pick(I_SHOULD)} briefing with action=\"create\", remind_at=\"{remind_at}\".")
        elif event_time and before_min:
            lines.append(f"{pick(USER_WANTS)} a reminder {before_min} minutes before {event_time}: \"{msg}\".")
            lines.append(f"{pick(I_SHOULD)} briefing with action=\"create\", event_time=\"{event_time}\", remind_before_minutes={before_min}.")
        else:
            lines.append(f"{pick(USER_WANTS)} to create a reminder: \"{msg}\".")
            lines.append(f"{pick(I_SHOULD)} briefing with action=\"create\".")
    elif action == "list":
        lines.append(f"{pick(USER_WANTS)} to see pending reminders/briefings.")
        lines.append(f"{pick(I_SHOULD)} briefing with action=\"list\".")
    elif action == "dismiss":
        lines.append(f"{pick(USER_WANTS)} to cancel/dismiss a briefing.")
        lines.append(f"{pick(I_SHOULD)} briefing with action=\"dismiss\".")

    return " ".join(lines)


def think_notifications(args, user_prompt):
    type_filter = args.get("type_filter", "all")
    limit = args.get("limit")
    lines = []

    filter_desc = {
        "email": "email notifications",
        "news": "news updates",
        "other": "other notifications",
        "all": "all notifications",
    }
    desc = filter_desc.get(type_filter, "notifications")
    lines.append(f"{pick(USER_WANTS)} to check their {desc}.")
    detail = f"type_filter=\"{type_filter}\""
    if limit:
        detail += f", limit={limit}"
    lines.append(f"{pick(I_SHOULD)} get_notifications with {detail}.")

    return " ".join(lines)


def think_system_info(args, user_prompt):
    section = args.get("section", "overview")
    lines = []
    section_desc = {
        "overview": "how the assistant works",
        "architecture": "the system architecture",
        "providers": "the provider system",
        "orchestrator": "the orchestrator/state machine",
        "audio": "the audio pipeline",
        "tools": "available tools",
        "memory": "the memory system",
        "config": "configuration details",
        "all": "everything about the system",
    }
    desc = section_desc.get(section, section)
    lines.append(f"{pick(USER_WANTS)} information about {desc}.")
    lines.append(f"{pick(I_SHOULD)} system_info with section=\"{section}\".")
    return " ".join(lines)


def think_cursor(args, user_prompt):
    prompt = args.get("prompt", "")
    lines = []
    lines.append(f"{pick(USER_WANTS)} to make code changes: \"{prompt[:60]}\".")
    lines.append(f"{pick(I_SHOULD)} cursor_composer to send this to Cursor's Composer.")
    return " ".join(lines)


# Dispatch table
TOOL_THINKERS = {
    "weather": think_weather,
    "spotify_playback": think_spotify,
    "kasa_lighting": think_kasa,
    "calendar_data": think_calendar,
    "stickies": think_stickies,
    "send_sms": think_sms,
    "google_search": think_search,
    "read_clipboard": think_clipboard,
    "briefing": think_briefing,
    "get_notifications": think_notifications,
    "system_info": think_system_info,
    "cursor_composer": think_cursor,
}


def generate_thinking_trace(user_prompt: str, tool_calls: list) -> str:
    """Generate a <think> trace for a given user prompt and tool call list."""
    if not tool_calls:
        # Negative example
        no_tool_reasons = [
            "This is a short acknowledgment with no actionable request. No tool call is needed.",
            "The user is just responding/acknowledging, not making a request that requires a tool.",
            "This doesn't contain a task or query that maps to any tool. Returning an empty array.",
            "No tool is needed here — this is a conversational response, not a command.",
            "This is just an acknowledgment. No tool call required.",
        ]
        return random.choice(no_tool_reasons)

    if len(tool_calls) == 1:
        tc = tool_calls[0]
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        thinker = TOOL_THINKERS.get(name)
        if thinker:
            return thinker(args, user_prompt)
        return f"Using {name} tool."

    # Multi-tool call
    parts = []
    tool_names = [tc["name"] for tc in tool_calls]

    # Opening line describing the multi-request
    multi_openers = [
        f"The user is making a multi-part request that needs {len(tool_calls)} tools.",
        f"This request requires {len(tool_calls)} parallel tool calls.",
        f"Multiple actions needed here — {len(tool_calls)} tools.",
        f"The user wants several things done at once. I'll need {len(tool_calls)} tool calls.",
    ]
    parts.append(random.choice(multi_openers))

    # Individual tool reasoning (shorter for multi-tool)
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if name == "weather":
            if "specific_date" in args:
                parts.append(f"Weather for {args['specific_date']} → weather tool.")
            elif "days" in args:
                parts.append(f"{args['days']}-day weather → weather tool.")
            elif "hours" in args:
                parts.append(f"{args['hours']}-hour forecast → weather tool.")
        elif name == "spotify_playback":
            action = args.get("action", "play")
            query = args.get("query")
            if action == "play" and query:
                parts.append(f"Play \"{query}\" → spotify_playback.")
            elif action == "play":
                parts.append(f"Resume music → spotify_playback.")
            elif action == "pause":
                parts.append(f"Stop/pause music → spotify_playback.")
            elif action == "volume":
                parts.append(f"Volume to {args.get('volume_level')}% → spotify_playback.")
            else:
                parts.append(f"Spotify {action} → spotify_playback.")
        elif name == "kasa_lighting":
            interaction = args.get("interaction", "direct")
            if interaction == "scene":
                parts.append(f"{args.get('scene_name', 'scene')} lights in {args.get('room', 'room')} → kasa_lighting scene.")
            else:
                parts.append(f"Light {args.get('action', 'on')} ({args.get('light_name', 'Light 1')}) → kasa_lighting direct.")
        elif name == "calendar_data":
            cmds = args.get("commands", [{}])
            cmd = cmds[0] if cmds else {}
            rw = cmd.get("read_or_write", "read")
            if rw == "read":
                rt = cmd.get("read_type", "next_events")
                date = cmd.get("date")
                if rt == "specific_date" and date:
                    parts.append(f"Calendar for {date} → calendar_data.")
                elif rt == "day_summary":
                    parts.append(f"Today's schedule → calendar_data.")
                elif rt == "week_summary":
                    parts.append(f"Week schedule → calendar_data.")
                else:
                    parts.append(f"Upcoming events → calendar_data.")
            else:
                parts.append(f"Create event \"{cmd.get('event_title', 'event')}\" → calendar_data.")
        elif name == "stickies":
            action = args.get("action", "read")
            if action == "read":
                parts.append(f"Read {args.get('section', 'both')} from stickies → stickies.")
            else:
                edits = args.get("edits", [])
                if edits:
                    op = edits[0].get("op", "edit")
                    item = edits[0].get("item", edits[0].get("match", ""))
                    parts.append(f"{op} \"{item}\" → stickies.")
                else:
                    parts.append(f"Edit stickies → stickies.")
        elif name == "send_sms":
            parts.append(f"Send text reminder → send_sms.")
        elif name == "google_search":
            parts.append(f"Search \"{args.get('query', '')}\" → google_search.")
        elif name == "read_clipboard":
            parts.append(f"Read clipboard → read_clipboard.")
        elif name == "briefing":
            action = args.get("action", "list")
            if action == "create":
                parts.append(f"Create reminder → briefing.")
            elif action == "list":
                parts.append(f"List reminders → briefing.")
            else:
                parts.append(f"Dismiss briefing → briefing.")
        elif name == "get_notifications":
            tf = args.get("type_filter", "all")
            parts.append(f"Check {tf} notifications → get_notifications.")
        elif name == "system_info":
            parts.append(f"System info → system_info.")
        elif name == "cursor_composer":
            parts.append(f"Code edit → cursor_composer.")

    return " ".join(parts)


def main():
    random.seed(42)  # Reproducible

    input_path = SCRIPT_DIR / "training_data.jsonl"
    output_path = SCRIPT_DIR / "training_data.jsonl"  # Overwrite in place

    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Read {len(examples)} examples")

    modified = 0
    for ex in examples:
        msgs = ex["messages"]
        # Process all assistant messages
        for msg in msgs:
            if msg["role"] == "assistant":
                tool_calls = json.loads(msg["content"])
                # Find the preceding user message
                user_prompt = ""
                for m in msgs:
                    if m["role"] == "user":
                        user_prompt = m["content"]

                thinking = generate_thinking_trace(user_prompt, tool_calls)
                # Format: <think>\nreasoning\n</think>\n[tool calls]
                new_content = f"<think>\n{thinking}\n</think>\n{msg['content']}"
                msg["content"] = new_content
                modified += 1

    # Write back
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Modified {modified} assistant messages with thinking traces")
    print(f"Output: {output_path}")

    # Show a few samples
    print("\n=== Sample outputs ===")
    sample_indices = [0, 5, 50, 300, len(examples)-1]
    for i in sample_indices:
        if i < len(examples):
            msgs = examples[i]["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
            print(f"\n--- Example {i} ---")
            print(f"User: {user_msg}")
            print(f"Assistant: {asst_msg[:300]}...")


if __name__ == "__main__":
    main()
