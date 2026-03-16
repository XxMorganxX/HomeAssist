#!/usr/bin/env python3
"""Upgrade thinking traces in training_data.jsonl for QWEN3-instruct finetuning."""

import json

Q = '"'  # quote helper for f-strings

def generate_thinking_trace(user_msg, tools):
    """Generate a comprehensive thinking trace based on user message and expected tool calls."""
    
    if not tools:
        return (
            f"The user said {Q}{user_msg}{Q} which is a conversational response - "
            f"an acknowledgment, greeting, or casual remark. "
            f"This does not contain any actionable request that maps to an available tool. "
            f"No tool call is needed; I should return an empty array."
        )
    
    is_multi = len(tools) > 1
    parts = []
    
    if is_multi:
        parts.append(
            f"The user is making a multi-part request: {Q}{user_msg}{Q}. "
            f"I need to break this down and determine which tools to call for each part."
        )
    
    for tool in tools:
        name = tool["name"]
        args = tool["arguments"]
        
        # WEATHER
        if name == "weather":
            if "specific_date" in args:
                dv = args["specific_date"]
                if dv == "today":
                    parts.append(
                        "The user is asking about today's weather specifically. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        "Since they mentioned today explicitly, I will use specific_date='today' for a focused daily forecast."
                    )
                elif dv == "tomorrow":
                    parts.append(
                        "The user wants the weather for tomorrow. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        "Since they are asking about a specific future day, I will use specific_date='tomorrow'. "
                        "The tool supports natural language date terms."
                    )
                elif dv in ("monday","tuesday","wednesday","thursday","friday","saturday","sunday"):
                    parts.append(
                        f"The user is asking about the weather for {dv}. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        f"Since they specified a day of the week, I will use specific_date='{dv}'. "
                        "The tool accepts relative day names."
                    )
                else:
                    parts.append(
                        f"The user wants the weather for a specific date: {dv}. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        f"I will use specific_date='{dv}' in the appropriate format."
                    )
            elif "days" in args:
                dv = args["days"]
                if dv == 1:
                    parts.append(
                        "The user is asking about the current weather or a general forecast. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        "Since this is a general weather query without a specific date or hourly request, "
                        "I will use days=1 to get today's forecast overview."
                    )
                else:
                    parts.append(
                        f"The user wants a multi-day weather forecast covering {dv} days. "
                        "The weather tool requires exactly one of: hours, days, or specific_date. "
                        f"I will use days={dv} to return daily forecasts for the next {dv} days."
                    )
            elif "hours" in args:
                hv = args["hours"]
                parts.append(
                    f"The user wants an hourly or short-range weather forecast for the next {hv} hours. "
                    "The weather tool requires exactly one of: hours, days, or specific_date. "
                    f"Since they want granular hourly detail, I will use hours={hv}. "
                    "The tool provides hourly forecasts when the timeframe is 36 hours or less."
                )
        
        # SPOTIFY PLAYBACK
        elif name == "spotify_playback":
            action = args.get("action", "")
            query = args.get("query", "")
            search_type = args.get("search_type", "")
            volume = args.get("volume_level")
            shuffle = args.get("shuffle_state")
            repeat_mode = args.get("repeat_mode")
            
            if action == "play" and not query:
                parts.append(
                    "The user wants to resume or start music playback. "
                    "There is no specific song or artist mentioned, so this is a simple resume command. "
                    "I will use spotify_playback with action='play' and no query, which resumes current playback."
                )
            elif action == "play" and query and search_type == "artist":
                parts.append(
                    f"The user wants to listen to music by the artist {Q}{query}{Q}. "
                    "Since they referenced an artist name rather than a specific song title, "
                    f"I will use spotify_playback with action='play', query='{query}', and search_type='artist' "
                    "to find and play music by this artist."
                )
            elif action == "play" and query and search_type == "track":
                parts.append(
                    f"The user wants to play a specific song: {Q}{query}{Q}. "
                    "Since they referenced a specific track title, "
                    f"I will use spotify_playback with action='play', query='{query}', and search_type='track' "
                    "to search for and play this song."
                )
            elif action == "play" and query:
                parts.append(
                    f"The user wants to play {Q}{query}{Q}. "
                    f"I will use spotify_playback with action='play' and query='{query}' to search and play this."
                )
            elif action == "pause":
                parts.append(
                    "The user wants to pause the music. "
                    "I will use spotify_playback with action='pause' to pause current playback."
                )
            elif action == "next":
                parts.append(
                    "The user wants to skip to the next track. "
                    "I will use spotify_playback with action='next' to advance to the next song."
                )
            elif action == "previous":
                parts.append(
                    "The user wants to go back to the previous track. "
                    "I will use spotify_playback with action='previous' to return to the prior song."
                )
            elif action == "volume" and volume is not None:
                parts.append(
                    f"The user wants to change the volume to {volume}%. "
                    f"I will use spotify_playback with action='volume' and volume_level={volume}. "
                    "The volume range is 0-100."
                )
            elif action == "shuffle" and shuffle is True:
                parts.append(
                    "The user wants to enable shuffle mode. "
                    "I will use spotify_playback with action='shuffle' and shuffle_state=true to turn on shuffling."
                )
            elif action == "shuffle" and shuffle is False:
                parts.append(
                    "The user wants to disable shuffle mode and play tracks in order. "
                    "I will use spotify_playback with action='shuffle' and shuffle_state=false to turn off shuffling."
                )
            elif action == "shuffle":
                parts.append(
                    "The user wants to toggle shuffle mode. "
                    "I will use spotify_playback with action='shuffle' without specifying shuffle_state, which toggles the current state."
                )
            elif action == "repeat":
                if repeat_mode:
                    mode_desc = {"off": "turn off repeat", "track": "loop the current song", "context": "loop the entire playlist/album"}
                    desc = mode_desc.get(repeat_mode, "change repeat mode")
                    parts.append(
                        f"The user wants to {desc}. "
                        f"I will use spotify_playback with action='repeat' and repeat_mode='{repeat_mode}'."
                    )
                else:
                    parts.append(
                        "The user wants to toggle or cycle through repeat modes. "
                        "I will use spotify_playback with action='repeat' without specifying a mode, which cycles through off/track/context."
                    )
            elif action == "status":
                parts.append(
                    "The user wants to know what is currently playing. "
                    "I will use spotify_playback with action='status' to get current playback info."
                )
            elif action == "search_track":
                parts.append(
                    f"The user wants to search for tracks matching {Q}{query}{Q}. "
                    f"I will use spotify_playback with action='search_track' and query='{query}'."
                )
            elif action == "search_artist":
                parts.append(
                    f"The user wants to search for artists matching {Q}{query}{Q}. "
                    f"I will use spotify_playback with action='search_artist' and query='{query}'."
                )
            elif action == "devices":
                parts.append(
                    "The user wants to see available Spotify playback devices. "
                    "I will use spotify_playback with action='devices' to list connected devices."
                )
            else:
                parts.append(f"The user wants a Spotify action. I will use action='{action}'.")
        
        # KASA LIGHTING
        elif name == "kasa_lighting":
            interaction = args.get("interaction", "")
            light_name = args.get("light_name", "")
            room = args.get("room", "")
            action = args.get("action", "")
            scene_name = args.get("scene_name", "")
            
            if interaction == "direct":
                on_off = "turn on" if action == "on" else "turn off"
                parts.append(
                    f"The user wants to {on_off} a light. "
                    "This is direct control of a single light, not a scene preset. "
                    f"I will use kasa_lighting with interaction='direct', light_name='{light_name}', and action='{action}'."
                )
            elif interaction == "scene":
                room_part = f" in the {room}" if room else ""
                parts.append(
                    f"The user wants to apply a '{scene_name}' lighting scene{room_part}. "
                    "This is a scene preset that configures multiple lights with appropriate settings. "
                    f"I will use kasa_lighting with interaction='scene', scene_name='{scene_name}'"
                    f"{', room=' + Q + room + Q if room else ''}."
                )
        
        # CALENDAR DATA
        elif name == "calendar_data":
            commands = args.get("commands", [{}])
            cmd = commands[0] if commands else {}
            rw = cmd.get("read_or_write", "read")
            read_type = cmd.get("read_type", "")
            cal = cmd.get("calendar", "all")
            date = cmd.get("date", "")
            title = cmd.get("event_title", "")
            start = cmd.get("start_time", "")
            end = cmd.get("end_time", "")
            loc = cmd.get("location", "")
            
            if rw == "read":
                if read_type == "next_events":
                    cal_desc = "all calendars" if cal == "all" else f"the {cal} calendar"
                    parts.append(
                        f"The user wants to see their upcoming calendar events. "
                        f"I will use calendar_data with read_type='next_events' and calendar='{cal}' "
                        f"to fetch the next scheduled events across {cal_desc}."
                    )
                elif read_type == "day_summary":
                    day_desc = date if date else "today"
                    parts.append(
                        f"The user wants a summary of events for {day_desc}. "
                        f"I will use calendar_data with read_type='day_summary'"
                        f"{', date=' + Q + date + Q if date else ''} and calendar='{cal}'."
                    )
                elif read_type == "week_summary":
                    parts.append(
                        "The user wants a weekly calendar overview. "
                        f"I will use calendar_data with read_type='week_summary' and calendar='{cal}'."
                    )
                elif read_type == "specific_date":
                    parts.append(
                        f"The user wants calendar events for {date}. "
                        f"I will use calendar_data with read_type='specific_date', date='{date}', and calendar='{cal}'."
                    )
                else:
                    parts.append(
                        "The user wants to check their calendar. "
                        "I will use calendar_data with a read command to fetch events."
                    )
            elif rw == "create_event":
                time_str = f" at {start}" if start else ""
                end_str = f" to {end}" if end else ""
                loc_str = f" at {loc}" if loc else ""
                date_str = f" on {date}" if date else ""
                time_note = "The user specified a time, so I will include it." if start else "The user did not specify a time, so I will not invent one."
                parts.append(
                    f"The user wants to create a calendar event: '{title}'{date_str}{time_str}{end_str}{loc_str}. "
                    f"I will use calendar_data with read_or_write='create_event' and event_title='{title}'. "
                    f"{time_note}"
                )
        
        # STICKIES
        elif name == "stickies":
            saction = args.get("action", "")
            section = args.get("section", "both")
            edits = args.get("edits", [])
            
            if saction == "read":
                section_map = {"notes": "their notes section", "todo": "their to-do list", "both": "their full sticky note (notes and to-do list)"}
                parts.append(
                    f"The user wants to read {section_map.get(section, 'their sticky note')}. "
                    f"I will use stickies with action='read' and section='{section}'."
                )
            elif saction == "write":
                edit_descs = []
                for edit in edits:
                    op = edit.get("op", "")
                    if op == "add_todo":
                        item = edit.get("item", "")
                        due = edit.get("due", "")
                        due_str = f" (due {due})" if due else ""
                        edit_descs.append(f"add '{item}'{due_str} to the to-do list")
                    elif op == "remove_todo":
                        edit_descs.append(f"remove to-do matching '{edit.get('match', '')}'")
                    elif op == "edit_todo":
                        edit_descs.append(f"edit to-do '{edit.get('old', '')}' to '{edit.get('new', '')}'")
                    elif op == "add_note":
                        edit_descs.append(f"add a note under '{edit.get('subheading', '')}'")
                    elif op == "remove_note":
                        edit_descs.append(f"remove note matching '{edit.get('match', '')}'")
                    elif op == "edit_note":
                        edit_descs.append(f"edit note '{edit.get('old', '')}' to '{edit.get('new', '')}'")
                
                desc_str = "; ".join(edit_descs) if edit_descs else "make edits"
                parts.append(
                    f"The user wants to update their sticky note: {desc_str}. "
                    "I will use stickies with action='write' and the appropriate edit operations. "
                    "Each edit requires an 'op' field specifying the operation type."
                )
        
        # SEND SMS
        elif name == "send_sms":
            message = args.get("message", "")
            parts.append(
                "The user wants to send themselves a text message as a reminder. "
                f"I will use send_sms with message='{message}'. "
                "This sends an iMessage to the user's configured phone number."
            )
        
        # GOOGLE SEARCH
        elif name == "google_search":
            query = args.get("query", "")
            query_type = args.get("query_type", "general")
            type_map = {"general": "an informational query", "link": "looking for a specific URL", "directions": "a navigation/directions query"}
            parts.append(
                f"The user wants to search the web - this is {type_map.get(query_type, 'a search')}. "
                f"I will use google_search with query='{query}' and query_type='{query_type}'. "
                "The search tool should only be called once per user request."
            )
        
        # READ CLIPBOARD
        elif name == "read_clipboard":
            parts.append(
                "The user wants to read the contents of their system clipboard. "
                "I will use read_clipboard to retrieve what they last copied. "
                "No additional parameters are needed."
            )
        
        # BRIEFING
        elif name == "briefing":
            baction = args.get("action", "")
            message = args.get("message", "")
            remind_at = args.get("remind_at", "")
            remind_before = args.get("remind_before_minutes")
            event_time = args.get("event_time", "")
            priority = args.get("priority", "")
            bid = args.get("briefing_id", "")
            
            if baction == "create":
                timing = ""
                if remind_at:
                    timing = f" scheduled for {remind_at}"
                elif remind_before and event_time:
                    timing = f" {remind_before} minutes before the event"
                
                pri_str = f" Priority: {priority}." if priority and priority != "normal" else ""
                parts.append(
                    f"The user wants to create a briefing reminder{timing}. "
                    "Briefings are spoken announcements delivered when the user next wakes the assistant. "
                    f"I will use briefing with action='create' and message='{message}'"
                    f"{', remind_at=' + Q + remind_at + Q if remind_at else ''}"
                    f"{', remind_before_minutes=' + str(remind_before) if remind_before else ''}"
                    f".{pri_str}"
                )
            elif baction == "list":
                parts.append(
                    "The user wants to see their pending briefings and reminders. "
                    "I will use briefing with action='list' to retrieve all upcoming briefings."
                )
            elif baction == "dismiss":
                parts.append(
                    "The user wants to dismiss or cancel a briefing. "
                    f"I will use briefing with action='dismiss'"
                    f"{' and briefing_id=' + Q + bid + Q if bid else ''}."
                )
        
        # GET NOTIFICATIONS
        elif name == "get_notifications":
            type_filter = args.get("type_filter", "all")
            filter_map = {"email": "email summaries", "news": "news updates", "all": "all notifications (emails and news)", "other": "other notifications"}
            parts.append(
                f"The user wants to check their {filter_map.get(type_filter, 'notifications')}. "
                f"I will use get_notifications with type_filter='{type_filter}' to retrieve pending notifications."
            )
        
        # SYSTEM INFO
        elif name == "system_info":
            section = args.get("section", "overview")
            section_map = {
                "overview": "a general overview of the assistant",
                "architecture": "the system architecture details",
                "providers": "the provider system",
                "orchestrator": "the orchestrator and state machine",
                "audio": "the audio pipeline",
                "tools": "the available tools",
                "memory": "the memory systems",
                "config": "the configuration system",
                "all": "all documentation sections"
            }
            parts.append(
                f"The user is asking about the assistant's internals - they want {section_map.get(section, 'information')}. "
                f"I will use system_info with section='{section}' to retrieve the relevant documentation."
            )
        
        # CURSOR COMPOSER
        elif name == "cursor_composer":
            parts.append(
                "The user wants to perform a coding task which should be sent to Cursor's Composer. "
                "I will use cursor_composer with the request as the prompt. "
                "Composer handles multi-file code edits on macOS."
            )
        
        else:
            parts.append(f"Using the {name} tool with the provided arguments.")
    
    return " ".join(parts)


def main():
    input_file = "training_data.jsonl"
    output_file = "training_data_upgraded.jsonl"
    
    with open(input_file) as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} entries...")
    
    updated = 0
    errors = 0
    output_lines = []
    
    for i, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
            msgs = data["messages"]
            user_msg = msgs[1]["content"]
            assistant_msg = msgs[2]["content"]
            
            think_end = assistant_msg.find("</think>")
            if think_end >= 0:
                tool_json_str = assistant_msg[think_end + 8:].strip()
            else:
                tool_json_str = assistant_msg.strip()
            
            try:
                tools = json.loads(tool_json_str)
            except json.JSONDecodeError:
                print(f"  WARNING: Line {i+1} has invalid JSON, skipping")
                output_lines.append(line.strip())
                errors += 1
                continue
            
            new_think = generate_thinking_trace(user_msg, tools)
            new_assistant = f"<think>\n{new_think}\n</think>\n{tool_json_str}"
            
            msgs[2]["content"] = new_assistant
            data["messages"] = msgs
            
            output_lines.append(json.dumps(data, ensure_ascii=False))
            updated += 1
            
        except Exception as e:
            print(f"  ERROR on line {i+1}: {e}")
            output_lines.append(line.strip())
            errors += 1
    
    with open(output_file, "w") as f:
        for ol in output_lines:
            f.write(ol + "\n")
    
    print(f"\nDone! Updated {updated} entries, {errors} errors.")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
