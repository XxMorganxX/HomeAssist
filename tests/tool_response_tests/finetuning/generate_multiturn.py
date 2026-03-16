#!/usr/bin/env python3
"""
Generate multi-turn conversation examples for fine-tuning.
All examples are tool-calling only - no conversational responses.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Load tool schemas and system prompt
with open(SCRIPT_DIR / "tool_schemas.json") as f:
    TOOL_SCHEMAS = json.load(f)

with open(SCRIPT_DIR / "system_prompt.txt") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{tools}", 
    json.dumps(TOOL_SCHEMAS["tools"], indent=2)
)

# Multi-turn conversations: list of (user, assistant_tool_calls) tuples
# ALL responses must be tool calls - this model is specialized for tool calling only
MULTI_TURN_CONVERSATIONS = [
    # Weather follow-ups
    [
        ("What's the weather today?", [{"name": "weather", "arguments": {"days": 1}}]),
        ("What about tomorrow?", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
        ("And Friday?", [{"name": "weather", "arguments": {"specific_date": "friday"}}]),
        ("Give me the whole week", [{"name": "weather", "arguments": {"days": 7}}]),
    ],
    [
        ("5 day forecast", [{"name": "weather", "arguments": {"days": 5}}]),
        ("What about Saturday specifically?", [{"name": "weather", "arguments": {"specific_date": "saturday"}}]),
    ],
    [
        ("Is it going to rain?", [{"name": "weather", "arguments": {"days": 1}}]),
        ("What about this weekend?", [{"name": "weather", "arguments": {"specific_date": "saturday"}}]),
    ],
    
    # Music control flow
    [
        ("Play some jazz", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}}]),
        ("Turn it up", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 80}}]),
        ("Next song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
        ("Pause it", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ],
    [
        ("What's playing?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
        ("Skip this one", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
        ("Lower the volume to 30", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
    ],
    [
        ("Put on Taylor Swift", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Taylor Swift", "search_type": "artist"}}]),
        ("Actually, play The Beatles instead", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "The Beatles", "search_type": "artist"}}]),
        ("Turn on shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
    ],
    [
        ("Play Bohemian Rhapsody", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Bohemian Rhapsody", "search_type": "track"}}]),
        ("Put it on repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
        ("Volume 50", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 50}}]),
    ],
    [
        ("Play some lo-fi", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi", "search_type": "artist"}}]),
        ("This is good, what song is it?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
        ("Skip to the next", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ],
    
    # Light scenes
    [
        ("Movie mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
        ("Actually make it brighter", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
        ("Turn off the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ],
    [
        ("Turn on the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
        ("Set reading mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
        ("Now turn them off", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ],
    [
        ("Relax mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
        ("Switch to party mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
    ],
    
    # Calendar operations
    [
        ("What's on my calendar?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
        ("Just show today", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
        ("Schedule lunch tomorrow at noon", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch", "start_time": "tomorrow 12pm"}]}}]),
    ],
    [
        ("Add a meeting at 3pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "3pm"}]}}]),
        ("What does my week look like?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ],
    [
        ("What's tomorrow looking like?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
        ("And Friday?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "friday"}]}}]),
    ],
    
    # To-do list management
    [
        ("Show my to-do list", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
        ("Add buy groceries", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "buy groceries"}]}}]),
        ("Also add call mom", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "call mom"}]}}]),
        ("Remove buy groceries", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "buy groceries"}]}}]),
    ],
    [
        ("What notes do I have?", [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]),
        ("Add a note: meeting password is 12345", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "meeting password", "content": "12345"}]}}]),
    ],
    [
        ("Add milk to my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "milk"}]}}]),
        ("Add eggs too", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "eggs"}]}}]),
        ("And bread", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "bread"}]}}]),
        ("Show me the full list", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ],
    
    # Mixed tool operations
    [
        ("What's the weather?", [{"name": "weather", "arguments": {"days": 1}}]),
        ("Play some music", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
        ("Dim the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ],
    [
        ("Check my calendar for tomorrow", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
        ("And the weather?", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ],
    [
        ("Any emails?", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
        ("What about news?", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ],
    
    # Interruptions and corrections
    [
        ("Play jazz music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}}]),
        ("Wait, I meant classical", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "classical", "search_type": "artist"}}]),
    ],
    [
        ("Turn on movie mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
        ("Never mind, just turn off the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ],
    [
        ("Schedule a meeting at 2pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2pm"}]}}]),
        ("Actually make it 3pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "3pm"}]}}]),
    ],
    
    # Search follow-ups
    [
        ("Search for Python tutorials", [{"name": "google_search", "arguments": {"query": "Python tutorials", "query_type": "general"}}]),
        ("Find the link to Python.org", [{"name": "google_search", "arguments": {"query": "Python.org", "query_type": "link"}}]),
    ],
    [
        ("Directions to the nearest Starbucks", [{"name": "google_search", "arguments": {"query": "nearest Starbucks", "query_type": "directions"}}]),
        ("What about McDonald's?", [{"name": "google_search", "arguments": {"query": "nearest McDonald's", "query_type": "directions"}}]),
    ],
    
    # Briefing/reminder flow
    [
        ("Remind me to call the dentist at 2pm", [{"name": "briefing", "arguments": {"action": "create", "message": "Call the dentist", "remind_at": "2pm"}}]),
        ("What reminders do I have?", [{"name": "briefing", "arguments": {"action": "list"}}]),
        ("Cancel the dentist reminder", [{"name": "briefing", "arguments": {"action": "dismiss", "briefing_id": "dentist"}}]),
    ],
    [
        ("Set a reminder for 9am tomorrow", [{"name": "briefing", "arguments": {"action": "create", "message": "Morning reminder", "remind_at": "tomorrow 9am"}}]),
        ("List my briefings", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ],
    
    # Notifications flow
    [
        ("Any new emails?", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
        ("What about news?", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
        ("Show me everything", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
    ],
    
    # Morning routine sequences
    [
        ("Weather today", [{"name": "weather", "arguments": {"days": 1}}]),
        ("Calendar", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
        ("Any emails?", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
        ("Play some morning music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "morning music", "search_type": "artist"}}]),
    ],
    
    # Evening wind-down
    [
        ("Relax mode lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
        ("Play soft jazz", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "soft jazz", "search_type": "artist"}}]),
        ("Volume down to 25", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 25}}]),
    ],
]


def generate_conversation_examples(conversation):
    """Generate training examples from a multi-turn conversation."""
    examples = []
    messages_so_far = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for user_msg, assistant_output in conversation:
        messages_so_far.append({"role": "user", "content": user_msg})
        
        # Create training example with full context
        example = {
            "messages": messages_so_far.copy() + [
                {"role": "assistant", "content": json.dumps(assistant_output)}
            ]
        }
        examples.append(example)
        
        # Add assistant response to context for next turn
        messages_so_far.append({"role": "assistant", "content": json.dumps(assistant_output)})
    
    return examples


def main():
    output_path = SCRIPT_DIR / "multiturn_training_data.jsonl"
    
    all_examples = []
    for conversation in MULTI_TURN_CONVERSATIONS:
        all_examples.extend(generate_conversation_examples(conversation))
    
    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Generated {len(all_examples)} multi-turn training examples")
    print(f"From {len(MULTI_TURN_CONVERSATIONS)} conversations")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
