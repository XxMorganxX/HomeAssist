#!/usr/bin/env python3
"""
Generate comprehensive fine-tuning dataset for Qwen 8B tool-calling model.
This model is specialized ONLY for tool calling - no conversational responses.

All examples are validated against actual tool schemas.
Output: JSONL file suitable for LoRA fine-tuning via Ollama.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Load tool schemas
with open(SCRIPT_DIR / "tool_schemas.json") as f:
    TOOL_SCHEMAS = json.load(f)

# Load system prompt template
with open(SCRIPT_DIR / "system_prompt.txt") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

# Format system prompt with tool schemas
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{tools}", 
    json.dumps(TOOL_SCHEMAS["tools"], indent=2)
)

# =============================================================================
# TRAINING EXAMPLES
# Format: (user_prompt, expected_tool_calls)
# Note: This model is always invoked when a tool call is expected, so nearly
# all examples should produce tool calls. A minimal safety set of [] examples
# covers acknowledgments that may leak through after a tool response.
# =============================================================================

TRAINING_EXAMPLES = [
    # =========================================================================
    # NO TOOL NEEDED - minimal safety set (post-response acknowledgments)
    # The orchestrator gates what reaches this model, so these only cover
    # short acknowledgments that might leak through after a tool result.
    # =========================================================================
    ("Thanks", []),
    ("Thank you", []),
    ("Got it", []),
    ("Okay", []),
    ("Never mind", []),
    
    # =========================================================================
    # WEATHER TOOL - 50+ examples
    # Params: hours (1-168), days (1-7), specific_date (string)
    # Only ONE of these should be provided (oneOf)
    # =========================================================================
    
    # --- Days parameter ---
    ("What's the weather?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Weather", [{"name": "weather", "arguments": {"days": 1}}]),
    ("How's the weather?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Weather forecast", [{"name": "weather", "arguments": {"days": 1}}]),
    ("What's it like outside?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Is it cold out?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Is it hot today?", [{"name": "weather", "arguments": {"specific_date": "today"}}]),
    ("Will it rain?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Is it going to rain today?", [{"name": "weather", "arguments": {"specific_date": "today"}}]),
    ("Should I bring an umbrella?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Do I need a jacket?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("What's the temperature?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("How warm is it?", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Current weather conditions", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Weather report", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Give me the weather", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Tell me the weather", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Check weather", [{"name": "weather", "arguments": {"days": 1}}]),
    ("Weather check", [{"name": "weather", "arguments": {"days": 1}}]),
    ("How's it looking outside?", [{"name": "weather", "arguments": {"days": 1}}]),
    
    # Multi-day forecasts
    ("3 day forecast", [{"name": "weather", "arguments": {"days": 3}}]),
    ("5 day forecast", [{"name": "weather", "arguments": {"days": 5}}]),
    ("7 day forecast", [{"name": "weather", "arguments": {"days": 7}}]),
    ("Week forecast", [{"name": "weather", "arguments": {"days": 7}}]),
    ("Weekly weather", [{"name": "weather", "arguments": {"days": 7}}]),
    ("Weather for the week", [{"name": "weather", "arguments": {"days": 7}}]),
    ("This week's weather", [{"name": "weather", "arguments": {"days": 7}}]),
    ("Give me a 5 day forecast", [{"name": "weather", "arguments": {"days": 5}}]),
    ("What's the weather for the next 3 days?", [{"name": "weather", "arguments": {"days": 3}}]),
    ("Weather for the next few days", [{"name": "weather", "arguments": {"days": 3}}]),
    ("Extended forecast", [{"name": "weather", "arguments": {"days": 7}}]),
    ("Long range forecast", [{"name": "weather", "arguments": {"days": 7}}]),
    ("2 day forecast", [{"name": "weather", "arguments": {"days": 2}}]),
    ("4 day forecast", [{"name": "weather", "arguments": {"days": 4}}]),
    ("6 day forecast", [{"name": "weather", "arguments": {"days": 6}}]),
    
    # --- Specific date parameter ---
    ("Weather tomorrow", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ("Tomorrow's weather", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ("What's the weather tomorrow?", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ("How's the weather looking tomorrow?", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ("Will it rain tomorrow?", [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]),
    ("Weather on Friday", [{"name": "weather", "arguments": {"specific_date": "friday"}}]),
    ("What's the weather on Saturday?", [{"name": "weather", "arguments": {"specific_date": "saturday"}}]),
    ("Weather for Sunday", [{"name": "weather", "arguments": {"specific_date": "sunday"}}]),
    ("Monday's weather", [{"name": "weather", "arguments": {"specific_date": "monday"}}]),
    ("Tuesday weather", [{"name": "weather", "arguments": {"specific_date": "tuesday"}}]),
    ("Wednesday forecast", [{"name": "weather", "arguments": {"specific_date": "wednesday"}}]),
    ("Thursday's forecast", [{"name": "weather", "arguments": {"specific_date": "thursday"}}]),
    ("What's it going to be like on Friday?", [{"name": "weather", "arguments": {"specific_date": "friday"}}]),
    ("Will Saturday be nice?", [{"name": "weather", "arguments": {"specific_date": "saturday"}}]),
    ("Is Sunday going to be warm?", [{"name": "weather", "arguments": {"specific_date": "sunday"}}]),
    ("Weather today", [{"name": "weather", "arguments": {"specific_date": "today"}}]),
    ("Today's weather", [{"name": "weather", "arguments": {"specific_date": "today"}}]),
    ("What's the weather on December 25th?", [{"name": "weather", "arguments": {"specific_date": "2024-12-25"}}]),
    ("Weather for January 1st", [{"name": "weather", "arguments": {"specific_date": "2025-01-01"}}]),
    ("What's the forecast for next Monday?", [{"name": "weather", "arguments": {"specific_date": "monday"}}]),
    
    # --- Hours parameter ---
    ("Next 12 hours weather", [{"name": "weather", "arguments": {"hours": 12}}]),
    ("Weather for the next 24 hours", [{"name": "weather", "arguments": {"hours": 24}}]),
    ("Hourly forecast", [{"name": "weather", "arguments": {"hours": 24}}]),
    ("Hour by hour forecast", [{"name": "weather", "arguments": {"hours": 12}}]),
    ("Next 6 hours", [{"name": "weather", "arguments": {"hours": 6}}]),
    ("Weather for the next few hours", [{"name": "weather", "arguments": {"hours": 6}}]),
    ("What's the weather in the next 3 hours?", [{"name": "weather", "arguments": {"hours": 3}}]),
    ("36 hour forecast", [{"name": "weather", "arguments": {"hours": 36}}]),
    ("Weather for the next 8 hours", [{"name": "weather", "arguments": {"hours": 8}}]),
    ("Give me an hourly breakdown", [{"name": "weather", "arguments": {"hours": 12}}]),
    
    # =========================================================================
    # SPOTIFY PLAYBACK TOOL - 80+ examples
    # Params: action (required), query, search_type, volume_level
    # Actions: play, pause, next, previous, volume, search_track, search_artist, 
    #          status, devices, shuffle, repeat
    # =========================================================================
    
    # --- Play action (no query - resume) ---
    ("Play music", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Play", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Resume", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Resume music", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Resume playback", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Start the music", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Start playing", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Continue playing", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Unpause", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    ("Keep playing", [{"name": "spotify_playback", "arguments": {"action": "play"}}]),
    
    # --- Play action with query (track search) ---
    ("Play Bohemian Rhapsody", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Bohemian Rhapsody", "search_type": "track"}}]),
    ("Play Hotel California", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Hotel California", "search_type": "track"}}]),
    ("Play Stairway to Heaven", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Stairway to Heaven", "search_type": "track"}}]),
    ("Play Blinding Lights", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Blinding Lights", "search_type": "track"}}]),
    ("Play Shape of You", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Shape of You", "search_type": "track"}}]),
    ("Play Uptown Funk", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Uptown Funk", "search_type": "track"}}]),
    ("Play Bad Guy by Billie Eilish", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Bad Guy Billie Eilish", "search_type": "track"}}]),
    ("Play Someone Like You by Adele", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Someone Like You Adele", "search_type": "track"}}]),
    ("Put on Wonderwall", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Wonderwall", "search_type": "track"}}]),
    ("I want to hear Sweet Child O Mine", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Sweet Child O Mine", "search_type": "track"}}]),
    ("Can you play Thriller?", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Thriller", "search_type": "track"}}]),
    ("Play the song Imagine", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Imagine", "search_type": "track"}}]),
    ("Play Let It Be", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Let It Be", "search_type": "track"}}]),
    ("Play Smells Like Teen Spirit", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Smells Like Teen Spirit", "search_type": "track"}}]),
    ("Play Billie Jean", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Billie Jean", "search_type": "track"}}]),
    
    # --- Play action with query (artist search) ---
    ("Play Taylor Swift", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Taylor Swift", "search_type": "artist"}}]),
    ("Play The Beatles", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "The Beatles", "search_type": "artist"}}]),
    ("Play Drake", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Drake", "search_type": "artist"}}]),
    ("Play Ed Sheeran", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Ed Sheeran", "search_type": "artist"}}]),
    ("Play Adele", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Adele", "search_type": "artist"}}]),
    ("Play Coldplay", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Coldplay", "search_type": "artist"}}]),
    ("Play Kendrick Lamar", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Kendrick Lamar", "search_type": "artist"}}]),
    ("Play some Beyoncé", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Beyoncé", "search_type": "artist"}}]),
    ("Put on some Bob Marley", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Bob Marley", "search_type": "artist"}}]),
    ("I want to listen to Queen", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Queen", "search_type": "artist"}}]),
    ("Play music by The Weeknd", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "The Weeknd", "search_type": "artist"}}]),
    ("Play something by Dua Lipa", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Dua Lipa", "search_type": "artist"}}]),
    ("Play Ariana Grande", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Ariana Grande", "search_type": "artist"}}]),
    ("Play Post Malone", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "Post Malone", "search_type": "artist"}}]),
    
    # --- Play action with genre/mood (artist search type) ---
    ("Play jazz", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}}]),
    ("Play classical music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "classical music", "search_type": "artist"}}]),
    ("Play some rock", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "rock", "search_type": "artist"}}]),
    ("Play hip hop", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "hip hop", "search_type": "artist"}}]),
    ("Play country music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "country music", "search_type": "artist"}}]),
    ("Play pop music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "pop music", "search_type": "artist"}}]),
    ("Play electronic music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "electronic music", "search_type": "artist"}}]),
    ("Play lo-fi", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi", "search_type": "artist"}}]),
    ("Play lo-fi beats", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi beats", "search_type": "artist"}}]),
    ("Play chill music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "chill music", "search_type": "artist"}}]),
    ("Play relaxing music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "relaxing music", "search_type": "artist"}}]),
    ("Play study music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "study music", "search_type": "artist"}}]),
    ("Play focus music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "focus music", "search_type": "artist"}}]),
    ("Play workout music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "workout music", "search_type": "artist"}}]),
    ("Play party music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "party music", "search_type": "artist"}}]),
    ("Play sleep music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "sleep music", "search_type": "artist"}}]),
    ("Play meditation music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "meditation music", "search_type": "artist"}}]),
    ("Play R&B", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "R&B", "search_type": "artist"}}]),
    ("Play indie music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "indie music", "search_type": "artist"}}]),
    ("Play soft music", [{"name": "spotify_playback", "arguments": {"action": "play", "query": "soft music", "search_type": "artist"}}]),
    
    # --- Pause action ---
    ("Pause", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Pause the music", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Pause music", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Stop", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Stop the music", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Stop playing", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Pause playback", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Hold the music", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Shut up", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Quiet", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    ("Silence", [{"name": "spotify_playback", "arguments": {"action": "pause"}}]),
    
    # --- Next action ---
    ("Next", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Next song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Skip", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Skip song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Skip this song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Skip track", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Next track", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Play the next one", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("Go to the next song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    ("I don't like this song", [{"name": "spotify_playback", "arguments": {"action": "next"}}]),
    
    # --- Previous action ---
    ("Previous", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Previous song", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Go back", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Previous track", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Last song", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Play the previous song", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Go back a track", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Replay", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    ("Play that again", [{"name": "spotify_playback", "arguments": {"action": "previous"}}]),
    
    # --- Volume action ---
    ("Volume 50", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 50}}]),
    ("Set volume to 50", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 50}}]),
    ("Volume at 30%", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
    ("Turn the volume to 75", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 75}}]),
    ("Set volume to 100", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 100}}]),
    ("Max volume", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 100}}]),
    ("Full volume", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 100}}]),
    ("Turn it up to 80", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 80}}]),
    ("Turn it down to 20", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 20}}]),
    ("Volume 25 percent", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 25}}]),
    ("Make it louder", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 70}}]),
    ("Make it quieter", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
    ("Turn up the music", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 70}}]),
    ("Turn down the music", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
    ("Lower the volume", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 30}}]),
    ("Raise the volume", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 70}}]),
    ("Mute", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 0}}]),
    ("Mute the music", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 0}}]),
    ("Volume 0", [{"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 0}}]),
    
    # --- Status action ---
    ("What's playing?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("What song is this?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("What's this song?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("Current song", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("What am I listening to?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("Song name", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("What track is this?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("Who sings this?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("Who's the artist?", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    ("Music status", [{"name": "spotify_playback", "arguments": {"action": "status"}}]),
    
    # --- Devices action ---
    ("Show devices", [{"name": "spotify_playback", "arguments": {"action": "devices"}}]),
    ("List devices", [{"name": "spotify_playback", "arguments": {"action": "devices"}}]),
    ("What devices are available?", [{"name": "spotify_playback", "arguments": {"action": "devices"}}]),
    ("Spotify devices", [{"name": "spotify_playback", "arguments": {"action": "devices"}}]),
    ("Available devices", [{"name": "spotify_playback", "arguments": {"action": "devices"}}]),
    
    # --- Shuffle action ---
    ("Shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
    ("Turn on shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": True}}]),
    ("Enable shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": True}}]),
    ("Shuffle mode", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
    ("Shuffle the playlist", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
    ("Mix it up", [{"name": "spotify_playback", "arguments": {"action": "shuffle"}}]),
    ("Turn off shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": False}}]),
    ("Disable shuffle", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": False}}]),
    ("Stop shuffling", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": False}}]),
    ("Play in order", [{"name": "spotify_playback", "arguments": {"action": "shuffle", "shuffle_state": False}}]),
    
    # --- Repeat action ---
    ("Repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
    ("Turn on repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
    ("Enable repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
    ("Loop this song", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "track"}}]),
    ("Repeat this song", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "track"}}]),
    ("Play this song on repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "track"}}]),
    ("Loop the playlist", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "context"}}]),
    ("Repeat the album", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "context"}}]),
    ("Turn off repeat", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "off"}}]),
    ("Stop repeating", [{"name": "spotify_playback", "arguments": {"action": "repeat", "repeat_mode": "off"}}]),
    ("Repeat mode", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
    ("Loop", [{"name": "spotify_playback", "arguments": {"action": "repeat"}}]),
    
    # --- Search track action ---
    ("Search for Bohemian Rhapsody", [{"name": "spotify_playback", "arguments": {"action": "search_track", "query": "Bohemian Rhapsody"}}]),
    ("Find the song Hotel California", [{"name": "spotify_playback", "arguments": {"action": "search_track", "query": "Hotel California"}}]),
    ("Look up Shape of You", [{"name": "spotify_playback", "arguments": {"action": "search_track", "query": "Shape of You"}}]),
    
    # --- Search artist action ---
    ("Search for Taylor Swift", [{"name": "spotify_playback", "arguments": {"action": "search_artist", "query": "Taylor Swift"}}]),
    ("Find songs by Adele", [{"name": "spotify_playback", "arguments": {"action": "search_artist", "query": "Adele"}}]),
    ("Look up The Beatles", [{"name": "spotify_playback", "arguments": {"action": "search_artist", "query": "The Beatles"}}]),
    
    # =========================================================================
    # KASA LIGHTING TOOL - 60+ examples
    # Params: interaction (required), light_name, room, action, scene_name
    # interaction: "direct" or "scene"
    # =========================================================================
    
    # --- Direct control - on ---
    ("Turn on the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Lights on", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Turn on Light 1", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Switch on the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Light on", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Turn the lights on", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("I need some light", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("It's dark, turn on the light", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    ("Can you turn on the lights?", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}]),
    
    # --- Direct control - off ---
    ("Turn off the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Lights off", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Turn off Light 1", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Switch off the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Kill the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Turn the lights off", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Shut off the lights", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Can you turn off the lights?", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    ("Lights out", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}]),
    
    # --- Scene mode - movie ---
    ("Movie mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Set movie mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Movie lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Movie lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Set up movie lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Dim the lights for a movie", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("I'm watching a movie", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    ("Cinema mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}}]),
    
    # --- Scene mode - work ---
    ("Work mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("Work lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("Office lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("Bright lights for working", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("I need to work", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("Set up work mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    ("Study lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}}]),
    
    # --- Scene mode - reading ---
    ("Reading mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
    ("Reading lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
    ("Reading lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
    ("I want to read", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
    ("Set up reading mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}}]),
    
    # --- Scene mode - mood ---
    ("Mood lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ("Set the mood", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ("Romantic lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ("Cozy lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    ("Ambient lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}]),
    
    # --- Scene mode - party ---
    ("Party mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
    ("Party lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
    ("Party lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
    ("Set up party mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}}]),
    
    # --- Scene mode - relax ---
    ("Relax mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
    ("Relaxing lights", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
    ("Chill lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
    ("Relaxation mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
    ("I want to relax", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}}]),
    
    # --- Direct control - Light 2 ---
    ("Turn on Light 2", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "on"}}]),
    ("Switch on Light 2", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "on"}}]),
    ("Turn off Light 2", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "off"}}]),
    ("Switch off Light 2", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "off"}}]),
    ("Light 2 on", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "on"}}]),
    ("Light 2 off", [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 2", "action": "off"}}]),
    
    # --- Scene with bedroom ---
    ("Movie mode in the bedroom", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "bedroom"}}]),
    ("Bedroom relax mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}}]),
    ("Reading lights in the bedroom", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "bedroom"}}]),
    ("Bedroom movie mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "bedroom"}}]),
    ("Work lights in the bedroom", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "bedroom"}}]),
    ("Bedroom mood lighting", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "bedroom"}}]),
    ("Party lights in the bedroom", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "bedroom"}}]),
    ("Set bedroom to relax mode", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}}]),
    ("Relaxing lights in the bedroom", [{"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}}]),
    
    # =========================================================================
    # CALENDAR TOOL - 60+ examples  
    # Params: commands array with ONE command object
    # Commands have: read_or_write, calendar, read_type, limit, date, 
    #                event_title, start_time, end_time, location
    # =========================================================================
    
    # --- Read - next events ---
    ("What's on my calendar?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("Show my calendar", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("My calendar", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("Calendar", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("Upcoming events", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("What's coming up?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("What do I have scheduled?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("My schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("Show schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}]),
    ("My next 5 events", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events", "limit": 5}]}}]),
    ("Next 3 appointments", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events", "limit": 3}]}}]),
    
    # --- Read - day summary ---
    ("What's happening today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("Today's schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("What do I have today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("Any meetings today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("Today's events", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("What's on for today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("Am I free today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    ("Do I have anything today?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}]),
    
    # --- Read - week summary ---
    ("What's happening this week?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ("This week's schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ("Weekly schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ("Week overview", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ("What do I have this week?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    ("What's on this week?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}]),
    
    # --- Read - specific date ---
    ("What's on tomorrow?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
    ("Tomorrow's schedule", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
    ("What do I have tomorrow?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}]),
    ("Schedule for Friday", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "friday"}]}}]),
    ("What's happening on Saturday?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "saturday"}]}}]),
    ("Monday's calendar", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "monday"}]}}]),
    ("Events on Sunday", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "sunday"}]}}]),
    ("What's on Wednesday?", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "wednesday"}]}}]),
    
    # --- Create event ---
    ("Schedule a meeting at 2pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2pm"}]}}]),
    ("Add a meeting tomorrow at 3pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "tomorrow 3pm"}]}}]),
    ("Schedule lunch at noon", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch", "start_time": "12pm"}]}}]),
    ("Add dentist appointment on Monday at 10am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dentist appointment", "start_time": "monday 10am"}]}}]),
    ("Schedule dinner with Sarah on Friday at 7pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dinner with Sarah", "start_time": "friday 7pm"}]}}]),
    ("Add gym session tomorrow at 6am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Gym session", "start_time": "tomorrow 6am"}]}}]),
    ("Create event called Team Standup at 9am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Team Standup", "start_time": "9am"}]}}]),
    ("Put doctor appointment on Wednesday at 2:30pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Doctor appointment", "start_time": "wednesday 2:30pm"}]}}]),
    ("Schedule a call with John tomorrow at 4pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Call with John", "start_time": "tomorrow 4pm"}]}}]),
    ("Add haircut appointment Saturday at 11am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Haircut appointment", "start_time": "saturday 11am"}]}}]),
    ("Schedule meeting with boss at 3pm on Tuesday", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting with boss", "start_time": "tuesday 3pm"}]}}]),
    ("Add birthday party on Saturday at 5pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Birthday party", "start_time": "saturday 5pm"}]}}]),
    ("Schedule interview on Thursday at 1pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Interview", "start_time": "thursday 1pm"}]}}]),
    
    # --- Create event with location ---
    ("Schedule lunch at The Italian Place tomorrow at noon", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch", "start_time": "tomorrow 12pm", "location": "The Italian Place"}]}}]),
    ("Add meeting at Conference Room B at 2pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2pm", "location": "Conference Room B"}]}}]),
    ("Schedule dentist at 123 Main Street on Monday at 10am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dentist", "start_time": "monday 10am", "location": "123 Main Street"}]}}]),
    
    # --- Create event with end time ---
    ("Schedule a meeting from 2pm to 3pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2pm", "end_time": "3pm"}]}}]),
    ("Add a class from 9am to 10:30am tomorrow", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Class", "start_time": "tomorrow 9am", "end_time": "tomorrow 10:30am"}]}}]),
    ("Schedule a workout from 6am to 7am", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Workout", "start_time": "6am", "end_time": "7am"}]}}]),
    ("Block off 1pm to 2pm for lunch", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch", "start_time": "1pm", "end_time": "2pm"}]}}]),
    
    # --- Create event with specific dates (date embedded in start_time for consistency) ---
    ("Schedule a meeting on March 15th at 3pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2026-03-15 3pm"}]}}]),
    ("Add anniversary dinner this Saturday at 7pm", [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Anniversary dinner", "start_time": "saturday 7pm"}]}}]),
    
    # =========================================================================
    # STICKIES TOOL - 50+ examples
    # Params: action (required), section (for read), edits array (for write)
    # =========================================================================
    
    # --- Read operations ---
    ("What's on my sticky note?", [{"name": "stickies", "arguments": {"action": "read", "section": "both"}}]),
    ("Read my notes", [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]),
    ("Show my notes", [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]),
    ("My notes", [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]),
    ("Show my to-do list", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("What's on my to-do list?", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("My to-do list", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("Read my to-dos", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("What tasks do I have?", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("Show me my tasks", [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]),
    ("Check my stickies", [{"name": "stickies", "arguments": {"action": "read", "section": "both"}}]),
    ("Read my sticky note", [{"name": "stickies", "arguments": {"action": "read", "section": "both"}}]),
    ("What did I write down?", [{"name": "stickies", "arguments": {"action": "read", "section": "both"}}]),
    
    # --- Add to-do items ---
    ("Add buy milk to my to-do", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "buy milk"}]}}]),
    ("Add call mom to the list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "call mom"}]}}]),
    ("Put pick up dry cleaning on my to-do", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "pick up dry cleaning"}]}}]),
    ("Add grocery shopping to my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "grocery shopping"}]}}]),
    ("Remind me to take out the trash", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "take out the trash"}]}}]),
    ("Add send email to boss", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "send email to boss"}]}}]),
    ("Put pay bills on my to-do list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "pay bills"}]}}]),
    ("Add finish report to my tasks", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "finish report"}]}}]),
    ("Add walk the dog to my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "walk the dog"}]}}]),
    ("Add schedule dentist appointment", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "schedule dentist appointment"}]}}]),
    
    # --- Add multiple to-do items ---
    ("Add eggs, milk, and bread to my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "eggs"}, {"op": "add_todo", "item": "milk"}, {"op": "add_todo", "item": "bread"}]}}]),
    ("Add these to my to-do: call mom, pay rent, buy groceries", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "call mom"}, {"op": "add_todo", "item": "pay rent"}, {"op": "add_todo", "item": "buy groceries"}]}}]),
    ("Put apples and oranges on my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "apples"}, {"op": "add_todo", "item": "oranges"}]}}]),
    
    # --- Remove to-do items ---
    ("Remove buy milk from my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "buy milk"}]}}]),
    ("Cross off call mom", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "call mom"}]}}]),
    ("Delete grocery shopping from my to-do", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "grocery shopping"}]}}]),
    ("I finished the report, remove it", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "report"}]}}]),
    ("Done with taking out the trash", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "trash"}]}}]),
    ("Check off pay bills", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_todo", "match": "pay bills"}]}}]),
    
    # --- Edit to-do items ---
    ("Change call mom to call dad", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "edit_todo", "old": "call mom", "new": "call dad"}]}}]),
    ("Update buy milk to buy almond milk", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "edit_todo", "old": "buy milk", "new": "buy almond milk"}]}}]),
    
    # --- Add notes ---
    ("Add a note about the wifi password: abc123", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "wifi password", "content": "abc123"}]}}]),
    ("Note down meeting room: 302", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "meeting room", "content": "302"}]}}]),
    ("Save a note: parking spot A5", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "parking spot", "content": "A5"}]}}]),
    ("Add a note for the locker combination: 12-34-56", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "locker combination", "content": "12-34-56"}]}}]),
    
    # --- Add to-do with due dates ---
    ("Add finish report by Friday to my list", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "finish report", "due": "Friday"}]}}]),
    ("Add call landlord by Monday", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "call landlord", "due": "Monday"}]}}]),
    ("Add submit application due Feb 15", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "submit application", "due": "Feb 15"}]}}]),
    
    # --- Edit notes ---
    ("Update the wifi password note to xyz789", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "edit_note", "old": "wifi", "new": "xyz789"}]}}]),
    ("Change the meeting room note from 302 to 405", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "edit_note", "old": "302", "new": "405"}]}}]),
    
    # --- Remove notes ---
    ("Remove the note about parking", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_note", "match": "parking"}]}}]),
    ("Delete the wifi password note", [{"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "remove_note", "match": "wifi"}]}}]),
    
    # =========================================================================
    # SMS TOOL - 20+ examples
    # Params: message (required)
    # =========================================================================
    
    ("Text me a reminder to take out the trash", [{"name": "send_sms", "arguments": {"message": "Reminder: Take out the trash"}}]),
    ("Send me a text about the meeting", [{"name": "send_sms", "arguments": {"message": "Don't forget about the meeting"}}]),
    ("SMS reminder to pick up groceries", [{"name": "send_sms", "arguments": {"message": "Reminder: Pick up groceries"}}]),
    ("Text me: don't forget your keys", [{"name": "send_sms", "arguments": {"message": "Don't forget your keys"}}]),
    ("Send a text reminder about the dentist", [{"name": "send_sms", "arguments": {"message": "Reminder: Dentist appointment"}}]),
    ("Message me to call mom", [{"name": "send_sms", "arguments": {"message": "Remember to call mom"}}]),
    ("Text me the wifi password", [{"name": "send_sms", "arguments": {"message": "Wifi password reminder"}}]),
    ("Send me a reminder to take my medication", [{"name": "send_sms", "arguments": {"message": "Reminder: Take your medication"}}]),
    ("Text me about the presentation", [{"name": "send_sms", "arguments": {"message": "Don't forget about the presentation"}}]),
    ("SMS me a reminder to water the plants", [{"name": "send_sms", "arguments": {"message": "Reminder: Water the plants"}}]),
    ("Send me a text to check my email", [{"name": "send_sms", "arguments": {"message": "Reminder: Check your email"}}]),
    ("Text me to charge my laptop", [{"name": "send_sms", "arguments": {"message": "Reminder: Charge your laptop"}}]),
    
    # =========================================================================
    # GOOGLE SEARCH TOOL - 30+ examples
    # Params: query (required), query_type (general/link/directions)
    # =========================================================================
    
    # --- General search ---
    ("Search for best pizza places nearby", [{"name": "google_search", "arguments": {"query": "best pizza places nearby", "query_type": "general"}}]),
    ("Look up how to make pasta", [{"name": "google_search", "arguments": {"query": "how to make pasta", "query_type": "general"}}]),
    ("Google the capital of France", [{"name": "google_search", "arguments": {"query": "capital of France", "query_type": "general"}}]),
    ("Search for Python tutorials", [{"name": "google_search", "arguments": {"query": "Python tutorials", "query_type": "general"}}]),
    ("Find information about Mars", [{"name": "google_search", "arguments": {"query": "Mars planet information", "query_type": "general"}}]),
    ("Look up symptoms of the flu", [{"name": "google_search", "arguments": {"query": "flu symptoms", "query_type": "general"}}]),
    ("Search for healthy recipes", [{"name": "google_search", "arguments": {"query": "healthy recipes", "query_type": "general"}}]),
    ("Google the weather in Paris", [{"name": "google_search", "arguments": {"query": "weather in Paris", "query_type": "general"}}]),
    ("Find out who won the Super Bowl", [{"name": "google_search", "arguments": {"query": "who won the Super Bowl", "query_type": "general"}}]),
    ("Search how to change a tire", [{"name": "google_search", "arguments": {"query": "how to change a tire", "query_type": "general"}}]),
    ("Look up the best restaurants in town", [{"name": "google_search", "arguments": {"query": "best restaurants in town", "query_type": "general"}}]),
    ("Search for movie times", [{"name": "google_search", "arguments": {"query": "movie times near me", "query_type": "general"}}]),
    ("Google who invented the telephone", [{"name": "google_search", "arguments": {"query": "who invented the telephone", "query_type": "general"}}]),
    
    # --- Link search ---
    ("Get me the link to Amazon", [{"name": "google_search", "arguments": {"query": "Amazon", "query_type": "link"}}]),
    ("Find the website for Netflix", [{"name": "google_search", "arguments": {"query": "Netflix", "query_type": "link"}}]),
    ("Link to the New York Times", [{"name": "google_search", "arguments": {"query": "New York Times", "query_type": "link"}}]),
    ("Get the URL for Wikipedia", [{"name": "google_search", "arguments": {"query": "Wikipedia", "query_type": "link"}}]),
    ("Find the link to GitHub", [{"name": "google_search", "arguments": {"query": "GitHub", "query_type": "link"}}]),
    ("Website for Stack Overflow", [{"name": "google_search", "arguments": {"query": "Stack Overflow", "query_type": "link"}}]),
    
    # --- Directions search ---
    ("Directions to the nearest Starbucks", [{"name": "google_search", "arguments": {"query": "nearest Starbucks", "query_type": "directions"}}]),
    ("How do I get to the airport?", [{"name": "google_search", "arguments": {"query": "directions to airport", "query_type": "directions"}}]),
    ("Navigate to Central Park", [{"name": "google_search", "arguments": {"query": "Central Park", "query_type": "directions"}}]),
    ("Directions to the grocery store", [{"name": "google_search", "arguments": {"query": "grocery store", "query_type": "directions"}}]),
    ("How to get to the mall", [{"name": "google_search", "arguments": {"query": "directions to mall", "query_type": "directions"}}]),
    ("Directions to the hospital", [{"name": "google_search", "arguments": {"query": "directions to hospital", "query_type": "directions"}}]),
    ("Take me to the nearest gas station", [{"name": "google_search", "arguments": {"query": "nearest gas station", "query_type": "directions"}}]),
    
    # =========================================================================
    # CLIPBOARD TOOL - 15+ examples
    # Params: max_length (optional)
    # =========================================================================
    
    ("What's on my clipboard?", [{"name": "read_clipboard", "arguments": {}}]),
    ("Read my clipboard", [{"name": "read_clipboard", "arguments": {}}]),
    ("Paste what I copied", [{"name": "read_clipboard", "arguments": {}}]),
    ("Show clipboard contents", [{"name": "read_clipboard", "arguments": {}}]),
    ("What did I copy?", [{"name": "read_clipboard", "arguments": {}}]),
    ("Clipboard", [{"name": "read_clipboard", "arguments": {}}]),
    ("Check my clipboard", [{"name": "read_clipboard", "arguments": {}}]),
    ("What's in my clipboard?", [{"name": "read_clipboard", "arguments": {}}]),
    ("Show me what I copied", [{"name": "read_clipboard", "arguments": {}}]),
    ("Read what's on my clipboard", [{"name": "read_clipboard", "arguments": {}}]),
    ("What text is on my clipboard?", [{"name": "read_clipboard", "arguments": {}}]),
    
    # =========================================================================
    # BRIEFING TOOL - 30+ examples
    # Params: action (required), message, remind_at, remind_before_minutes, 
    #         event_time, briefing_id
    # =========================================================================
    
    # --- Create briefings ---
    ("Remind me at 9am to take my medication", [{"name": "briefing", "arguments": {"action": "create", "message": "Take your medication", "remind_at": "9am"}}]),
    ("Set a reminder for 3pm about the meeting", [{"name": "briefing", "arguments": {"action": "create", "message": "You have a meeting", "remind_at": "3pm"}}]),
    ("Remind me tomorrow at 8am to call mom", [{"name": "briefing", "arguments": {"action": "create", "message": "Call mom", "remind_at": "tomorrow 8am"}}]),
    ("Create a reminder for 5pm to leave for dinner", [{"name": "briefing", "arguments": {"action": "create", "message": "Time to leave for dinner", "remind_at": "5pm"}}]),
    ("Remind me at noon to eat lunch", [{"name": "briefing", "arguments": {"action": "create", "message": "Time for lunch", "remind_at": "12pm"}}]),
    ("Set a briefing for 7am about the gym", [{"name": "briefing", "arguments": {"action": "create", "message": "Time for the gym", "remind_at": "7am"}}]),
    ("Remind me at 10pm to go to bed", [{"name": "briefing", "arguments": {"action": "create", "message": "Time for bed", "remind_at": "10pm"}}]),
    ("Create a reminder at 2pm to check email", [{"name": "briefing", "arguments": {"action": "create", "message": "Check your email", "remind_at": "2pm"}}]),
    
    # --- Create briefings with relative timing ---
    ("Remind me 30 minutes before my 3pm meeting", [{"name": "briefing", "arguments": {"action": "create", "message": "Meeting in 30 minutes", "event_time": "3pm", "remind_before_minutes": 30}}]),
    ("Remind me 15 minutes before the dentist at 2pm", [{"name": "briefing", "arguments": {"action": "create", "message": "Dentist appointment in 15 minutes", "event_time": "2pm", "remind_before_minutes": 15}}]),
    ("Give me a 1 hour heads up before my 5pm dinner", [{"name": "briefing", "arguments": {"action": "create", "message": "Dinner in 1 hour", "event_time": "5pm", "remind_before_minutes": 60}}]),
    
    # --- List briefings ---
    ("What reminders do I have?", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("Show my briefings", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("List my reminders", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("What briefings are pending?", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("Any pending reminders?", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("Show all reminders", [{"name": "briefing", "arguments": {"action": "list"}}]),
    ("My briefings", [{"name": "briefing", "arguments": {"action": "list"}}]),
    
    # --- Dismiss briefings ---
    # Note: briefing_id is omitted since real IDs are runtime values.
    # The orchestrator resolves which briefing to dismiss from context.
    ("Cancel the medication reminder", [{"name": "briefing", "arguments": {"action": "dismiss"}}]),
    ("Delete that reminder", [{"name": "briefing", "arguments": {"action": "dismiss"}}]),
    ("Dismiss all briefings", [{"name": "briefing", "arguments": {"action": "dismiss"}}]),
    ("Cancel my reminders", [{"name": "briefing", "arguments": {"action": "dismiss"}}]),
    ("Remove the meeting reminder", [{"name": "briefing", "arguments": {"action": "dismiss"}}]),
    
    # =========================================================================
    # NOTIFICATIONS TOOL - 20+ examples
    # Params: type_filter (email/news/other/all), limit
    # =========================================================================
    
    ("Check my notifications", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
    ("Any notifications?", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
    ("Show notifications", [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]),
    ("Any new emails?", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("Check my email", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("Email updates", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("Show me my emails", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("Any emails?", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("Email notifications", [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]),
    ("What's in the news?", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("News updates", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("Show me the news", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("Any news?", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("Latest news", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("Tech news", [{"name": "get_notifications", "arguments": {"type_filter": "news"}}]),
    ("Show last 5 notifications", [{"name": "get_notifications", "arguments": {"type_filter": "all", "limit": 5}}]),
    ("Show 3 most recent emails", [{"name": "get_notifications", "arguments": {"type_filter": "email", "limit": 3}}]),
    
    # =========================================================================
    # SYSTEM INFO TOOL - 15+ examples
    # Params: section (overview/architecture/providers/orchestrator/audio/
    #                  tools/memory/config/all)
    # =========================================================================
    
    ("How do you work?", [{"name": "system_info", "arguments": {"section": "overview"}}]),
    ("Tell me about yourself", [{"name": "system_info", "arguments": {"section": "overview"}}]),
    ("What are you?", [{"name": "system_info", "arguments": {"section": "overview"}}]),
    ("Explain your architecture", [{"name": "system_info", "arguments": {"section": "architecture"}}]),
    ("How are you built?", [{"name": "system_info", "arguments": {"section": "architecture"}}]),
    ("What providers do you use?", [{"name": "system_info", "arguments": {"section": "providers"}}]),
    ("How does your audio work?", [{"name": "system_info", "arguments": {"section": "audio"}}]),
    ("Audio pipeline explanation", [{"name": "system_info", "arguments": {"section": "audio"}}]),
    ("What tools do you have?", [{"name": "system_info", "arguments": {"section": "tools"}}]),
    ("List your tools", [{"name": "system_info", "arguments": {"section": "tools"}}]),
    ("How does your memory work?", [{"name": "system_info", "arguments": {"section": "memory"}}]),
    ("Memory system", [{"name": "system_info", "arguments": {"section": "memory"}}]),
    ("How are you configured?", [{"name": "system_info", "arguments": {"section": "config"}}]),
    ("Configuration details", [{"name": "system_info", "arguments": {"section": "config"}}]),
    ("Tell me everything about yourself", [{"name": "system_info", "arguments": {"section": "all"}}]),
    
    # =========================================================================
    # CURSOR COMPOSER TOOL - 30+ examples
    # Params: prompt (required)
    # =========================================================================
    
    ("Add a dark mode toggle to the settings", [{"name": "cursor_composer", "arguments": {"prompt": "Add a dark mode toggle to the settings"}}]),
    ("Refactor the authentication module", [{"name": "cursor_composer", "arguments": {"prompt": "Refactor the authentication module"}}]),
    ("Create a new API endpoint for users", [{"name": "cursor_composer", "arguments": {"prompt": "Create a new API endpoint for users"}}]),
    ("Fix the bug in the login form", [{"name": "cursor_composer", "arguments": {"prompt": "Fix the bug in the login form"}}]),
    ("Add error handling to the database queries", [{"name": "cursor_composer", "arguments": {"prompt": "Add error handling to the database queries"}}]),
    ("Write unit tests for the payment service", [{"name": "cursor_composer", "arguments": {"prompt": "Write unit tests for the payment service"}}]),
    ("Optimize the image loading", [{"name": "cursor_composer", "arguments": {"prompt": "Optimize the image loading"}}]),
    ("Add TypeScript types to the utils", [{"name": "cursor_composer", "arguments": {"prompt": "Add TypeScript types to the utils"}}]),
    ("Implement pagination for the list", [{"name": "cursor_composer", "arguments": {"prompt": "Implement pagination for the list"}}]),
    ("Create a reusable modal component", [{"name": "cursor_composer", "arguments": {"prompt": "Create a reusable modal component"}}]),
    ("Add logging to the middleware", [{"name": "cursor_composer", "arguments": {"prompt": "Add logging to the middleware"}}]),
    ("Convert CSS to Tailwind", [{"name": "cursor_composer", "arguments": {"prompt": "Convert CSS to Tailwind"}}]),
    ("Add input validation to the form", [{"name": "cursor_composer", "arguments": {"prompt": "Add input validation to the form"}}]),
    ("Create a database migration", [{"name": "cursor_composer", "arguments": {"prompt": "Create a database migration"}}]),
    ("Implement caching for API responses", [{"name": "cursor_composer", "arguments": {"prompt": "Implement caching for API responses"}}]),
    ("Add a search feature to the app", [{"name": "cursor_composer", "arguments": {"prompt": "Add a search feature to the app"}}]),
    ("Refactor the code to use hooks", [{"name": "cursor_composer", "arguments": {"prompt": "Refactor the code to use hooks"}}]),
    ("Add authentication middleware", [{"name": "cursor_composer", "arguments": {"prompt": "Add authentication middleware"}}]),
    ("Create a new component for the dashboard", [{"name": "cursor_composer", "arguments": {"prompt": "Create a new component for the dashboard"}}]),
    ("Fix the styling issues on mobile", [{"name": "cursor_composer", "arguments": {"prompt": "Fix the styling issues on mobile"}}]),
    ("Add a loading spinner", [{"name": "cursor_composer", "arguments": {"prompt": "Add a loading spinner"}}]),
    ("Implement drag and drop", [{"name": "cursor_composer", "arguments": {"prompt": "Implement drag and drop"}}]),
    ("Add websocket support", [{"name": "cursor_composer", "arguments": {"prompt": "Add websocket support"}}]),
    ("Create an email template", [{"name": "cursor_composer", "arguments": {"prompt": "Create an email template"}}]),
    ("Add rate limiting to the API", [{"name": "cursor_composer", "arguments": {"prompt": "Add rate limiting to the API"}}]),
    ("Implement file upload feature", [{"name": "cursor_composer", "arguments": {"prompt": "Implement file upload feature"}}]),
    ("Add notifications system", [{"name": "cursor_composer", "arguments": {"prompt": "Add notifications system"}}]),
    ("Create a chart component", [{"name": "cursor_composer", "arguments": {"prompt": "Create a chart component"}}]),
    ("Add OAuth login", [{"name": "cursor_composer", "arguments": {"prompt": "Add OAuth login"}}]),
    ("Implement lazy loading", [{"name": "cursor_composer", "arguments": {"prompt": "Implement lazy loading"}}]),
    
    # =========================================================================
    # MULTI-TOOL CALLS - 80+ examples
    # Multiple tools called in parallel
    # =========================================================================
    
    # ---- MUSIC + LIGHTS (atmosphere combos) ----
    ("Play jazz and dim the lights", [
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}},
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}}
    ]),
    ("Movie night - movie mode and pause the music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}}
    ]),
    ("Work mode - bright lights and focus music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "focus music", "search_type": "artist"}}
    ]),
    ("Reading time - reading lights and soft classical", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "reading", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "soft classical", "search_type": "artist"}}
    ]),
    ("Party setup - party lights and party music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "party music", "search_type": "artist"}}
    ]),
    ("Relax mode - relaxing lights and chill music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "chill music", "search_type": "artist"}}
    ]),
    ("Turn off the lights and stop the music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}}
    ]),
    ("Turn on the lights and play some music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}},
        {"name": "spotify_playback", "arguments": {"action": "play"}}
    ]),
    ("Mood lighting and play R&B", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "R&B", "search_type": "artist"}}
    ]),
    ("Bedtime - turn off the lights and pause the music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}}
    ]),
    ("Study time - work lights and lo-fi music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi", "search_type": "artist"}}
    ]),
    ("Date night - mood lighting and play smooth jazz", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "smooth jazz", "search_type": "artist"}}
    ]),
    ("Yoga time - relax lights and meditation music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "meditation music", "search_type": "artist"}}
    ]),
    ("Dinner party - mood lights and play soul music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "mood", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "soul music", "search_type": "artist"}}
    ]),
    ("Play some rock and turn on the lights", [
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "rock", "search_type": "artist"}},
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}
    ]),
    ("Turn on the bedroom relax mode and play sleep music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "bedroom"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "sleep music", "search_type": "artist"}}
    ]),
    ("Movie mode in the bedroom and play Hans Zimmer", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "bedroom"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "Hans Zimmer", "search_type": "artist"}}
    ]),
    ("Shut everything down - lights off and music off", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}}
    ]),
    
    # ---- WEATHER + CALENDAR (planning) ----
    ("What's the weather and my schedule for tomorrow?", [
        {"name": "weather", "arguments": {"specific_date": "tomorrow"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}
    ]),
    ("Weather and calendar for today", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("What's the weather like and do I have any meetings?", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Friday weather and Friday schedule", [
        {"name": "weather", "arguments": {"specific_date": "friday"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "friday"}]}}
    ]),
    ("What's the weather this weekend and any Saturday plans?", [
        {"name": "weather", "arguments": {"specific_date": "saturday"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "saturday"}]}}
    ]),
    ("Weekly weather and weekly schedule", [
        {"name": "weather", "arguments": {"days": 7}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "week_summary"}]}}
    ]),
    ("Tomorrow's weather and events", [
        {"name": "weather", "arguments": {"specific_date": "tomorrow"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}
    ]),
    
    # ---- WEATHER + MUSIC ----
    ("Check the weather and play some relaxing music", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "relaxing music", "search_type": "artist"}}
    ]),
    ("What's the weather and play some morning jazz", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "morning jazz", "search_type": "artist"}}
    ]),
    ("Weather and play some country music", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "country music", "search_type": "artist"}}
    ]),
    
    # ---- STICKIES + BRIEFING (task + reminder) ----
    ("Add eggs to my list and remind me at 5pm to go shopping", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "eggs"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Time to go shopping", "remind_at": "5pm"}}
    ]),
    ("Add milk to my to-do and set a reminder for the grocery store", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "milk"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Don't forget to buy milk", "remind_at": "4pm"}}
    ]),
    ("Add take medicine to my list and remind me at 9am", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "take medicine"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Take your medicine", "remind_at": "9am"}}
    ]),
    ("Add submit report and set a reminder for noon", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "submit report"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Submit your report", "remind_at": "12pm"}}
    ]),
    ("Put laundry on my list and remind me at 6pm", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "laundry"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Do the laundry", "remind_at": "6pm"}}
    ]),
    
    # ---- STICKIES + SMS (task + text reminder) ----
    ("Add buy groceries to my list and text me about it", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "buy groceries"}]}},
        {"name": "send_sms", "arguments": {"message": "Reminder: Buy groceries"}}
    ]),
    ("Add call dentist to my to-do and send me a text reminder", [
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "call dentist"}]}},
        {"name": "send_sms", "arguments": {"message": "Reminder: Call the dentist"}}
    ]),
    
    # ---- CALENDAR + STICKIES (event + to-do) ----
    ("Schedule a meeting at 3pm and add it to my to-do list", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "3pm"}]}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "meeting at 3pm"}]}}
    ]),
    ("Add dentist to my calendar tomorrow at 2pm and my to-do list", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dentist", "start_time": "tomorrow 2pm"}]}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "dentist appointment tomorrow"}]}}
    ]),
    ("Schedule gym at 7am and add it to my list", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Gym", "start_time": "7am"}]}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "gym at 7am"}]}}
    ]),
    ("Add lunch with John at noon to both my calendar and to-do", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Lunch with John", "start_time": "12pm"}]}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "lunch with John at noon"}]}}
    ]),
    
    # ---- CALENDAR + BRIEFING (event + reminder) ----
    ("Schedule a meeting at 3pm and remind me 30 minutes before", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "3pm"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Meeting in 30 minutes", "event_time": "3pm", "remind_before_minutes": 30}}
    ]),
    ("Add dentist at 2pm and set a reminder for 1:30pm", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dentist", "start_time": "2pm"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Dentist appointment soon", "remind_at": "1:30pm"}}
    ]),
    ("Schedule dinner at 7pm and remind me an hour before", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dinner", "start_time": "7pm"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Dinner in 1 hour", "event_time": "7pm", "remind_before_minutes": 60}}
    ]),
    
    # ---- CALENDAR + SMS (event + text) ----
    ("Schedule meeting at 2pm and text me about it", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Meeting", "start_time": "2pm"}]}},
        {"name": "send_sms", "arguments": {"message": "Meeting scheduled at 2pm"}}
    ]),
    
    # ---- NOTIFICATIONS + WEATHER ----
    ("Any emails and what's the weather?", [
        {"name": "get_notifications", "arguments": {"type_filter": "email"}},
        {"name": "weather", "arguments": {"days": 1}}
    ]),
    ("Weather and any news?", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "get_notifications", "arguments": {"type_filter": "news"}}
    ]),
    
    # ---- NOTIFICATIONS + CALENDAR ----
    ("Check notifications and calendar", [
        {"name": "get_notifications", "arguments": {"type_filter": "all"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Any emails and what's on my schedule?", [
        {"name": "get_notifications", "arguments": {"type_filter": "email"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Check news and my calendar for today", [
        {"name": "get_notifications", "arguments": {"type_filter": "news"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    
    # ---- CLIPBOARD (sequential intent - just read clipboard first) ----
    # Note: Clipboard content is unknown at call time, so these are single-tool
    # calls. The orchestrator handles follow-up actions after reading.
    ("Read my clipboard and save it as a note", [{"name": "read_clipboard", "arguments": {}}]),
    ("What's on my clipboard and add it to my to-do", [{"name": "read_clipboard", "arguments": {}}]),
    ("Read my clipboard and text it to me", [{"name": "read_clipboard", "arguments": {}}]),
    
    # ---- SEARCH + STICKIES (find info + save) ----
    ("Google the best sushi places and add 'find sushi restaurant' to my to-do", [
        {"name": "google_search", "arguments": {"query": "best sushi places nearby", "query_type": "general"}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "find sushi restaurant"}]}}
    ]),
    ("Search for flight prices to LA and add it to my notes", [
        {"name": "google_search", "arguments": {"query": "flight prices to Los Angeles", "query_type": "general"}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_note", "subheading": "LA flights", "content": "check flight prices"}]}}
    ]),
    
    # ---- WEATHER + LIGHTS ----
    ("What's the weather and turn on the lights", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}}
    ]),
    
    # ---- STICKIES + STICKIES (read + write in one) ----
    ("Show my to-do list and add buy milk", [
        {"name": "stickies", "arguments": {"action": "read", "section": "todo"}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "buy milk"}]}}
    ]),
    
    # ---- CALENDAR + STICKIES + BRIEFING (triple combo) ----
    ("Schedule dentist at 2pm, add it to my list, and remind me 1 hour before", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "create_event", "event_title": "Dentist", "start_time": "2pm"}]}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "dentist at 2pm"}]}},
        {"name": "briefing", "arguments": {"action": "create", "message": "Dentist in 1 hour", "event_time": "2pm", "remind_before_minutes": 60}}
    ]),
    
    # ---- WEATHER + CALENDAR + MUSIC (start the day) ----
    ("Morning update - weather, calendar, and play morning music", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "morning music", "search_type": "artist"}}
    ]),
    
    # ---- MORNING ROUTINES (3-tool combos) ----
    ("Morning briefing - weather, calendar, and emails", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "get_notifications", "arguments": {"type_filter": "email"}}
    ]),
    ("Good morning - what's the weather, my schedule, and any news?", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "get_notifications", "arguments": {"type_filter": "news"}}
    ]),
    ("Start of day - weather, emails, and calendar", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "get_notifications", "arguments": {"type_filter": "email"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Morning check - any notifications, weather, and schedule", [
        {"name": "get_notifications", "arguments": {"type_filter": "all"}},
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Wake up routine - weather, calendar, news, and morning jazz", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "get_notifications", "arguments": {"type_filter": "news"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "morning jazz", "search_type": "artist"}}
    ]),
    
    # ---- EVENING ROUTINES ----
    ("Winding down - relax lights, soft music, and check tomorrow's calendar", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "relax", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "soft ambient", "search_type": "artist"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}
    ]),
    ("Goodnight - turn off lights and stop music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}}
    ]),
    ("Bedtime routine - tomorrow's weather and schedule", [
        {"name": "weather", "arguments": {"specific_date": "tomorrow"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}
    ]),
    
    # ---- WORK SESSIONS ----
    ("Work session - work lights, focus music, and check calendar", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "focus music", "search_type": "artist"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    ("Study mode - work lights and lo-fi beats", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "lo-fi beats", "search_type": "artist"}}
    ]),
    ("Deep work - work lights, volume at 20, and what's on my calendar", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "work", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "volume", "volume_level": 20}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
    
    # ---- STATUS CHECKS (multiple reads) ----
    ("What's on my calendar and my to-do list?", [
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "stickies", "arguments": {"action": "read", "section": "todo"}}
    ]),
    ("Check my notes and my schedule", [
        {"name": "stickies", "arguments": {"action": "read", "section": "both"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}
    ]),
    ("What do I have on my to-do list and any reminders?", [
        {"name": "stickies", "arguments": {"action": "read", "section": "todo"}},
        {"name": "briefing", "arguments": {"action": "list"}}
    ]),
    ("Show my reminders and my calendar", [
        {"name": "briefing", "arguments": {"action": "list"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "next_events"}]}}
    ]),
    ("Weather and my to-do list", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "stickies", "arguments": {"action": "read", "section": "todo"}}
    ]),
    ("Any emails and what's on my to-do?", [
        {"name": "get_notifications", "arguments": {"type_filter": "email"}},
        {"name": "stickies", "arguments": {"action": "read", "section": "todo"}}
    ]),
    ("Check everything - weather, calendar, emails, and my notes", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}},
        {"name": "get_notifications", "arguments": {"type_filter": "email"}},
        {"name": "stickies", "arguments": {"action": "read", "section": "both"}}
    ]),
    
    # ---- NATURAL LANGUAGE COMBOS ----
    ("I'm heading out - weather and turn off the lights", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "off"}}
    ]),
    ("I'm home - turn on the lights and play some music", [
        {"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "Light 1", "action": "on"}},
        {"name": "spotify_playback", "arguments": {"action": "play"}}
    ]),
    ("Getting ready for a party - party lights, party music, and check if anyone texted", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "party", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "play", "query": "party music", "search_type": "artist"}},
        {"name": "get_notifications", "arguments": {"type_filter": "all"}}
    ]),
    ("Quick update - weather and emails", [
        {"name": "weather", "arguments": {"days": 1}},
        {"name": "get_notifications", "arguments": {"type_filter": "email"}}
    ]),
    ("Planning tomorrow - weather and schedule for tomorrow", [
        {"name": "weather", "arguments": {"specific_date": "tomorrow"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "specific_date", "date": "tomorrow"}]}}
    ]),
    ("Cooking time - search for pasta recipe and add groceries to my list", [
        {"name": "google_search", "arguments": {"query": "easy pasta recipe", "query_type": "general"}},
        {"name": "stickies", "arguments": {"action": "write", "edits": [{"op": "add_todo", "item": "buy groceries for pasta"}]}}
    ]),
    ("Set up for the movie - movie lights, pause the music, and what's on my calendar tonight", [
        {"name": "kasa_lighting", "arguments": {"interaction": "scene", "scene_name": "movie", "room": "living room"}},
        {"name": "spotify_playback", "arguments": {"action": "pause"}},
        {"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "calendar": "all", "read_type": "day_summary"}]}}
    ]),
]


def generate_training_example(user_prompt: str, expected_output: list) -> dict:
    """Generate a single training example in JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(expected_output)}
        ]
    }


def validate_value_against_schema(value, prop_schema: dict, path: str) -> list:
    """Validate a single value against its property schema. Returns list of errors."""
    errors = []
    expected_type = prop_schema.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
        elif "enum" in prop_schema and value not in prop_schema["enum"]:
            errors.append(f"{path}: '{value}' not in enum {prop_schema['enum']}")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
        else:
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(f"{path}: {value} < minimum {prop_schema['minimum']}")
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(f"{path}: {value} > maximum {prop_schema['maximum']}")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
    elif expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
        elif "items" in prop_schema:
            for i, item in enumerate(value):
                errors.extend(validate_object_against_schema(
                    item, prop_schema["items"], f"{path}[{i}]"
                ))

    return errors


def validate_object_against_schema(obj: dict, schema: dict, path: str = "") -> list:
    """Validate an object against a JSON schema. Returns list of errors."""
    errors = []

    if schema.get("type") == "object" and not isinstance(obj, dict):
        return [f"{path}: expected object, got {type(obj).__name__}"]

    if not isinstance(obj, dict):
        return errors

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for req in required:
        if req not in obj:
            errors.append(f"{path}: missing required param '{req}'")

    for key, value in obj.items():
        if key not in properties:
            errors.append(f"{path}: unknown param '{key}'")
            continue
        prop_schema = properties[key]
        errors.extend(validate_value_against_schema(value, prop_schema, f"{path}.{key}"))

    return errors


def validate_tool_call(tool_call: dict, user_prompt: str = "") -> list:
    """Validate a tool call against known schemas. Returns list of error strings."""
    errors = []
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})

    # Find schema
    schema = None
    for tool in TOOL_SCHEMAS["tools"]:
        if tool["name"] == name:
            schema = tool
            break

    if not schema:
        return [f"Unknown tool '{name}'"]

    params_schema = schema.get("parameters", {})

    # Check required params
    required = params_schema.get("required", [])
    for req in required:
        if req not in args:
            errors.append(f"[{name}] missing required param '{req}'")

    # Check each argument against schema
    properties = params_schema.get("properties", {})
    for key, value in args.items():
        if key not in properties:
            errors.append(f"[{name}] unknown param '{key}'")
            continue
        prop_schema = properties[key]
        errors.extend(validate_value_against_schema(value, prop_schema, f"[{name}].{key}"))

    # Check oneOf constraints (e.g., weather must have exactly one of hours/days/specific_date)
    one_of = params_schema.get("oneOf", [])
    if one_of:
        matches = 0
        for constraint in one_of:
            constraint_required = constraint.get("required", [])
            if all(r in args for r in constraint_required):
                matches += 1
        if matches == 0:
            groups = [c.get("required", []) for c in one_of]
            errors.append(f"[{name}] must satisfy one of: {groups}")
        elif matches > 1:
            errors.append(f"[{name}] satisfies multiple oneOf constraints (should be exactly one)")

    return errors


def main():
    output_path = SCRIPT_DIR / "training_data.jsonl"

    # Validate all examples
    print("Validating examples...")
    total_errors = 0
    for user_prompt, expected_output in TRAINING_EXAMPLES:
        for tool_call in expected_output:
            call_errors = validate_tool_call(tool_call, user_prompt)
            for err in call_errors:
                print(f"  ERROR: {err}")
                print(f"    Prompt: {user_prompt[:60]}...")
                total_errors += 1

    if total_errors > 0:
        print(f"\n{total_errors} validation errors found!")
    else:
        print("All examples validated successfully!")

    # Write output
    with open(output_path, "w") as f:
        for user_prompt, expected_output in TRAINING_EXAMPLES:
            example = generate_training_example(user_prompt, expected_output)
            f.write(json.dumps(example) + "\n")

    print(f"\nGenerated {len(TRAINING_EXAMPLES)} training examples")
    print(f"Output: {output_path}")

    # Stats
    no_tool = sum(1 for _, output in TRAINING_EXAMPLES if len(output) == 0)
    single_tool = sum(1 for _, output in TRAINING_EXAMPLES if len(output) == 1)
    multi_tool = sum(1 for _, output in TRAINING_EXAMPLES if len(output) > 1)

    # Tool distribution
    tool_counts = {}
    for _, output in TRAINING_EXAMPLES:
        for call in output:
            name = call["name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

    print("\nBreakdown:")
    print(f"  No-tool (negative): {no_tool}")
    print(f"  Single-tool calls: {single_tool}")
    print(f"  Multi-tool calls: {multi_tool}")
    print("\nTool distribution:")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
