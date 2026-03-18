#!/usr/bin/env python3
"""Add training entries to disambiguate google_search from other tools."""

import json

with open("_sys_msg.txt") as f:
    SYSTEM_MSG = f.read()


def entry(user_msg, think, tools):
    """Build a single JSONL entry."""
    tool_json = json.dumps(tools, ensure_ascii=False)
    assistant = f"<think>\n{think}\n</think>\n{tool_json}"
    return json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant}
        ]
    }, ensure_ascii=False)


new_entries = []

# =============================================================================
# SECTION 1: Explicit "search" keyword → google_search
# These teach that "search" / "search for" / "google" are strong google_search signals
# =============================================================================

new_entries.append(entry(
    "Search for the best hiking trails in Colorado",
    "The user explicitly said 'search for', which indicates a web search request. "
    "They want general information about hiking trails in Colorado - this is not a request "
    "that maps to any personal tool like weather, calendar, or stickies. "
    "I will use google_search with a general query.",
    [{"name": "google_search", "arguments": {"query": "best hiking trails in Colorado", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search how to make sourdough bread",
    "The user said 'search how to' which is a clear web search intent. "
    "They want instructional information from the web, not a personal tool action. "
    "The word 'search' signals google_search.",
    [{"name": "google_search", "arguments": {"query": "how to make sourdough bread", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for iPhone 16 reviews",
    "The user explicitly uses the word 'search' to find product reviews online. "
    "This is clearly a web information request. No personal tool (weather, calendar, spotify, etc.) is relevant here. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "iPhone 16 reviews", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for apartments for rent near me",
    "The user said 'search for' which signals a google_search request. "
    "They want real estate listings from the web. This is external information, not something handled by weather, "
    "calendar, stickies, or any other personal tool.",
    [{"name": "google_search", "arguments": {"query": "apartments for rent near me", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search when does the new Star Wars movie come out",
    "The user said 'search when' - the word 'search' indicates they want to look this up on the web. "
    "Movie release dates are external information, not handled by any personal tool. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "new Star Wars movie release date", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for the best noise cancelling headphones",
    "The user explicitly said 'search for', indicating a web search. "
    "They want product recommendations from the web. This does not map to spotify, weather, or any personal tool. "
    "The word 'search' is a strong signal for google_search.",
    [{"name": "google_search", "arguments": {"query": "best noise cancelling headphones", "query_type": "general"}}]
))

new_entries.append(entry(
    "Google what year the Titanic sank",
    "The user said 'Google', which is an explicit request to search the web. "
    "This is a factual/historical question requiring web information. "
    "No personal tool is relevant here.",
    [{"name": "google_search", "arguments": {"query": "what year did the Titanic sink", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for how to train for a marathon",
    "The user used 'search for' which directly signals a web search. "
    "They want training advice from the internet. This is external info, not a personal tool action.",
    [{"name": "google_search", "arguments": {"query": "how to train for a marathon", "query_type": "general"}}]
))

new_entries.append(entry(
    "Can you search for vegan dinner ideas?",
    "The user asked me to 'search for' something - the word 'search' is the key signal here. "
    "They want recipe ideas from the web. Even though this is a question format, "
    "the explicit 'search' keyword means I should use google_search.",
    [{"name": "google_search", "arguments": {"query": "vegan dinner ideas", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for the meaning of the word 'ephemeral'",
    "The user said 'search for', clearly indicating a web search request. "
    "They want a word definition, which is general web information. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "meaning of ephemeral", "query_type": "general"}}]
))

# =============================================================================
# SECTION 2: Weather tool vs google_search for weather
# Local weather → weather tool. Non-local or "search" + weather → google_search
# =============================================================================

new_entries.append(entry(
    "What's the weather in Tokyo?",
    "The user is asking about weather in Tokyo, which is a different city from my location. "
    "The weather tool only provides forecasts for the user's current location - it does not support "
    "querying other cities. Since this is about a remote location, I need to use google_search "
    "to find weather information for Tokyo.",
    [{"name": "google_search", "arguments": {"query": "weather in Tokyo", "query_type": "general"}}]
))

new_entries.append(entry(
    "How's the weather in London right now?",
    "The user wants weather for London, which is not their current location. "
    "The weather tool only handles local forecasts - it has no location parameter. "
    "For weather in other cities, I must use google_search.",
    [{"name": "google_search", "arguments": {"query": "weather in London right now", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for the weather in New York",
    "The user said 'search for the weather in New York'. Two signals point to google_search here: "
    "(1) the explicit word 'search', and (2) they are asking about a specific city (New York) "
    "which the local-only weather tool cannot handle. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "weather in New York", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the weather in Miami this weekend?",
    "The user is asking about weather in Miami - a specific city that is not their current location. "
    "The weather tool only provides local forecasts and cannot query other cities. "
    "I need to use google_search for non-local weather.",
    [{"name": "google_search", "arguments": {"query": "Miami weather this weekend", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the weather?",
    "The user is asking about the weather without specifying a location. "
    "This means they want their local weather forecast. The weather tool handles local forecasts. "
    "There is no mention of 'search' and no remote city specified. "
    "I will use the weather tool with days=1 for a general forecast, not google_search.",
    [{"name": "weather", "arguments": {"days": 1}}]
))

new_entries.append(entry(
    "How's the weather looking today?",
    "The user wants to know about today's local weather. No other city is mentioned, and they "
    "did not say 'search'. This is a straightforward local weather request. "
    "I will use the weather tool with specific_date='today', not google_search.",
    [{"name": "weather", "arguments": {"specific_date": "today"}}]
))

new_entries.append(entry(
    "Is it going to rain tomorrow?",
    "The user is asking about tomorrow's local weather (will it rain). No remote city is mentioned "
    "and there is no 'search' keyword. This is a local forecast question. "
    "I will use the weather tool with specific_date='tomorrow', not google_search.",
    [{"name": "weather", "arguments": {"specific_date": "tomorrow"}}]
))

new_entries.append(entry(
    "What's the forecast for this week?",
    "The user wants a weekly local forecast. No other city is mentioned, no 'search' keyword. "
    "This is a local weather request for multiple days. I will use weather with days=7, not google_search.",
    [{"name": "weather", "arguments": {"days": 7}}]
))

new_entries.append(entry(
    "Is it warm in Barcelona right now?",
    "The user is asking about the temperature in Barcelona, a specific city that is not their location. "
    "The weather tool only handles local forecasts. Even though there is no 'search' keyword, "
    "the request for weather in a different city requires google_search.",
    [{"name": "google_search", "arguments": {"query": "current temperature in Barcelona", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the temperature in Dubai?",
    "The user wants to know the temperature in Dubai - a remote city. "
    "The weather tool only supports the user's current location and has no city/location parameter. "
    "I must use google_search for non-local weather information.",
    [{"name": "google_search", "arguments": {"query": "temperature in Dubai", "query_type": "general"}}]
))

# =============================================================================
# SECTION 3: Spotify search/play vs google_search for music info
# Playing/finding music → spotify_playback. Music facts/lyrics/history → google_search
# =============================================================================

new_entries.append(entry(
    "Find the song Stairway to Heaven",
    "The user wants to find a specific song - 'Stairway to Heaven'. "
    "Even though they said 'find', the context is clearly about locating a music track for playback. "
    "This maps to spotify_playback with action='search_track', not google_search. "
    "The word 'find' + a song title = Spotify search, not web search.",
    [{"name": "spotify_playback", "arguments": {"action": "search_track", "query": "Stairway to Heaven"}}]
))

new_entries.append(entry(
    "Look up Radiohead",
    "The user said 'look up' followed by an artist name. In the context of music, "
    "'look up' an artist means searching for them on Spotify, not googling them. "
    "I will use spotify_playback with action='search_artist' and query='Radiohead'.",
    [{"name": "spotify_playback", "arguments": {"action": "search_artist", "query": "Radiohead"}}]
))

new_entries.append(entry(
    "Search for the lyrics to Bohemian Rhapsody",
    "The user said 'search for the lyrics' - this is asking for song lyrics, which is web content. "
    "Even though it mentions a song, they want the actual text of the lyrics, not to play the song. "
    "The word 'search' combined with 'lyrics' (information, not playback) means google_search.",
    [{"name": "google_search", "arguments": {"query": "Bohemian Rhapsody lyrics", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search who wrote Yesterday by The Beatles",
    "The user said 'search who wrote' - they want factual information about a song's songwriter. "
    "This is a knowledge question, not a music playback request. "
    "The word 'search' + a factual question = google_search, not spotify.",
    [{"name": "google_search", "arguments": {"query": "who wrote Yesterday by The Beatles", "query_type": "general"}}]
))

new_entries.append(entry(
    "Play some jazz",
    "The user wants to listen to jazz music. There is no 'search' keyword and the intent is playback. "
    "This maps directly to spotify_playback with action='play', not google_search. "
    "'Play' is a strong signal for the Spotify tool.",
    [{"name": "spotify_playback", "arguments": {"action": "play", "query": "jazz", "search_type": "artist"}}]
))

new_entries.append(entry(
    "Search for the history of jazz music",
    "The user said 'search for the history of jazz'. Despite mentioning a music genre, "
    "they want historical information about jazz, not to play jazz. "
    "The word 'search' + 'history' = web information request = google_search.",
    [{"name": "google_search", "arguments": {"query": "history of jazz music", "query_type": "general"}}]
))

new_entries.append(entry(
    "Find songs by Taylor Swift",
    "The user wants to find songs by Taylor Swift. 'Find songs by [artist]' is a music discovery "
    "request in the context of available playback tools. This maps to spotify_playback "
    "with search_artist, not google_search. There is no 'search' keyword and the context is music playback.",
    [{"name": "spotify_playback", "arguments": {"action": "search_artist", "query": "Taylor Swift"}}]
))

new_entries.append(entry(
    "Search for Taylor Swift net worth",
    "The user said 'search for' followed by a celebrity fact question. "
    "Even though Taylor Swift is a musician, they want biographical/financial information, not music playback. "
    "The word 'search' + factual info = google_search.",
    [{"name": "google_search", "arguments": {"query": "Taylor Swift net worth", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's playing right now?",
    "The user wants to know the currently playing track. This is a Spotify status request, "
    "not a web search. There is no 'search' keyword. "
    "'What's playing' in the context of music = spotify_playback with action='status'.",
    [{"name": "spotify_playback", "arguments": {"action": "status"}}]
))

new_entries.append(entry(
    "Search for upcoming concerts near me",
    "The user said 'search for upcoming concerts'. This is asking for event listings from the web, "
    "not requesting music playback or checking their calendar. "
    "The word 'search' + external event info = google_search.",
    [{"name": "google_search", "arguments": {"query": "upcoming concerts near me", "query_type": "general"}}]
))

# =============================================================================
# SECTION 4: Calendar/Stickies (personal data) vs google_search
# =============================================================================

new_entries.append(entry(
    "What do I have today?",
    "The user is asking about their personal schedule for today. "
    "'What do I have' refers to their own calendar events, not a web search. "
    "There is no 'search' keyword. This maps to calendar_data with read_type='day_summary'.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "day_summary", "calendar": "all"}]}}]
))

new_entries.append(entry(
    "Search for events happening this weekend",
    "The user said 'search for events'. The word 'search' combined with 'events happening' "
    "means they want to find public events in their area from the web - not check their personal calendar. "
    "If they wanted their own calendar, they would say 'what's on my calendar' or 'check my schedule'. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "events happening this weekend", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's on my schedule tomorrow?",
    "The user is asking about their personal schedule. The word 'my' indicates personal data. "
    "There is no 'search' keyword. This maps to calendar_data, not google_search.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "specific_date", "date": "tomorrow", "calendar": "all"}]}}]
))

new_entries.append(entry(
    "Search for fun things to do this weekend",
    "The user said 'search for fun things to do'. They want activity suggestions from the web. "
    "The word 'search' signals web information, and 'things to do' is general interest, not personal calendar data. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "fun things to do this weekend", "query_type": "general"}}]
))

new_entries.append(entry(
    "Check my to-do list",
    "The user wants to see their personal to-do list. 'My to-do list' refers to their stickies data. "
    "There is no 'search' keyword and this is personal data. "
    "I will use stickies with action='read' and section='todo', not google_search.",
    [{"name": "stickies", "arguments": {"action": "read", "section": "todo"}}]
))

new_entries.append(entry(
    "Search for productivity tips",
    "The user said 'search for productivity tips'. The word 'search' signals web information. "
    "They want general advice from the internet, not their personal to-do list or notes. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "productivity tips", "query_type": "general"}}]
))

new_entries.append(entry(
    "Show me my notes",
    "The user wants to see their personal notes stored in stickies. "
    "'My notes' refers to personal data. There is no 'search' keyword. "
    "This maps to stickies with action='read' and section='notes'.",
    [{"name": "stickies", "arguments": {"action": "read", "section": "notes"}}]
))

new_entries.append(entry(
    "Search for how to take better notes",
    "The user said 'search for how to take better notes'. Despite mentioning 'notes', "
    "the word 'search' and the instructional nature of the request indicate they want web information. "
    "They are not asking to read their personal notes. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "how to take better notes", "query_type": "general"}}]
))

# =============================================================================
# SECTION 5: General knowledge / "what is" / "who is" questions → google_search
# These have no 'search' keyword but are clearly web lookups
# =============================================================================

new_entries.append(entry(
    "Who is the president of France?",
    "The user is asking a factual question about the current president of France. "
    "This is a general knowledge question requiring up-to-date web information. "
    "No personal tool (weather, calendar, spotify, stickies) can answer this. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "president of France", "query_type": "general"}}]
))

new_entries.append(entry(
    "How tall is the Eiffel Tower?",
    "The user is asking a factual question about the Eiffel Tower's height. "
    "This is general knowledge that requires a web search. "
    "No available personal tool handles trivia or factual lookups.",
    [{"name": "google_search", "arguments": {"query": "how tall is the Eiffel Tower", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the population of Japan?",
    "The user wants to know Japan's population - a factual question. "
    "This requires web information. It does not match any personal tool like weather, calendar, or spotify.",
    [{"name": "google_search", "arguments": {"query": "population of Japan", "query_type": "general"}}]
))

new_entries.append(entry(
    "When was the Great Wall of China built?",
    "The user is asking a historical factual question. "
    "This requires web information and does not map to any personal tool.",
    [{"name": "google_search", "arguments": {"query": "when was the Great Wall of China built", "query_type": "general"}}]
))

new_entries.append(entry(
    "What does NASA stand for?",
    "The user is asking what an acronym stands for - a factual question. "
    "This requires web information and maps to google_search.",
    [{"name": "google_search", "arguments": {"query": "what does NASA stand for", "query_type": "general"}}]
))

new_entries.append(entry(
    "How do you say hello in Japanese?",
    "The user wants a translation/language question answered. "
    "This is general knowledge from the web. No personal tool handles translations. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "how to say hello in Japanese", "query_type": "general"}}]
))

# =============================================================================
# SECTION 6: "Find" / "Look up" disambiguation - context determines tool
# =============================================================================

new_entries.append(entry(
    "Find me a good restaurant",
    "The user wants to find a restaurant. Although they did not say 'search', "
    "this is a request for external information (restaurant recommendations) from the web. "
    "No personal tool handles restaurant discovery. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "good restaurants near me", "query_type": "general"}}]
))

new_entries.append(entry(
    "Find my upcoming events",
    "The user said 'find my upcoming events'. The word 'my' signals personal data. "
    "They want to see their own calendar events. Despite the word 'find', the context is "
    "personal schedule data, not a web search. I will use calendar_data with read_type='next_events'.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "next_events", "calendar": "all"}]}}]
))

new_entries.append(entry(
    "Look up cheap flights to Hawaii",
    "The user said 'look up cheap flights'. This is a request for travel pricing information "
    "from the web. 'Look up' here functions the same as 'search for'. "
    "No personal tool handles flight searches. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "cheap flights to Hawaii", "query_type": "general"}}]
))

new_entries.append(entry(
    "Look up what I have on Wednesday",
    "The user said 'look up what I have on Wednesday'. Despite 'look up', the phrase 'what I have' "
    "indicates personal calendar data. They want to check their own schedule for Wednesday. "
    "This is calendar_data, not google_search.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "specific_date", "date": "wednesday", "calendar": "all"}]}}]
))

new_entries.append(entry(
    "Find out how to reset my iPhone",
    "The user wants to find out how to do something - this is a 'how to' question. "
    "Despite no explicit 'search' keyword, 'find out how to' is asking for instructional "
    "web content. No personal tool handles tech support. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "how to reset iPhone", "query_type": "general"}}]
))

new_entries.append(entry(
    "Find the track Blinding Lights",
    "The user said 'find the track' followed by a song title. In the music context, "
    "'find the track' means searching for a song to play. "
    "This is spotify_playback search_track, not google_search.",
    [{"name": "spotify_playback", "arguments": {"action": "search_track", "query": "Blinding Lights"}}]
))

# =============================================================================
# SECTION 7: "What is" / "How does" → google_search (not confused with status queries)
# =============================================================================

new_entries.append(entry(
    "What is machine learning?",
    "The user is asking a 'what is' conceptual question. "
    "This requires a web-sourced explanation. It does not match any personal tool. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "what is machine learning", "query_type": "general"}}]
))

new_entries.append(entry(
    "How does photosynthesis work?",
    "The user is asking a scientific 'how does X work' question. "
    "This is a knowledge query requiring web information. "
    "No personal tool is relevant.",
    [{"name": "google_search", "arguments": {"query": "how does photosynthesis work", "query_type": "general"}}]
))

new_entries.append(entry(
    "What is playing?",
    "The user is asking 'what is playing' - this is a music playback status question, "
    "not a general knowledge query. Despite the 'what is' phrasing, the context of 'playing' "
    "clearly refers to current music. I will use spotify_playback with action='status', not google_search.",
    [{"name": "spotify_playback", "arguments": {"action": "status"}}]
))

new_entries.append(entry(
    "What is quantum computing?",
    "The user is asking 'what is quantum computing' - a conceptual knowledge question. "
    "Unlike 'what is playing' (which is a Spotify status query), this is asking for a definition/explanation. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "what is quantum computing", "query_type": "general"}}]
))

# =============================================================================
# SECTION 8: Notifications/briefings vs google_search for news
# =============================================================================

new_entries.append(entry(
    "Check my notifications",
    "The user wants to check their personal notifications. 'My notifications' refers to their "
    "pending notification data. There is no 'search' keyword. "
    "I will use get_notifications, not google_search.",
    [{"name": "get_notifications", "arguments": {"type_filter": "all"}}]
))

new_entries.append(entry(
    "Search for the latest tech news",
    "The user said 'search for the latest tech news'. The word 'search' signals a web search. "
    "They want current news articles from the web, not their personal notification feed. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "latest tech news", "query_type": "general"}}]
))

new_entries.append(entry(
    "What are my emails about?",
    "The user is asking about their personal emails. 'My emails' refers to their notification data. "
    "No 'search' keyword. This maps to get_notifications with type_filter='email'.",
    [{"name": "get_notifications", "arguments": {"type_filter": "email"}}]
))

new_entries.append(entry(
    "Search for tips on managing email",
    "The user said 'search for tips'. Despite mentioning 'email', the word 'search' plus "
    "'tips on managing' makes this a web information request, not a request to check their own emails. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "tips on managing email", "query_type": "general"}}]
))

# =============================================================================
# SECTION 9: Link / URL / directions queries → google_search
# =============================================================================

new_entries.append(entry(
    "Find me the link to Reddit",
    "The user wants a URL/link to a specific website. "
    "Even without the word 'search', requesting a website link is a web lookup. "
    "I will use google_search with query_type='link'.",
    [{"name": "google_search", "arguments": {"query": "Reddit", "query_type": "link"}}]
))

new_entries.append(entry(
    "How do I get to the nearest pharmacy?",
    "The user wants navigation directions. This is a directions/location request. "
    "No personal tool handles navigation. I will use google_search with query_type='directions'.",
    [{"name": "google_search", "arguments": {"query": "nearest pharmacy", "query_type": "directions"}}]
))

new_entries.append(entry(
    "Search for directions to the beach",
    "The user said 'search for directions'. Both the word 'search' and the request for directions "
    "point to google_search with query_type='directions'.",
    [{"name": "google_search", "arguments": {"query": "directions to the beach", "query_type": "directions"}}]
))

# =============================================================================
# SECTION 10: Ambiguous "what's the..." that could go either way
# =============================================================================

new_entries.append(entry(
    "What's the best programming language?",
    "The user is asking an opinion/recommendation question about programming languages. "
    "This is general knowledge from the web. It does not match weather, calendar, spotify, or any personal tool. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "best programming language", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the weather this week?",
    "The user is asking about their local weather for the week. No specific city is mentioned "
    "and there is no 'search' keyword. This is a local weather forecast request. "
    "I will use the weather tool with days=7, not google_search.",
    [{"name": "weather", "arguments": {"days": 7}}]
))

new_entries.append(entry(
    "What's on my calendar this week?",
    "The user is asking about their personal calendar for the week. 'My calendar' signals personal data. "
    "No 'search' keyword. I will use calendar_data with read_type='week_summary', not google_search.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "week_summary", "calendar": "all"}]}}]
))

new_entries.append(entry(
    "What's the stock price of Apple?",
    "The user is asking about a stock price - this is real-time financial data from the web. "
    "No personal tool handles stock prices. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "Apple stock price", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's the score of the Lakers game?",
    "The user wants a live sports score. This is real-time data from the web. "
    "No personal tool handles sports scores. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "Lakers game score", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's my next meeting?",
    "The user is asking about their next meeting. 'My next meeting' is personal calendar data. "
    "No 'search' keyword. I will use calendar_data with read_type='next_events', not google_search.",
    [{"name": "calendar_data", "arguments": {"commands": [{"read_or_write": "read", "read_type": "next_events", "calendar": "all"}]}}]
))

# =============================================================================
# SECTION 11: Tricky edge cases
# =============================================================================

new_entries.append(entry(
    "What time is it in Sydney?",
    "The user is asking about the current time in a different city. "
    "No personal tool handles timezone conversions. This is a factual web lookup. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "current time in Sydney", "query_type": "general"}}]
))

new_entries.append(entry(
    "Remind me about the dentist",
    "The user wants a reminder about a dentist appointment. This is a briefing/reminder request, "
    "not a web search. There is no 'search' keyword. "
    "I will use the briefing tool with action='create'.",
    [{"name": "briefing", "arguments": {"action": "create", "message": "Dentist appointment reminder"}}]
))

new_entries.append(entry(
    "Search for a good dentist near me",
    "The user said 'search for a good dentist'. The word 'search' signals a web search. "
    "They want dentist recommendations from the web, not a reminder or calendar entry. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "good dentist near me", "query_type": "general"}}]
))

new_entries.append(entry(
    "Text me the grocery list",
    "The user wants to send themselves a text with a grocery list. "
    "This is an SMS action, not a web search. No 'search' keyword. "
    "I will use send_sms.",
    [{"name": "send_sms", "arguments": {"message": "Grocery list reminder"}}]
))

new_entries.append(entry(
    "Search for grocery delivery services",
    "The user said 'search for grocery delivery services'. The word 'search' indicates a web lookup. "
    "They want information about delivery services from the internet, not a personal tool action. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "grocery delivery services", "query_type": "general"}}]
))

new_entries.append(entry(
    "Turn on the living room lights",
    "The user wants to control their smart lights. This is a kasa_lighting action, not a web search. "
    "There is no 'search' keyword and the intent is home automation. "
    "I will use kasa_lighting with interaction='direct'.",
    [{"name": "kasa_lighting", "arguments": {"interaction": "direct", "light_name": "living room", "action": "on"}}]
))

new_entries.append(entry(
    "Search for smart light bulb recommendations",
    "The user said 'search for smart light bulb recommendations'. The word 'search' signals web info. "
    "Despite mentioning lights, they want product recommendations from the internet, "
    "not to control their home lights. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "smart light bulb recommendations", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's on Netflix tonight?",
    "The user is asking about what's available to watch on Netflix. "
    "This is not personal data - it's external content information. "
    "No personal tool handles streaming catalogs. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "what's on Netflix tonight", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for the best shows on Netflix",
    "The user said 'search for the best shows on Netflix'. The word 'search' plus entertainment "
    "recommendations is clearly a web search request. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "best shows on Netflix", "query_type": "general"}}]
))

new_entries.append(entry(
    "How many calories in an avocado?",
    "The user is asking a factual nutrition question. "
    "This is general knowledge requiring web information. No personal tool handles nutrition data. "
    "I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "calories in avocado", "query_type": "general"}}]
))

new_entries.append(entry(
    "Search for nearby coffee shops",
    "The user said 'search for nearby coffee shops'. The word 'search' plus a local business query "
    "is clearly a google_search request.",
    [{"name": "google_search", "arguments": {"query": "coffee shops near me", "query_type": "general"}}]
))

new_entries.append(entry(
    "What's my clipboard say?",
    "The user wants to know what is on their clipboard. This is a personal data request. "
    "No 'search' keyword. I will use read_clipboard, not google_search.",
    [{"name": "read_clipboard", "arguments": {}}]
))

new_entries.append(entry(
    "Search for clipboard manager apps",
    "The user said 'search for clipboard manager apps'. The word 'search' signals web info. "
    "Despite mentioning 'clipboard', they want software recommendations from the web, "
    "not to read their clipboard contents. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "clipboard manager apps", "query_type": "general"}}]
))

# =============================================================================
# SECTION 12: No "search" but still google_search (factual questions with no tool match)
# =============================================================================

new_entries.append(entry(
    "Why is the sky blue?",
    "The user is asking a scientific question. Even without the word 'search', "
    "this is a factual/educational query that requires web information. "
    "No personal tool can answer this. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "why is the sky blue", "query_type": "general"}}]
))

new_entries.append(entry(
    "What happened in the news today?",
    "The user wants current news. Although they did not say 'search', this is asking for "
    "external news content from the web. This could also be get_notifications with type_filter='news' "
    "if checking their notification feed, but the phrasing 'in the news' suggests general world news. "
    "I will use google_search for broad current events.",
    [{"name": "google_search", "arguments": {"query": "news today", "query_type": "general"}}]
))

new_entries.append(entry(
    "How far is it to the moon?",
    "The user is asking a factual distance question. No 'search' keyword, but this is clearly "
    "a knowledge question. No personal tool is relevant. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "distance to the moon", "query_type": "general"}}]
))

new_entries.append(entry(
    "When is Thanksgiving this year?",
    "The user is asking about a holiday date. This is factual calendar knowledge from the web, "
    "not their personal calendar. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "Thanksgiving date this year", "query_type": "general"}}]
))

new_entries.append(entry(
    "Is the stock market open today?",
    "The user is asking about stock market hours. This is financial/factual web information. "
    "No personal tool handles this. I will use google_search.",
    [{"name": "google_search", "arguments": {"query": "is the stock market open today", "query_type": "general"}}]
))


# =============================================================================
# Write all entries
# =============================================================================
print(f"Generated {len(new_entries)} new disambiguation entries")

with open("training_data.jsonl", "a") as f:
    for e in new_entries:
        f.write(e + "\n")

# Verify
with open("training_data.jsonl") as f:
    total = sum(1 for _ in f)
print(f"Total entries in training_data.jsonl: {total}")
