"""
Calendar Reminder Analyzer Agent

Uses AI to analyze calendar events and determine optimal reminder timing
based on event type, preparation needs, location, and other factors.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Setup project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Try to import Gemini for AI analysis
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


class ReminderAnalyzer:
    """
    Agent that analyzes calendar events and determines optimal reminder timing.
    
    Considers factors like:
    - Event type (meeting, appointment, deadline, etc.)
    - Location/travel time needed
    - Preparation requirements
    - Event importance/priority
    - Time of day
    """
    
    # Default reminder suggestions by event type (in minutes before event)
    DEFAULT_REMINDERS = {
        "meeting": [30, 10],
        "appointment": [60, 30],
        "deadline": [1440, 60],  # 24 hours and 1 hour
        "travel": [120, 60],
        "social": [60, 15],
        "work": [30, 10],
        "personal": [30],
        "default": [30, 10],
    }
    
    def __init__(self):
        """Initialize the analyzer with AI configuration."""
        self.model = None
        self._configure_ai()
    
    def _configure_ai(self):
        """Configure the Gemini AI model if available."""
        if not GENAI_AVAILABLE:
            print("⚠️  google-generativeai not installed, using heuristic analysis only")
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  GEMINI_API_KEY not set, using heuristic analysis only")
            return
        
        try:
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.model = genai.GenerativeModel(
                model_name,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 4000,
                    "response_mime_type": "application/json",
                },
            )
            print(f"✅ Reminder analyzer initialized with {model_name}")
        except Exception as e:
            print(f"⚠️  Failed to initialize AI model: {e}")
            self.model = None
    
    def analyze_events(self, events: List[Dict[str, Any]], user: str) -> List[Dict[str, Any]]:
        """
        Analyze a list of calendar events and determine optimal reminder timing.
        
        Args:
            events: List of formatted calendar events
            user: User identifier for context
            
        Returns:
            List of reminder suggestions for each event
        """
        if not events:
            print("📅 No events to analyze")
            return []
        
        print(f"🔍 Analyzing {len(events)} events for {user}...")
        
        if self.model:
            return self._ai_analyze_events(events, user)
        else:
            return self._heuristic_analyze_events(events)
    
    def _ai_analyze_events(self, events: List[Dict[str, Any]], user: str) -> List[Dict[str, Any]]:
        """Use AI to analyze events and suggest optimal reminder timing."""
        # Build event summaries for the prompt
        event_summaries = []
        for i, event in enumerate(events[:20]):  # Cap to keep prompt bounded
            summary = self._format_event_for_prompt(event, i)
            event_summaries.append(summary)
        
        events_block = "\n\n".join(event_summaries)
        current_time = datetime.now(timezone.utc).isoformat()
        
        prompt = f"""Analyze calendar events and suggest reminder timing in minutes before each event.

Events:
{events_block}

Return JSON array. Each item needs: event_index (int), event_type (string), reminders_minutes_before (array of ints), priority (high/medium/low).

Example: [{{"event_index":0,"event_type":"travel","reminders_minutes_before":[1440,120],"priority":"high"}}]"""
        
        try:
            response = self.model.generate_content(prompt)
            content = self._extract_response_text(response)
            
            if content:
                try:
                    results = json.loads(content)
                except json.JSONDecodeError:
                    # Try to fix truncated JSON by closing brackets
                    fixed = content.rstrip()
                    if not fixed.endswith("]"):
                        # Count open brackets
                        open_brackets = fixed.count("[") - fixed.count("]")
                        open_braces = fixed.count("{") - fixed.count("}")
                        fixed += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
                    results = json.loads(fixed)
                
                if isinstance(results, list):
                    print(f"✅ AI analyzed {len(results)} events")
                    return self._merge_ai_results(events, results)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse AI response as JSON: {e}")
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
        
        # Fallback to heuristic if AI fails
        print("📊 Falling back to heuristic analysis")
        return self._heuristic_analyze_events(events)
    
    def _format_event_for_prompt(self, event: Dict[str, Any], index: int) -> str:
        """Format a single event for the AI prompt."""
        title = event.get("summary", "No Title")
        start_date = event.get("start_date", "Unknown")
        start_time = event.get("start_time", "Unknown")
        end_time = event.get("end_time", "Unknown")
        location = event.get("location", "Not specified")
        description = event.get("description", "")[:200]
        calendar = event.get("calendar_name", "Primary")
        event_id = event.get("id", f"event_{index}")
        all_day = event.get("all_day", False)
        
        return f"""Event {index}:
- ID: {event_id}
- Title: {title}
- Date: {start_date}
- Time: {start_time} - {end_time} {"(All Day)" if all_day else ""}
- Calendar: {calendar}
- Location: {location}
- Description: {description if description else "None"}"""
    
    def _extract_response_text(self, response) -> str:
        """Extract text content from Gemini response and clean JSON."""
        content = ""
        try:
            candidates = getattr(response, "candidates", []) or []
            for c in candidates:
                cont = getattr(c, "content", None)
                parts = getattr(cont, "parts", []) if cont else []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        content += t
        except Exception:
            try:
                content = response.text or ""
            except Exception:
                content = ""
        
        content = content.strip()
        
        # Strip markdown code blocks if present
        if "```json" in content:
            content = content.split("```json", 1)[-1]
        if "```" in content:
            content = content.split("```")[0]
        
        # Try to extract JSON array from content
        content = content.strip()
        if content and not content.startswith("["):
            # Find the JSON array start
            idx = content.find("[")
            if idx >= 0:
                content = content[idx:]
        
        return content.strip()
    
    def _merge_ai_results(
        self, events: List[Dict[str, Any]], ai_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge AI analysis results with original event data."""
        merged = []
        ai_by_index = {r.get("event_index", -1): r for r in ai_results}
        
        for i, event in enumerate(events):
            ai_result = ai_by_index.get(i, {})
            
            # Ensure reminders is a list of ints
            reminders = ai_result.get("reminders_minutes_before", self.DEFAULT_REMINDERS["default"])
            if not isinstance(reminders, list):
                reminders = self.DEFAULT_REMINDERS["default"]
            
            merged.append({
                "event_id": event.get("id", f"event_{i}"),
                "event_title": event.get("summary", "No Title"),
                "event_date": event.get("start_date"),
                "event_time": event.get("start_time"),
                "calendar_name": event.get("calendar_name"),
                "event_type": ai_result.get("event_type", "default"),
                "reminders_minutes_before": reminders,
                "priority": ai_result.get("priority", "medium"),
                "reasoning": f"AI analysis: {ai_result.get('event_type', 'default')} event",
                "preparation_notes": "",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            })
        
        return merged
    
    def _heuristic_analyze_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use heuristic rules to determine reminder timing when AI is unavailable."""
        results = []
        
        for i, event in enumerate(events):
            title = (event.get("summary", "") or "").lower()
            location = (event.get("location", "") or "").lower()
            description = (event.get("description", "") or "").lower()
            all_day = event.get("all_day", False)
            
            # Determine event type from keywords
            event_type = self._classify_event_type(title, description)
            
            # Get base reminders for event type
            reminders = list(self.DEFAULT_REMINDERS.get(event_type, self.DEFAULT_REMINDERS["default"]))
            
            # Adjust for location/travel
            if location and any(kw in location for kw in ["airport", "station", "downtown", "office"]):
                # Add earlier reminder for travel
                if 120 not in reminders:
                    reminders.insert(0, 120)
            
            # Adjust for all-day events (deadlines often)
            if all_day:
                reminders = [1440, 480]  # Day before and morning of
            
            # Determine priority
            priority = self._determine_priority(title, description)
            
            results.append({
                "event_id": event.get("id", f"event_{i}"),
                "event_title": event.get("summary", "No Title"),
                "event_date": event.get("start_date"),
                "event_time": event.get("start_time"),
                "calendar_name": event.get("calendar_name"),
                "event_type": event_type,
                "reminders_minutes_before": sorted(reminders, reverse=True),
                "priority": priority,
                "reasoning": f"Heuristic analysis: {event_type} event",
                "preparation_notes": "",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            })
        
        return results
    
    def _classify_event_type(self, title: str, description: str) -> str:
        """Classify event type based on keywords in title/description."""
        combined = f"{title} {description}"
        
        type_keywords = {
            "meeting": ["meeting", "sync", "standup", "call", "zoom", "teams", "1:1", "one-on-one"],
            "appointment": ["appointment", "doctor", "dentist", "interview", "consultation"],
            "deadline": ["deadline", "due", "submit", "final", "assignment"],
            "travel": ["flight", "airport", "train", "travel", "trip", "vacation"],
            "social": ["dinner", "lunch", "party", "birthday", "hangout", "drinks", "coffee"],
            "work": ["review", "presentation", "demo", "sprint", "planning"],
        }
        
        for event_type, keywords in type_keywords.items():
            if any(kw in combined for kw in keywords):
                return event_type
        
        return "default"
    
    def _determine_priority(self, title: str, description: str) -> str:
        """Determine event priority based on keywords."""
        combined = f"{title} {description}".lower()
        
        high_priority_keywords = [
            "urgent", "important", "deadline", "interview", "final",
            "presentation", "client", "exam", "doctor"
        ]
        low_priority_keywords = [
            "optional", "tentative", "maybe", "catch up"
        ]
        
        if any(kw in combined for kw in high_priority_keywords):
            return "high"
        elif any(kw in combined for kw in low_priority_keywords):
            return "low"
        
        return "medium"
    
    def format_reminder_summary(self, analyses: List[Dict[str, Any]]) -> str:
        """Format analysis results into a human-readable summary."""
        if not analyses:
            return "No events analyzed."
        
        lines = [
            f"📅 Reminder Analysis Summary ({len(analyses)} events)",
            "=" * 50,
        ]
        
        for analysis in analyses:
            title = analysis.get("event_title", "Unknown")
            date = analysis.get("event_date", "")
            time = analysis.get("event_time", "")
            priority = analysis.get("priority", "medium")
            reminders = analysis.get("reminders_minutes_before", [])
            
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            
            reminder_strs = []
            for mins in reminders:
                if mins >= 1440:
                    reminder_strs.append(f"{mins // 1440}d before")
                elif mins >= 60:
                    reminder_strs.append(f"{mins // 60}h before")
                else:
                    reminder_strs.append(f"{mins}m before")
            
            lines.append(f"\n{priority_emoji} {title}")
            lines.append(f"   📆 {date} at {time}")
            lines.append(f"   ⏰ Reminders: {', '.join(reminder_strs)}")
            
            if analysis.get("preparation_notes"):
                lines.append(f"   📝 {analysis['preparation_notes']}")
        
        return "\n".join(lines)

