"""
Persistent Memory Manager.

Maintains long-term memory across all conversations.
Updated at the end of each conversation session with lasting facts and preferences.
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


def _get_gemini_client():
    """Lazy-load Gemini client to avoid import errors if not configured."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        return None


class PersistentMemoryManager:
    """
    Manages persistent long-term memory across all conversations.
    
    This captures:
    - User preferences (units, voice, tone)
    - Personal information (name, location, timezone)
    - Frequently used tools/requests
    - Important facts mentioned over time
    - Behavioral patterns
    
    Updated at the end of each conversation session.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the persistent memory manager.
        
        Args:
            config: Configuration dict with:
                - enabled: bool
                - output_file: str (path to output JSON)
                - gemini_model: str (default "gemini-2.0-flash")
                - prompt: str (extraction prompt template)
        """
        self.enabled = config.get("enabled", True)
        self.output_file = config.get("output_file", "state_management/persistent_memory.json")
        self.gemini_model = config.get("gemini_model", "gemini-2.0-flash")
        self.prompt_template = config.get("prompt", self._default_prompt())
        
        # State tracking
        self._is_updating = False
        self._current_memory: Optional[Dict[str, Any]] = None
        
        # Ensure output directory exists
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory on startup
        self._current_memory = self.load_memory()
    
    def _default_prompt(self) -> str:
        """Default prompt template for extracting lasting information."""
        return """You are updating a persistent memory store for a personal AI assistant.

{existing_memory_section}

Based on the conversation summary below, extract any NEW lasting information that should be remembered across ALL future conversations.

IMPORTANT: Be EXTREMELY concise. Use the absolute minimum words necessary. 
- Facts should be 3-15 words each
- Preferences as single key-value pairs
- No explanations, no elaboration, no redundancy

Extract ONLY:
1. User preferences (as terse key-value pairs)
2. Personal info (name, location - single words/phrases)
3. Important lasting facts (3-8 words max each)
4. Corrections to existing memory

DO NOT include:
- Temporary information (weather, current events)
- One-time requests
- Anything already in existing memory (unless correcting it)
- Verbose descriptions or explanations

CONVERSATION SUMMARY:
{conversation_summary}

Respond with MINIMAL JSON:
{{
    "user_profile": {{"name": "str|null", "location": "str|null", "preferences": {{}}}},
    "known_facts": ["terse fact 1", "terse fact 2"],
    "corrections": [],
    "new_patterns": []
}}

Omit empty fields. Be terse.
JSON:"""
    
    def update_after_conversation(self, conversation_summary: str) -> None:
        """
        Trigger asynchronous memory update after a conversation ends.
        
        Args:
            conversation_summary: Summary of the conversation that just ended
        """
        if not self.enabled:
            return
        
        if self._is_updating:
            print("⚠️  Memory update already in progress, skipping")
            return
        
        if not conversation_summary or not conversation_summary.strip():
            print("⚠️  No conversation summary provided, skipping memory update")
            return
        
        self._is_updating = True
        thread = threading.Thread(
            target=self._update_sync,
            args=(conversation_summary,),
            daemon=True
        )
        thread.start()
    
    def _update_sync(self, conversation_summary: str) -> None:
        """
        Synchronous memory update (runs in background thread).
        """
        try:
            genai = _get_gemini_client()
            if not genai:
                print("⚠️  Gemini not configured - skipping persistent memory update")
                return
            
            # Build existing memory section
            if self._current_memory:
                existing_memory_section = f"EXISTING MEMORY:\n{json.dumps(self._current_memory, indent=2)}"
            else:
                existing_memory_section = "EXISTING MEMORY: None (this is the first conversation)"
            
            # Generate memory extraction
            model = genai.GenerativeModel(
                self.gemini_model,
                generation_config={
                    "temperature": 0.2,  # Low temperature for consistency
                }
            )
            
            prompt = self.prompt_template.format(
                existing_memory_section=existing_memory_section,
                conversation_summary=conversation_summary
            )
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            new_memory = self._parse_memory_response(response_text)
            
            if new_memory:
                # Merge with existing memory
                merged_memory = self._merge_memory(self._current_memory, new_memory)
                
                # Save to file
                self._save_memory(merged_memory, conversation_summary)
                
                self._current_memory = merged_memory
                print("🧠 Persistent memory updated")
            else:
                print("ℹ️  No new lasting information to remember")
            
        except Exception as e:
            print(f"⚠️  Persistent memory update failed: {e}")
        finally:
            self._is_updating = False
    
    def _parse_memory_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON response from Gemini."""
        try:
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            parsed = json.loads(response_text)
            
            # Validate structure
            if not isinstance(parsed, dict):
                return None
            
            # Check if there's actually new information
            has_new_info = (
                parsed.get("user_profile", {}).get("name") or
                parsed.get("user_profile", {}).get("location") or
                parsed.get("user_profile", {}).get("preferences") or
                parsed.get("known_facts") or
                parsed.get("corrections") or
                parsed.get("new_patterns")
            )
            
            return parsed if has_new_info else None
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse memory response: {e}")
            return None
    
    def _merge_memory(self, existing: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new memory with existing memory."""
        if not existing:
            existing = {
                "user_profile": {"name": None, "location": None, "preferences": {}},
                "known_facts": [],
                "patterns": [],
                "update_history": []
            }
        
        merged = existing.copy()
        
        # Update user profile
        new_profile = new.get("user_profile", {})
        if new_profile:
            if new_profile.get("name"):
                merged["user_profile"]["name"] = new_profile["name"]
            if new_profile.get("location"):
                merged["user_profile"]["location"] = new_profile["location"]
            if new_profile.get("preferences"):
                merged["user_profile"]["preferences"].update(new_profile["preferences"])
        
        # Add new facts (avoid duplicates)
        new_facts = new.get("known_facts", [])
        existing_facts = set(merged.get("known_facts", []))
        for fact in new_facts:
            if fact and fact not in existing_facts:
                merged.setdefault("known_facts", []).append(fact)
        
        # Add new patterns
        new_patterns = new.get("new_patterns", [])
        existing_patterns = set(merged.get("patterns", []))
        for pattern in new_patterns:
            if pattern and pattern not in existing_patterns:
                merged.setdefault("patterns", []).append(pattern)
        
        # Handle corrections (log them but don't auto-apply - just note them)
        corrections = new.get("corrections", [])
        if corrections:
            merged.setdefault("corrections_log", []).extend([
                {"correction": c, "timestamp": datetime.utcnow().isoformat()}
                for c in corrections
            ])
            # Keep only last 20 corrections
            merged["corrections_log"] = merged["corrections_log"][-20:]
        
        return merged
    
    def _save_memory(self, memory: Dict[str, Any], source_summary: str) -> None:
        """Save memory to JSON file."""
        # Add metadata
        update_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_summary_preview": source_summary[:200] + "..." if len(source_summary) > 200 else source_summary
        }
        
        memory.setdefault("update_history", []).append(update_entry)
        # Keep only last 50 updates
        memory["update_history"] = memory["update_history"][-50:]
        
        memory["last_updated"] = update_entry["timestamp"]
        
        with open(self.output_file, "w") as f:
            json.dump(memory, f, indent=2)
    
    def load_memory(self) -> Optional[Dict[str, Any]]:
        """Load the current memory from file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load persistent memory: {e}")
        return None
    
    @property
    def current_memory(self) -> Optional[Dict[str, Any]]:
        """Get the current persistent memory."""
        return self._current_memory
    
    def get_memory_summary(self) -> str:
        """Get a text summary of current memory for injection into prompts."""
        if not self._current_memory:
            return ""
        
        parts = []
        
        profile = self._current_memory.get("user_profile", {})
        if profile.get("name"):
            parts.append(f"User's name: {profile['name']}")
        if profile.get("location"):
            parts.append(f"User's location: {profile['location']}")
        if profile.get("preferences"):
            prefs = ", ".join(f"{k}: {v}" for k, v in profile["preferences"].items())
            parts.append(f"User preferences: {prefs}")
        
        facts = self._current_memory.get("known_facts", [])
        if facts:
            parts.append(f"Known facts: {'; '.join(facts[:10])}")  # Limit to 10 facts
        
        return "\n".join(parts)
    
    def reset(self) -> None:
        """Reset persistent memory (use with caution!)."""
        self._current_memory = None
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        print("🗑️  Persistent memory reset")
