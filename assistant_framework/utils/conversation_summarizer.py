"""
Conversation summarizer using Gemini.

Periodically summarizes conversation history to a file for reference.
Runs asynchronously in the background to avoid blocking the main flow.
"""

import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
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


class ConversationSummarizer:
    """
    Summarizes conversation history using Gemini and writes to a file.
    
    Triggers:
    - First summary at `first_summary_at` messages
    - Then every `summarize_every` additional messages
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the summarizer.
        
        Args:
            config: Summarization config dict with:
                - enabled: bool
                - first_summary_at: int (default 10)
                - summarize_every: int (default 4)
                - output_file: str (path to output JSON)
                - gemini_model: str (default "gemini-2.0-flash")
        """
        self.enabled = config.get("enabled", True)
        self.first_summary_at = config.get("first_summary_at", 10)
        self.summarize_every = config.get("summarize_every", 4)
        self.output_file = config.get("output_file", "state_management/conversation_summary.json")
        self.gemini_model = config.get("gemini_model", "gemini-2.0-flash")
        self.prompt_template = config.get("prompt", self._default_prompt())
        
        # State tracking
        self._message_count = 0
        self._last_summary_at = 0
        self._is_summarizing = False
        self._current_summary: Optional[str] = None
        
        # Ensure output directory exists
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load previous summary if exists (for continuity across restarts)
        self._current_summary = self.load_previous_summary()
    
    def _default_prompt(self) -> str:
        """Default summarization prompt template."""
        return """Summarize or update the summary of this conversation between a user and an AI assistant.

{previous_summary_section}

Focus on:
1. Key topics discussed
2. Important information shared (names, dates, preferences, requests)
3. Any actions taken or tools used
4. Ongoing context that would be useful for future interactions

If a previous summary exists, integrate new information into it rather than starting fresh.
Keep the summary length proportional to the conversation - longer conversations warrant more detail.

CONVERSATION:
{conversation}

SUMMARY:"""
    
    def _calculate_max_tokens(self, conversation_text: str) -> int:
        """
        Calculate proportional max_tokens based on conversation length.
        
        Roughly targets 10-15% of input length, with bounds.
        """
        input_chars = len(conversation_text)
        # Estimate ~4 chars per token, target ~12% of input tokens
        estimated_input_tokens = input_chars / 4
        target_output = int(estimated_input_tokens * 0.12)
        
        # Clamp between reasonable bounds
        min_tokens = 150
        max_tokens = 1500
        return max(min_tokens, min(target_output, max_tokens))
    
    def should_summarize(self, message_count: int) -> bool:
        """
        Check if we should trigger a summarization.
        
        Args:
            message_count: Current total message count
            
        Returns:
            True if summarization should be triggered
        """
        if not self.enabled:
            return False
        
        if self._is_summarizing:
            return False
        
        # First summary threshold
        if self._last_summary_at == 0:
            return message_count >= self.first_summary_at
        
        # Subsequent summaries every N messages
        messages_since_last = message_count - self._last_summary_at
        return messages_since_last >= self.summarize_every
    
    def summarize_async(self, messages: List[Dict[str, Any]], message_count: int) -> None:
        """
        Trigger asynchronous summarization in background thread.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            message_count: Current message count (for tracking)
        """
        if self._is_summarizing:
            return
        
        self._is_summarizing = True
        thread = threading.Thread(
            target=self._summarize_sync,
            args=(messages, message_count),
            daemon=True
        )
        thread.start()
    
    def _summarize_sync(self, messages: List[Dict[str, Any]], message_count: int) -> None:
        """
        Synchronous summarization (runs in background thread).
        """
        try:
            genai = _get_gemini_client()
            if not genai:
                print("⚠️  Gemini not configured - skipping summarization")
                return
            
            # Build conversation text
            conversation_text = self._format_messages(messages)
            
            # Calculate proportional max_tokens
            max_tokens = self._calculate_max_tokens(conversation_text)
            
            # Build previous summary section for context
            previous_summary = self._current_summary or self.load_previous_summary()
            if previous_summary:
                previous_summary_section = f"PREVIOUS SUMMARY (update this with new information):\n{previous_summary}"
            else:
                previous_summary_section = "No previous summary exists - create a new one."
            
            # Generate summary
            model = genai.GenerativeModel(
                self.gemini_model,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": max_tokens,
                }
            )
            
            # Use configurable prompt template with previous summary context
            prompt = self.prompt_template.format(
                conversation=conversation_text,
                previous_summary_section=previous_summary_section
            )
            
            response = model.generate_content(prompt)
            summary = response.text.strip()
            
            # Save to file
            self._save_summary(summary, messages, message_count)
            
            self._current_summary = summary
            self._last_summary_at = message_count
            print(f"📝 Conversation summarized ({message_count} messages, {max_tokens} max tokens)")
            
        except Exception as e:
            print(f"⚠️  Summarization failed: {e}")
        finally:
            self._is_summarizing = False
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for the prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
    
    def _save_summary(self, summary: str, messages: List[Dict[str, Any]], message_count: int) -> None:
        """Save summary to JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "message_count": message_count,
            "summary": summary,
            "topics": self._extract_topics(summary),
            "messages_summarized": len(messages)
        }
        
        # Load existing if present
        existing_summaries = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    existing = json.load(f)
                    existing_summaries = existing.get("history", [])
            except Exception:
                pass
        
        # Keep last 10 summaries
        existing_summaries.append(data)
        existing_summaries = existing_summaries[-10:]
        
        output = {
            "current_summary": summary,
            "last_updated": data["timestamp"],
            "total_messages": message_count,
            "history": existing_summaries
        }
        
        with open(self.output_file, "w") as f:
            json.dump(output, f, indent=2)
    
    def _extract_topics(self, summary: str) -> List[str]:
        """Extract key topics from summary (simple keyword extraction)."""
        # Simple extraction - could be enhanced with NLP
        topics = []
        keywords = ["weather", "calendar", "lights", "music", "timer", "reminder", 
                    "spotify", "search", "email", "news", "time", "date"]
        summary_lower = summary.lower()
        for kw in keywords:
            if kw in summary_lower:
                topics.append(kw)
        return topics
    
    @property
    def current_summary(self) -> Optional[str]:
        """Get the current conversation summary."""
        return self._current_summary
    
    def load_previous_summary(self) -> Optional[str]:
        """Load the most recent summary from file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    data = json.load(f)
                    return data.get("current_summary")
            except Exception:
                pass
        return None
    
    def reset(self) -> None:
        """Reset summarizer state for new conversation session."""
        self._message_count = 0
        self._last_summary_at = 0
        self._current_summary = None

