"""
Briefing Processor - Pre-generates conversation openers from briefing announcements.

This utility transforms briefings into natural conversation opener statements via LLM,
so that wake word activation only requires TTS (no LLM latency).

Usage by briefing input sources:
    from assistant_framework.utils.briefing.briefing_processor import BriefingProcessor
    
    processor = BriefingProcessor()
    
    # After inserting a briefing to Supabase, generate its opener:
    opener = await processor.generate_opener(briefing_content={
        "message": "Your package arrived",
        "llm_instructions": "mention casually",
        "meta": {"source": "delivery"}
    })
    
    # Or process all pending briefings that don't have openers yet:
    await processor.process_pending_briefings(user="Morgan")
"""

import os
import json
from typing import Dict, Any, Optional, List

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from dotenv import load_dotenv

# Import config - with fallback defaults
try:
    from ..config import BRIEFING_PROCESSOR_CONFIG
except ImportError:
    try:
        from assistant_framework.config import BRIEFING_PROCESSOR_CONFIG
    except ImportError:
        # Fallback defaults if config not available
        BRIEFING_PROCESSOR_CONFIG = {
            "model": "gpt-4o-mini",
            "max_tokens_single": 150,
            "max_tokens_combined": 200,
            "temperature": 0.7,
            "system_prompt": """You are a friendly voice assistant generating a brief conversation opener.

Given one or more briefings to share with the user, create a natural, concise spoken greeting that:
- Sounds warm and conversational (not robotic)
- Mentions the briefings naturally
- Is brief (1-3 sentences max)
- Ends by asking how you can help OR offering to handle/dismiss the items

Do NOT:
- Use bullet points or lists
- Be overly formal
- Repeat metadata verbatim (paraphrase naturally)
- Say "I have X briefings to share" - just share them naturally

Example input:
Briefing: Your Amazon package was delivered at 2pm
Instructions: mention casually

Example output:
Hey! Quick heads up - your Amazon package arrived this afternoon. Anything else I can help with?"""
        }

load_dotenv()


class BriefingProcessor:
    """
    Generates conversation openers from briefings using LLM.
    
    Call this after adding briefings to pre-generate what the assistant will say
    on wake word, eliminating LLM latency at conversation start.
    
    Configuration is loaded from BRIEFING_PROCESSOR_CONFIG in config.py.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with OpenAI client.
        
        Args:
            config: Optional config override. If None, uses BRIEFING_PROCESSOR_CONFIG from config.py.
        """
        self._client: Optional[AsyncOpenAI] = None
        
        # Load config (allow override for testing)
        self._config = config or BRIEFING_PROCESSOR_CONFIG
        
        # Extract config values
        self._model = self._config.get("model", "gpt-4o-mini")
        self._max_tokens_single = self._config.get("max_tokens_single", 150)
        self._max_tokens_combined = self._config.get("max_tokens_combined", 200)
        self._temperature = self._config.get("temperature", 0.7)
        self._system_prompt = self._config.get("system_prompt", "")
        
        if not OPENAI_AVAILABLE:
            print("⚠️  BriefingProcessor: openai package not installed")
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  BriefingProcessor: OPENAI_API_KEY not set")
            return
        
        self._client = AsyncOpenAI(api_key=api_key)
    
    def is_available(self) -> bool:
        """Check if processor is available."""
        return self._client is not None
    
    async def generate_opener(
        self,
        briefing_content: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a conversation opener from a single briefing.
        
        Args:
            briefing_content: Dict with 'message' (or 'fact'), optional 'llm_instructions', optional 'meta'
            additional_context: Optional extra context for generation
            
        Returns:
            Generated opener string, or None on failure
        """
        if not self.is_available():
            print("⚠️  BriefingProcessor not available, cannot generate opener")
            return None
        
        # Build the prompt from briefing content
        prompt_parts = []
        
        # Support both 'message' and legacy 'fact' key
        message = briefing_content.get("message", "") or briefing_content.get("fact", "")
        if not message:
            return None
        
        prompt_parts.append(f"Briefing: {message}")
        
        instructions = briefing_content.get("llm_instructions")
        if instructions:
            prompt_parts.append(f"Instructions: {instructions}")
        
        meta = briefing_content.get("meta", {})
        if meta:
            meta_items = []
            if meta.get("timestamp"):
                meta_items.append(f"time: {meta['timestamp']}")
            if meta.get("source"):
                meta_items.append(f"source: {meta['source']}")
            if meta_items:
                prompt_parts.append(f"Context: {', '.join(meta_items)}")
        
        if additional_context:
            prompt_parts.append(f"Additional context: {additional_context}")
        
        user_prompt = "\n".join(prompt_parts)
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self._max_tokens_single,
                temperature=self._temperature
            )
            
            opener = response.choices[0].message.content.strip()
            print(f"✅ BriefingProcessor: Generated opener ({len(opener)} chars)")
            return opener
            
        except Exception as e:
            print(f"❌ BriefingProcessor: Error generating opener - {e}")
            return None
    
    async def generate_combined_opener(
        self,
        briefings: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate a single conversation opener from multiple briefings.
        
        Args:
            briefings: List of briefing records (each with 'content' field)
            
        Returns:
            Generated opener string combining all briefings, or None on failure
        """
        if not self.is_available():
            return None
        
        if not briefings:
            return None
        
        # Build combined prompt
        prompt_parts = []
        
        for i, briefing in enumerate(briefings, 1):
            content = briefing.get("content", {})
            
            # Handle content as string or dict
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {"message": content}
            
            message = content.get("message", "") or content.get("fact", "")
            if not message:
                continue
            
            line = f"Briefing {i}: {message}"
            
            instructions = content.get("llm_instructions")
            if instructions:
                line += f" [Instructions: {instructions}]"
            
            meta = content.get("meta", {})
            if meta.get("source"):
                line += f" [Source: {meta['source']}]"
            
            prompt_parts.append(line)
        
        if not prompt_parts:
            return None
        
        user_prompt = "\n".join(prompt_parts)
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self._max_tokens_combined,
                temperature=self._temperature
            )
            
            opener = response.choices[0].message.content.strip()
            print(f"✅ BriefingProcessor: Generated combined opener for {len(briefings)} briefings ({len(opener)} chars)")
            return opener
            
        except Exception as e:
            print(f"❌ BriefingProcessor: Error generating combined opener - {e}")
            return None
    
    async def process_briefing_and_store_opener(
        self,
        briefing_id: str,
        briefing_content: Dict[str, Any],
        briefing_manager
    ) -> bool:
        """
        Generate opener for a briefing and store it back to the database.
        
        Args:
            briefing_id: ID of the briefing record
            briefing_content: The briefing content dict
            briefing_manager: BriefingManager instance to update the record
            
        Returns:
            True if successful, False otherwise
        """
        opener = await self.generate_opener(briefing_content)
        if not opener:
            return False
        
        return await briefing_manager.update_opener(briefing_id, opener)
    
    async def process_pending_briefings(self, user: str, briefing_manager) -> int:
        """
        Process all pending briefings for a user that don't have openers yet.
        
        Args:
            user: User ID
            briefing_manager: BriefingManager instance
            
        Returns:
            Number of briefings processed
        """
        pending = await briefing_manager.get_pending_briefings_without_opener(user)
        
        if not pending:
            return 0
        
        processed = 0
        for briefing in pending:
            briefing_id = briefing.get("id")
            content = briefing.get("content", {})
            
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {"message": content}
            
            success = await self.process_briefing_and_store_opener(
                briefing_id, content, briefing_manager
            )
            if success:
                processed += 1
        
        print(f"📝 BriefingProcessor: Processed {processed}/{len(pending)} briefings for {user}")
        return processed


# Convenience function for quick opener generation
async def generate_briefing_opener(briefing_content: Dict[str, Any]) -> Optional[str]:
    """
    Quick helper to generate an opener from a briefing.
    
    Args:
        briefing_content: Dict with 'message', optional 'llm_instructions', optional 'meta'
        
    Returns:
        Generated opener string
    """
    processor = BriefingProcessor()
    return await processor.generate_opener(briefing_content)
