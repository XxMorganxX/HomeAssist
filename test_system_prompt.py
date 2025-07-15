#!/usr/bin/env python3
"""Test script to verify improved system prompt adherence."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.speech_services_realtime import SpeechServices, ConversationManager
import config
from dotenv import load_dotenv

load_dotenv()

def test_system_prompt_adherence():
    """Test if the Realtime API follows the new system prompt better."""
    
    print("=" * 60)
    print("🧪 Testing System Prompt Adherence")
    print("=" * 60)
    
    # Initialize services
    openai_key = os.getenv("OPENAI_KEY")
    if not openai_key:
        print("❌ OPENAI_KEY not found in environment")
        return
    
    speech_services = SpeechServices(
        openai_api_key=openai_key,
        chat_model="gpt-4o-realtime-preview",
        tts_enabled=False
    )
    
    conversation = ConversationManager(config.SYSTEM_PROMPT)
    
    # Test queries that commonly trigger bad responses
    test_queries = [
        "What time is it?",
        "Turn on the living room lights",
        "What's my schedule today?",
        "Play some music"
    ]
    
    print(f"Temperature: {config.RESPONSE_TEMPERATURE}")
    print(f"System Prompt Length: {len(config.SYSTEM_PROMPT)} chars")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\n🎤 User: {query}")
        
        conversation.add_user_message(query)
        
        # Get response
        response = speech_services.chat_completion(
            messages=conversation.get_messages(),
            max_tokens=config.MAX_TOKENS,
            temperature=config.RESPONSE_TEMPERATURE
        )
        
        if response and response.get("content"):
            response_text = response["content"]
            conversation.add_assistant_message(response_text)
            print(f"🤖 Assistant: {response_text}")
            
            # Check for bad patterns
            bad_patterns = [
                "feel free", "let me know", "anything else", 
                "can I help", "need help", "?", "would you like"
            ]
            
            violations = [p for p in bad_patterns if p.lower() in response_text.lower()]
            
            if violations:
                print(f"❌ Violations found: {violations}")
            else:
                print("✅ Response follows guidelines!")
        else:
            print("❌ No response received")
    
    print("\n" + "=" * 60)
    print("✅ Test completed")
    print("=" * 60)

if __name__ == "__main__":
    test_system_prompt_adherence()