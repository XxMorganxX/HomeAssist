#!/usr/bin/env python3
"""Test script to verify the sliding window + summary context system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.speech_services_realtime import SpeechServices, ConversationManager
from core.context_manager import ContextManager
import config
from dotenv import load_dotenv
import json

load_dotenv()

def test_sliding_window_context():
    """Test the sliding window + summary context system."""
    
    print("=" * 70)
    print("🧪 Testing Sliding Window + Summary Context System")
    print("=" * 70)
    
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
    
    print(f"📋 Configuration:")
    print(f"   REALTIME_USE_SUMMARY_CONTEXT: {getattr(config, 'REALTIME_USE_SUMMARY_CONTEXT', False)}")
    print(f"   CONTEXT_SUMMARY_MIN_MESSAGES: {getattr(config, 'CONTEXT_SUMMARY_MIN_MESSAGES', 5)}")
    print(f"   REALTIME_SLIDING_WINDOW_SIZE: {getattr(config, 'REALTIME_SLIDING_WINDOW_SIZE', 6)}")
    print(f"   REALTIME_SUMMARY_AS_SYSTEM_MESSAGE: {getattr(config, 'REALTIME_SUMMARY_AS_SYSTEM_MESSAGE', True)}")
    
    # Test with a long conversation to trigger summary context
    test_exchanges = [
        ("What time is it?", "It's 2:30 PM."),
        ("Turn on the living room lights", "Living room lights are now on."),
        ("What's my schedule today?", "You have a meeting at 3 PM with Sarah."),
        ("Play some jazz music", "Playing jazz playlist on Spotify."),
        ("Turn off the bedroom lights", "Bedroom lights are now off."),
        ("What's the weather like?", "It's sunny and 72°F outside."),
        ("Set a timer for 10 minutes", "Timer set for 10 minutes."),
        ("Turn on kitchen lights", "Kitchen lights are now on."),
        ("What's for dinner tonight?", "You have lasagna planned for dinner."),
        ("Play relaxing music", "Playing relaxing playlist."),  # This should trigger summary context
    ]
    
    print(f"\n📝 Adding {len(test_exchanges)} conversation exchanges...")
    
    # Add all the test exchanges
    for i, (user_msg, assistant_msg) in enumerate(test_exchanges):
        print(f"\n--- Exchange {i+1} ---")
        print(f"🎤 User: {user_msg}")
        
        conversation.add_user_message(user_msg)
        conversation.add_assistant_message(assistant_msg)
        
        print(f"🤖 Assistant: {assistant_msg}")
        
        # Show current conversation length
        current_length = len(conversation.get_messages())
        print(f"📊 Total messages: {current_length}")
        
        # Test the context creation after we have enough messages
        if current_length >= config.CONTEXT_SUMMARY_MIN_MESSAGES + 1:
            print(f"\n🔍 Testing context creation with {current_length} messages...")
            
            # Test the _create_summary_context method
            original_messages = conversation.get_messages()
            summary_context = speech_services._create_summary_context(original_messages, config.SYSTEM_PROMPT)
            
            print(f"📋 Original context: {len(original_messages)} messages")
            print(f"📋 Summary context: {len(summary_context)} messages")
            
            # Show the structure of the summary context
            print(f"\n📄 Summary Context Structure:")
            for j, msg in enumerate(summary_context):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', '')
                print(f"   {j+1}. {role}: {content}")
            
            # Check that we have the expected structure
            if len(summary_context) >= 2:
                first_msg = summary_context[0]
                second_msg = summary_context[1]
                
                first_content = first_msg.get('content', '')
                if first_msg.get('role') == 'system' and isinstance(first_content, str) and 'voice-based home assistant' in first_content.lower():
                    print("✅ First message is system prompt")
                else:
                    print("❌ First message is not system prompt")
                
                if second_msg.get('role') == 'system' and 'CONVERSATION SUMMARY' in second_msg.get('content', ''):
                    print("✅ Second message is conversation summary")
                    print(f"📝 Summary content: {second_msg.get('content', '')[:200]}...")
                else:
                    print("❌ Second message is not conversation summary")
            
            # Test with the Realtime API if we're at the final exchange
            if i == len(test_exchanges) - 1:
                print(f"\n🚀 Testing with Realtime API...")
                test_response = speech_services.chat_completion(
                    messages=summary_context,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.RESPONSE_TEMPERATURE
                )
                
                if test_response and test_response.get("content"):
                    print(f"✅ Realtime API response: {test_response['content']}")
                else:
                    print("❌ No response from Realtime API")
    
    # Test edge cases
    print(f"\n🧪 Testing Edge Cases:")
    
    # Test with empty summary
    print("1. Testing with empty summary file...")
    try:
        with open(config.session_summary_file, 'w') as f:
            json.dump({"conversation": {}}, f)
        
        empty_context = speech_services._create_summary_context(conversation.get_messages(), config.SYSTEM_PROMPT)
        print(f"   Empty summary handled: {len(empty_context)} messages in context")
    except Exception as e:
        print(f"   ❌ Error with empty summary: {e}")
    
    # Test with short conversation
    print("2. Testing with short conversation...")
    short_conversation = ConversationManager(config.SYSTEM_PROMPT)
    short_conversation.add_user_message("Hello")
    short_conversation.add_assistant_message("Hi there!")
    
    short_context = speech_services._create_summary_context(short_conversation.get_messages(), config.SYSTEM_PROMPT)
    # With CONTEXT_SUMMARY_MIN_MESSAGES = 1, even a 2-message conversation gets summary treatment
    # Expected: system prompt + summary + 2 original messages = 4 total
    expected_length = 4  # system + summary + 2 messages
    
    if len(short_context) == expected_length:
        print(f"✅ Short conversation handled correctly: {len(short_context)} messages")
    else:
        print(f"❌ Short conversation issue: expected {expected_length}, got {len(short_context)}")
    
    print("\n" + "=" * 70)
    print("✅ Sliding Window + Summary Context Test Completed")
    print("=" * 70)

if __name__ == "__main__":
    test_sliding_window_context()