#!/usr/bin/env python3
"""
Test script for model providers.
Tests both OpenAI and Gemini providers with sample conversations.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.model_providers import create_provider

load_dotenv()

def test_openai_provider():
    """Test OpenAI provider."""
    print("🤖 Testing OpenAI Provider...")
    
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        print("❌ OPENAI_KEY not found in environment")
        return False
    
    try:
        provider = create_provider("openai", api_key=api_key, model="gpt-3.5-turbo")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and tell me your name."}
        ]
        
        response = provider.chat_completion(messages, max_tokens=50)
        print(f"✅ OpenAI Response: {response['content']}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return False

def test_gemini_provider():
    """Test Gemini provider."""
    print("\n🤖 Testing Gemini Provider...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not found or not set in environment")
        print("💡 Get your API key from: https://aistudio.google.com/app/apikey")
        return False
    
    try:
        provider = create_provider("gemini", api_key=api_key, model="gemini-1.5-flash")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and tell me your name."}
        ]
        
        response = provider.chat_completion(messages, max_tokens=50)
        print(f"✅ Gemini Response: {response['content']}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return False

def main():
    """Run provider tests."""
    print("🧪 Testing Model Providers\n")
    
    openai_success = test_openai_provider()
    gemini_success = test_gemini_provider()
    
    print(f"\n📊 Test Results:")
    print(f"   OpenAI: {'✅ PASS' if openai_success else '❌ FAIL'}")
    print(f"   Gemini: {'✅ PASS' if gemini_success else '❌ FAIL'}")
    
    if openai_success or gemini_success:
        print("\n✅ At least one provider is working! You can now use the chatbot.")
        if gemini_success:
            print("💡 Gemini is configured - the chatbot will use Gemini Flash by default.")
        else:
            print("💡 To use Gemini, set CHAT_PROVIDER='gemini' in config.py and add your GEMINI_API_KEY to .env")
    else:
        print("\n❌ No providers are working. Please check your API keys.")

if __name__ == "__main__":
    main()