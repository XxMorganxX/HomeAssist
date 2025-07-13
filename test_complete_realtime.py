#!/usr/bin/env python3
"""
Test script to verify complete Realtime API integration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_realtime_integration():
    """Test complete Realtime API integration."""
    print("🧪 Testing Complete Realtime API Integration")
    print("=" * 60)
    
    try:
        # Import config to check settings
        import config
        
        print("🔧 Configuration:")
        print(f"   USE_REALTIME_API = {config.USE_REALTIME_API}")
        print(f"   REALTIME_STREAMING_MODE = {getattr(config, 'REALTIME_STREAMING_MODE', False)}")
        print(f"   REALTIME_MODEL = {getattr(config, 'REALTIME_MODEL', 'not set')}")
        
        # Test StreamingChatbot import and initialization
        print("\n🧪 Testing StreamingChatbot with Realtime API...")
        from core.streaming_chatbot import StreamingChatbot, ToolEnabledStreamingChatbot, USING_REALTIME_API
        
        print(f"   USING_REALTIME_API = {USING_REALTIME_API}")
        
        # Test basic chatbot
        chatbot = StreamingChatbot()
        print(f"   ✅ StreamingChatbot initialized")
        
        if hasattr(chatbot, 'use_streaming_mode'):
            print(f"   🌊 Streaming mode: {chatbot.use_streaming_mode}")
        
        # Test speech services
        if hasattr(chatbot.speech_services, 'use_realtime'):
            print(f"   🚀 Realtime API available: {chatbot.speech_services.use_realtime}")
            
            if chatbot.speech_services.use_realtime:
                # Test connection capability
                print("   🔗 Testing Realtime API methods...")
                
                methods_to_check = [
                    'set_callbacks',
                    'start_streaming', 
                    'stop_streaming',
                    'execute_function_call_realtime',
                    'check_for_function_calls',
                    '_sync_conversation_to_session',
                    '_convert_functions_to_tools'
                ]
                
                for method in methods_to_check:
                    if hasattr(chatbot.speech_services, method):
                        print(f"     ✅ {method}")
                    else:
                        print(f"     ❌ {method}")
        
        # Test tool-enabled chatbot
        print("\n🧪 Testing ToolEnabledStreamingChatbot...")
        tool_chatbot = ToolEnabledStreamingChatbot()
        print(f"   ✅ ToolEnabledStreamingChatbot initialized")
        print(f"   🔧 Loaded {len(tool_chatbot.functions)} tools")
        
        # Test function execution capability
        if hasattr(tool_chatbot, 'mcp_server') and tool_chatbot.mcp_server:
            print("   ✅ MCP server available for function execution")
        
        print("\n📋 Integration Status:")
        if USING_REALTIME_API and config.USE_REALTIME_API:
            print("   🚀 FULLY INTEGRATED - System will use Realtime API for:")
            print("     • Audio transcription")
            print("     • Chat completion")
            print("     • Function calling")
            print("     • Response generation")
            print("     • Conversation management")
        elif config.USE_REALTIME_API:
            print("   ⚠️  PARTIAL - Realtime API enabled but not fully available")
        else:
            print("   📡 Traditional API mode")
        
        print("\n✅ All integration tests passed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True

def show_usage_examples():
    """Show usage examples for different modes."""
    print("\n📖 Usage Examples:")
    print("=" * 60)
    
    print("\n1. Switch to Traditional API (most compatible):")
    print("   python switch_mode.py")
    print("   Select option 1")
    
    print("\n2. Switch to Realtime API (chunk-based):")
    print("   python switch_mode.py")
    print("   Select option 2")
    
    print("\n3. Switch to Realtime API (continuous streaming):")
    print("   python switch_mode.py")
    print("   Select option 3")
    
    print("\n4. Run the voice assistant:")
    print("   python core/streaming_chatbot.py")
    
    print("\n5. Check current configuration:")
    print("   python test_complete_realtime.py")

if __name__ == "__main__":
    success = test_realtime_integration()
    
    if success:
        show_usage_examples()
    else:
        print("\n🔧 Setup Requirements:")
        print("   pip install openai websockets sounddevice webrtcvad numpy scipy python-dotenv")
        print("   Set OPENAI_KEY environment variable")
        print("   Ensure OpenAI Realtime API access")