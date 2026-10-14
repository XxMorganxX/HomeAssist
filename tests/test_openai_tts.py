#!/usr/bin/env python3
"""
Test script for the OpenAI TTS provider with streaming support.
"""

import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_framework.providers.tts.openai_tts import OpenAITTSProvider
from assistant_framework.models.data_models import AudioFormat


async def test_openai_tts():
    """Test the OpenAI TTS provider."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    print("🎤 Testing OpenAI TTS Provider...")
    print(f"   API Key: {'*' * 10}{api_key[-4:]}")
    
    config = {
        'api_key': api_key,
        'model': 'gpt-4o-mini-tts',
        'voice': 'alloy',
        'speed': 1.0,
        'response_format': 'mp3'
    }
    
    try:
        print("\n1️⃣  Initializing provider...")
        provider = OpenAITTSProvider(config)
        
        success = await provider.initialize()
        if not success:
            print("❌ Failed to initialize provider")
            return False
        print("✅ Provider initialized")
        
        # Test non-streaming synthesis
        print("\n2️⃣  Testing non-streaming synthesis...")
        test_text = "Hello! This is a test of the OpenAI text to speech provider."
        
        start = time.time()
        audio = await provider.synthesize(test_text)
        elapsed = time.time() - start
        
        print(f"✅ Synthesis successful ({elapsed:.2f}s)")
        print(f"   Format: {audio.format}")
        print(f"   Sample Rate: {audio.sample_rate} Hz")
        print(f"   Audio Size: {audio.get_size_mb():.2f} MB")
        print(f"   Voice: {audio.voice}")
        print(f"   Model: {audio.metadata.get('model')}")
        
        # Test streaming synthesis
        print("\n3️⃣  Testing streaming synthesis...")
        test_text_long = """
        This is a longer test of the streaming synthesis feature. 
        With streaming, audio playback can begin as soon as the first chunks arrive,
        rather than waiting for the entire response. This significantly reduces
        perceived latency, especially for longer messages like this one.
        """
        
        start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_bytes = 0
        
        async for chunk in provider.stream_synthesize(test_text_long.strip()):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"   First chunk received in {(first_chunk_time - start)*1000:.0f}ms")
            chunk_count += 1
            total_bytes += len(chunk.audio_data)
        
        elapsed = time.time() - start
        print(f"✅ Streaming synthesis complete ({elapsed:.2f}s)")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Total bytes: {total_bytes / 1024:.1f} KB")
        print(f"   Time to first chunk: {(first_chunk_time - start)*1000:.0f}ms")
        
        # Test capabilities
        print("\n4️⃣  Provider capabilities:")
        caps = provider.capabilities
        print(f"   Streaming: {caps['streaming']}")
        print(f"   Batch: {caps['batch']}")
        print(f"   Voices: {len(caps['voices'])} available")
        print(f"   Speed range: {caps['speed_range']}")
        print(f"   Models: {caps['models']}")
        print(f"   Features: {', '.join(caps['features'])}")
        
        # Cleanup
        print("\n5️⃣  Cleaning up...")
        await provider.cleanup()
        print("✅ Cleanup complete")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_playback():
    """Test streaming playback functionality."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print("\n🔊 Testing streaming playback...")
    print("   This test requires ffplay for true streaming.")
    print("   If ffplay is not installed, it will fallback to buffered playback.\n")
    
    config = {
        'api_key': api_key,
        'model': 'gpt-4o-mini-tts',
        'voice': 'nova',
        'speed': 1.4,
        'response_format': 'mp3'
    }
    
    try:
        provider = OpenAITTSProvider(config)
        await provider.initialize()
        
        # Test streaming playback
        text = """
        This is a test of streaming audio playback. 
        With true streaming, you should hear audio start playing almost immediately,
        rather than waiting for the full response to download first.
        This dramatically reduces perceived latency for a more responsive experience.
        """
        
        print(f"📝 Text length: {len(text.strip())} chars")
        print("🎵 Starting streaming playback...\n")
        
        start = time.time()
        await provider.synthesize_and_play_streaming(text.strip())
        elapsed = time.time() - start
        
        print(f"\n✅ Streaming playback complete ({elapsed:.2f}s total)")
        
        await provider.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Streaming playback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_latency_comparison():
    """Compare latency between streaming and non-streaming modes."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print("\n⏱️  Latency comparison test...")
    
    config = {
        'api_key': api_key,
        'model': 'gpt-4o-mini-tts',
        'voice': 'alloy',
        'speed': 1.0,
        'response_format': 'mp3'
    }
    
    test_text = "This is a latency comparison test between streaming and non-streaming modes."
    
    try:
        provider = OpenAITTSProvider(config)
        await provider.initialize()
        
        # Non-streaming (time to first playable audio)
        print("\n📊 Non-streaming mode:")
        start = time.time()
        audio = await provider.synthesize(test_text)
        non_stream_time = time.time() - start
        print(f"   Time until audio ready: {non_stream_time*1000:.0f}ms")
        print(f"   Audio size: {len(audio.audio_data) / 1024:.1f} KB")
        
        # Streaming (time to first chunk)
        print("\n📊 Streaming mode:")
        start = time.time()
        first_chunk_time = None
        
        async for chunk in provider.stream_synthesize(test_text):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
                break  # Just measure first chunk
        
        print(f"   Time to first chunk: {first_chunk_time*1000:.0f}ms")
        
        # Calculate improvement
        if first_chunk_time and non_stream_time:
            improvement = ((non_stream_time - first_chunk_time) / non_stream_time) * 100
            print(f"\n🚀 Streaming reduces perceived latency by {improvement:.0f}%")
            print(f"   Non-streaming: {non_stream_time*1000:.0f}ms")
            print(f"   Streaming first chunk: {first_chunk_time*1000:.0f}ms")
        
        await provider.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Latency comparison failed: {e}")
        return False


async def test_voice_samples():
    """Generate samples of all voices for comparison."""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print("\n🎭 Testing all voices...")
    
    config = {
        'api_key': api_key,
        'model': 'gpt-4o-mini-tts',
        'voice': 'alloy',
        'speed': 1.0,
        'response_format': 'mp3'
    }
    
    try:
        provider = OpenAITTSProvider(config)
        await provider.initialize()
        
        voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        
        for voice in voices:
            print(f"\n🎤 Playing voice: {voice}")
            await provider.synthesize_and_play_streaming(
                f"Hi, I'm {voice}. This is how I sound.",
                voice=voice
            )
            await asyncio.sleep(0.5)  # Brief pause between voices
        
        await provider.cleanup()
        print("\n✅ Voice samples complete")
        return True
        
    except Exception as e:
        print(f"❌ Voice samples test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenAI TTS provider")
    parser.add_argument('--playback', action='store_true', help='Test streaming playback')
    parser.add_argument('--latency', action='store_true', help='Compare streaming vs non-streaming latency')
    parser.add_argument('--voices', action='store_true', help='Play samples of all voices')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    args = parser.parse_args()
    
    async def main():
        if args.all:
            await test_openai_tts()
            await test_latency_comparison()
            await test_streaming_playback()
            await test_voice_samples()
        elif args.playback:
            await test_streaming_playback()
        elif args.latency:
            await test_latency_comparison()
        elif args.voices:
            await test_voice_samples()
        else:
            await test_openai_tts()
    
    asyncio.run(main())
