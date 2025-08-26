#!/usr/bin/env python3
"""
Example usage of the Assistant Framework.

This script demonstrates how to use the framework for:
1. Full pipeline execution (transcription → response → TTS)
2. Individual component usage
3. Single message processing

Prerequisites:
- Set environment variables: ASSEMBLYAI_API_KEY, OPENAI_API_KEY
- Install dependencies: assemblyai, openai, google-cloud-texttospeech, etc.
- Configure Google Cloud credentials if using Google TTS
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_framework_config
from orchestrator import create_orchestrator


async def example_full_pipeline():
    """Example: Run the full voice assistant pipeline."""
    print("🚀 Full Pipeline Example")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_framework_config()
        
        # Create orchestrator
        orchestrator = await create_orchestrator(config)
        
        # Run full pipeline - this will:
        # 1. Listen for speech input
        # 2. Generate response using OpenAI + MCP tools
        # 3. Convert response to speech and play it
        # 4. Repeat until Ctrl+C
        await orchestrator.run_full_pipeline(
            auto_save_audio=True,
            speech_audio_dir=Path("speech_audio"),
            play_responses=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


async def example_single_message():
    """Example: Process a single message."""
    print("💬 Single Message Example")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_framework_config()
        
        # Create orchestrator
        orchestrator = await create_orchestrator(config)
        
        # Process a single message
        message = "What's the weather like today?"
        print(f"User: {message}")
        
        result = await orchestrator.process_single_message(
            message=message,
            use_context=True,
            generate_tts=True,
            play_audio=True,
            save_path=Path("speech_audio/example_response.mp3")
        )
        
        print(f"Assistant: {result['response']}")
        if result['audio']:
            print(f"Audio saved: {result['audio'].get_size_mb():.2f} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


async def example_individual_components():
    """Example: Use individual components separately."""
    print("🔧 Individual Components Example")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_framework_config()
        
        # Create orchestrator
        orchestrator = await create_orchestrator(config)
        
        # Example 1: TTS only
        print("1. TTS Example:")
        audio = await orchestrator.run_tts_only(
            "Hello Mr. Stuart, this is a test of the text-to-speech system.",
            save_path=Path("speech_audio/tts_test.mp3"),
            play_audio=True
        )
        print(f"   Generated audio: {audio.get_size_mb():.2f} MB")
        
        # Example 2: Response only
        print("\n2. Response Example:")
        message = "Tell me a joke"
        async for chunk in orchestrator.run_response_only(message, use_context=False):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.is_complete:
                print()  # New line
                break
        
        # Example 3: Transcription only (commented out - requires microphone)
        # print("\n3. Transcription Example:")
        # print("   Speak into your microphone...")
        # async for result in orchestrator.run_transcription_only():
        #     print(f"   {'[FINAL]' if result.is_final else '[PARTIAL]'} {result.text}")
        #     if result.is_final:
        #         break
        
        # Example 4: Check status
        print("\n4. Status Check:")
        status = orchestrator.get_status()
        print(f"   Initialized: {status['initialized']}")
        print(f"   Available providers: {list(status['providers'].keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


async def example_context_management():
    """Example: Context management."""
    print("📝 Context Management Example")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_framework_config()
        
        # Create orchestrator
        orchestrator = await create_orchestrator(config)
        
        # Add some messages to context
        if orchestrator.context:
            orchestrator.context.add_message("user", "My name is Morgan Stuart")
            orchestrator.context.add_message("assistant", "Hello Mr. Stuart! Nice to meet you.")
            orchestrator.context.add_message("user", "What's my name?")
            
            # Get context summary
            summary = orchestrator.context.get_summary()
            print(f"Context summary: {summary}")
            
            # Get conversation history
            history = orchestrator.context.get_history()
            print(f"Conversation history ({len(history)} messages):")
            for msg in history:
                print(f"  {msg['role']}: {msg['content']}")
            
            # Test response with context
            print("\nTesting response with context:")
            async for chunk in orchestrator.run_response_only("What's my name?", use_context=True):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print()
                    break
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


def main():
    """Main function with example selection."""
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
    else:
        print("Assistant Framework Examples")
        print("=" * 30)
        print("1. full      - Full voice pipeline")
        print("2. message   - Single message processing")
        print("3. components - Individual components")
        print("4. context   - Context management")
        print()
        example_type = input("Choose an example (1-4): ").strip()
    
    # Map choices to functions
    examples = {
        '1': example_full_pipeline,
        'full': example_full_pipeline,
        '2': example_single_message, 
        'message': example_single_message,
        '3': example_individual_components,
        'components': example_individual_components,
        '4': example_context_management,
        'context': example_context_management,
    }
    
    example_func = examples.get(example_type)
    if not example_func:
        print(f"Invalid example: {example_type}")
        sys.exit(1)
    
    try:
        asyncio.run(example_func())
    except KeyboardInterrupt:
        print("\n👋 Example stopped by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()