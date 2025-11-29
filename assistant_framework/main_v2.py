"""
Simplified CLI for refactored assistant framework (v2).
"""

import asyncio
import argparse
from pathlib import Path

try:
    from .orchestrator_v2 import RefactoredOrchestrator
    from .config import get_framework_config, print_config_summary
    from .utils.logging_config import setup_logging
except ImportError:
    from assistant_framework.orchestrator_v2 import RefactoredOrchestrator
    from assistant_framework.config import get_framework_config, print_config_summary
    from assistant_framework.utils.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant Framework (v2 - Refactored)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single conversation
    single = subparsers.add_parser(
        'single',
        help='Run a single conversation (wake word -> transcribe -> respond -> speak)'
    )
    
    # Continuous loop
    continuous = subparsers.add_parser(
        'continuous',
        help='Run continuous conversation loop'
    )
    
    # Wake word only
    wakeword = subparsers.add_parser(
        'wakeword',
        help='Test wake word detection only'
    )
    
    # Transcription only
    transcribe = subparsers.add_parser(
        'transcribe',
        help='Test transcription only'
    )
    
    # Status
    status = subparsers.add_parser(
        'status',
        help='Show orchestrator status'
    )
    
    # Config
    config = subparsers.add_parser(
        'config',
        help='Show configuration'
    )
    
    return parser


async def cmd_single(orchestrator: RefactoredOrchestrator):
    """Run a single conversation."""
    print("\n" + "="*60)
    print("Single Conversation Mode")
    print("="*60 + "\n")
    
    await orchestrator.run_full_conversation()


async def cmd_continuous(orchestrator: RefactoredOrchestrator):
    """Run continuous conversation loop."""
    print("\n" + "="*60)
    print("Continuous Conversation Mode")
    print("="*60 + "\n")
    
    await orchestrator.run_continuous_loop()


async def cmd_wakeword(orchestrator: RefactoredOrchestrator):
    """Test wake word detection."""
    print("\n" + "="*60)
    print("Wake Word Detection Test")
    print("="*60 + "\n")
    
    print("Listening for wake word... (Press Ctrl+C to stop)\n")
    
    try:
        async for event in orchestrator.run_wake_word_detection():
            print(f"\n🔔 Detected: {event.model_name} (score: {event.score:.3f})")
            print("   Continuing to listen...\n")
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")


async def cmd_transcribe(orchestrator: RefactoredOrchestrator):
    """Test transcription."""
    print("\n" + "="*60)
    print("Transcription Test")
    print("="*60 + "\n")
    
    print("Starting transcription... (speak now)\n")
    
    text = await orchestrator.run_transcription()
    
    if text:
        print(f"\n✅ Final transcription: {text}\n")
    else:
        print("\n⚠️  No transcription received\n")


async def cmd_status(orchestrator: RefactoredOrchestrator):
    """Show orchestrator status."""
    print("\n" + "="*60)
    print("Orchestrator Status")
    print("="*60 + "\n")
    
    status = orchestrator.get_status()
    
    print(f"Initialized: {status['initialized']}")
    print(f"\nState Machine:")
    for key, value in status['state'].items():
        print(f"  {key}: {value}")
    
    print(f"\nProviders:")
    for key, value in status['providers'].items():
        status_str = "✅ Loaded" if value else "⚠️  Not loaded"
        print(f"  {key}: {status_str}")
    
    print(f"\nErrors:")
    for key, value in status['errors'].items():
        print(f"  {key}: {value}")
    
    print()


def cmd_config():
    """Show configuration."""
    print("\n" + "="*60)
    print("Configuration")
    print("="*60 + "\n")
    
    config = get_framework_config()
    print_config_summary()


async def async_main():
    """Async main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle config command (doesn't need orchestrator)
    if args.command == 'config':
        cmd_config()
        return
    
    # Build configuration
    print("📝 Loading configuration...")
    config = get_framework_config()
    
    # Create orchestrator
    print("🚀 Creating orchestrator...")
    orchestrator = RefactoredOrchestrator(config)
    
    # Determine if MCP server is needed for this command
    commands_needing_mcp = {'single', 'continuous'}
    start_mcp = args.command in commands_needing_mcp
    
    # Initialize orchestrator (starts MCP early if needed)
    if not await orchestrator.initialize(start_mcp=start_mcp):
        print("❌ Failed to initialize orchestrator")
        return
    
    try:
        # Route to command handler
        if args.command == 'single':
            await cmd_single(orchestrator)
        elif args.command == 'continuous':
            await cmd_continuous(orchestrator)
        elif args.command == 'wakeword':
            await cmd_wakeword(orchestrator)
        elif args.command == 'transcribe':
            await cmd_transcribe(orchestrator)
        elif args.command == 'status':
            await cmd_status(orchestrator)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    finally:
        # Cleanup
        await orchestrator.cleanup()


def main():
    """Synchronous entry point."""
    # Setup logging
    import os
    log_level = os.getenv("LOG_LEVEL", "INFO")
    try:
        setup_logging(level=log_level)
    except Exception as e:
        print(f"⚠️  Failed to setup logging: {e}")

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

