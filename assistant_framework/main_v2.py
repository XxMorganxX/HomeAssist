"""
Simplified CLI for refactored assistant framework (v2).
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from .orchestrator_v2 import RefactoredOrchestrator
    from .config import get_framework_config, print_config_summary
    from .utils.logging_config import setup_logging
except ImportError:
    from assistant_framework.orchestrator_v2 import RefactoredOrchestrator
    from assistant_framework.config import get_framework_config, print_config_summary
    from assistant_framework.utils.logging_config import setup_logging


# =============================================================================
# FIRST-TIME SETUP
# =============================================================================

def _get_state_file_path() -> Path:
    """Get path to app_state.json."""
    return Path(__file__).parent.parent / "state_management" / "app_state.json"


def _run_first_time_setup() -> dict:
    """
    Interactive first-time setup for new users.
    Called from main process (has stdin access).
    """
    print("\n" + "=" * 60)
    print("🏠 HOMEASSIST FIRST-TIME SETUP")
    print("=" * 60)
    print("\nWelcome! Let's configure your assistant.\n")
    
    # =========================================================================
    # PRIMARY USER
    # =========================================================================
    while True:
        primary_user = input("👤 Enter your name (primary user): ").strip()
        if primary_user:
            break
        print("   Name cannot be empty. Please try again.")
    
    # =========================================================================
    # NICKNAMES (optional)
    # =========================================================================
    print(f"\n🏷️  Nicknames & Titles (optional)")
    print("   What else can Sol call you? (e.g., Mr. Stuart, boss, chief)")
    print("   Enter nicknames separated by commas, or press Enter to skip.")
    nicknames_input = input("   Nicknames: ").strip()
    nicknames = []
    if nicknames_input:
        nicknames = [name.strip() for name in nicknames_input.split(",") if name.strip()]
    
    # =========================================================================
    # HOUSEHOLD MEMBERS (optional)
    # =========================================================================
    print(f"\n👥 Household Members (optional)")
    print("   Add other people who will use this assistant.")
    print("   Enter names separated by commas, or press Enter to skip.")
    household_input = input("   Household members: ").strip()
    household_members = []
    if household_input:
        household_members = [name.strip() for name in household_input.split(",") if name.strip()]
    
    # =========================================================================
    # INTEGRATIONS
    # =========================================================================
    integrations = {}
    
    # Spotify
    print(f"\n📻 Spotify Integration")
    spotify_enabled = input("   Enable Spotify? [Y/n]: ").strip().lower() != 'n'
    if spotify_enabled:
        spotify_user = input(f"   Spotify username [{primary_user}]: ").strip()
        if not spotify_user:
            spotify_user = primary_user
        integrations["spotify"] = {
            "enabled": True,
            "username": spotify_user
        }
    else:
        spotify_user = primary_user
        integrations["spotify"] = {"enabled": False}
    
    # Calendar
    print(f"\n📅 Google Calendar Integration")
    calendar_enabled = input("   Enable Google Calendar? [Y/n]: ").strip().lower() != 'n'
    if calendar_enabled:
        integrations["calendar"] = {
            "enabled": True,
            "default_calendar": "primary"
        }
    else:
        integrations["calendar"] = {"enabled": False}
    
    # Smart Home
    print(f"\n💡 Smart Home Integration")
    smart_home_enabled = input("   Enable smart home controls? [Y/n]: ").strip().lower() != 'n'
    if smart_home_enabled:
        integrations["smart_home"] = {"enabled": True}
    else:
        integrations["smart_home"] = {"enabled": False}
    
    # =========================================================================
    # DEFAULT PREFERENCES
    # =========================================================================
    print(f"\n⚙️  Default Preferences")
    
    # Lighting scene
    print("   Lighting scene options: mood, party, movie, all_on, all_off")
    lighting_scene = input("   Default lighting scene [all_on]: ").strip().lower()
    if lighting_scene not in ["mood", "party", "movie", "all_on", "all_off"]:
        lighting_scene = "all_on"
    
    # Volume
    volume_input = input("   Default volume (0-100) [50]: ").strip()
    try:
        volume_level = int(volume_input) if volume_input else 50
        volume_level = max(0, min(100, volume_level))
    except ValueError:
        volume_level = 50
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"✅ Setup complete for {primary_user}!")
    if household_members:
        print(f"   Household: {', '.join(household_members)}")
    enabled_integrations = [k for k, v in integrations.items() if v.get("enabled")]
    if enabled_integrations:
        print(f"   Integrations: {', '.join(enabled_integrations)}")
    print("=" * 60 + "\n")
    
    # Build initial state structure
    return {
        "user_state": {
            "primary_user": primary_user,
            "nicknames": nicknames,
            "household_members": household_members,
            "created_at": datetime.now().isoformat(),
            "integrations": integrations
        },
        "chat_controlled_state": {
            "current_spotify_user": spotify_user,
            "lighting_scene": lighting_scene,
            "volume_level": str(volume_level),
            "do_not_disturb": "false"
        },
        "autonomous_state": {
            "notification_queue": {
                primary_user: {
                    "notifications": [],
                    "emails": []
                }
            }
        }
    }


def ensure_first_time_setup():
    """
    Check if first-time setup is needed and run it if so.
    Must be called from main process BEFORE any subprocesses start.
    """
    state_file = _get_state_file_path()
    
    if state_file.exists():
        return  # Already configured
    
    # Check if we can run interactively
    if not sys.stdin.isatty():
        print("⚠️  First-time setup required but running non-interactively.")
        print("   Run the assistant in a terminal to complete setup.")
        sys.exit(1)
    
    # Ensure directory exists
    state_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run interactive setup
    state = _run_first_time_setup()
    
    # Save state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"📝 Configuration saved to {state_file}\n")


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

    # Check for first-time setup BEFORE anything else
    # This must happen in main process (has stdin access)
    ensure_first_time_setup()

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

