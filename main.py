#!/usr/bin/env python3
"""
RasPi Smart Home Voice Assistant - Main Entry Point

Unified entry point for the voice assistant system with configurable components.
Supports multiple operational modes: wake_word, continuous, and interactive.
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import config
    from core.components import ComponentOrchestrator
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class VoiceAssistantApp:
    """Main application class for the RasPi Smart Home Voice Assistant."""
    
    def __init__(self, config_overrides: Optional[dict] = None):
        """
        Initialize the voice assistant application.
        
        Args:
            config_overrides: Optional configuration overrides
        """
        self.orchestrator: Optional[ComponentOrchestrator] = None
        self.running = False
        

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🏠 RasPi Smart Home Voice Assistant")
        print("=" * 50)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown()
    
    def _print_system_info(self):
        """Print system configuration and status."""
        print("\n📋 System Configuration:")
        print(f"   Mode: {config.CONVERSATION_MODE}")
        print(f"   Wake Word: {'✓' if config.WAKE_WORD_ENABLED else '✗'}")
        print(f"   TTS: {'✓' if config.TTS_ENABLED else '✗'}")
        print(f"   AEC: {'✓' if config.AEC_ENABLED else '✗'}")
        print(f"   Debug Mode: {'✓' if config.DEBUG_MODE else '✗'}")
        print(f"   Auto Restart: {'✓' if config.AUTO_RESTART else '✗'}")
        
        print(f"\n🤖 AI Configuration:")
        print(f"   Chat Provider: {config.CHAT_PROVIDER}")
        if config.CHAT_PROVIDER == "openai":
            print(f"   OpenAI Model: {config.OPENAI_CHAT_MODEL}")
            print(f"   TTS Voice: {config.TTS_VOICE}")
        elif config.CHAT_PROVIDER == "gemini":
            print(f"   Gemini Model: {config.GEMINI_CHAT_MODEL}")
        
        print(f"\n🎤 Audio Configuration:")
        print(f"   Sample Rate: {config.SAMPLE_RATE} Hz")
        print(f"   VAD Mode: {config.VAD_MODE}")
        print(f"   Frame Size: {config.FRAME_MS} ms")
    
    def _check_dependencies(self) -> bool:
        """Check for required dependencies and API keys."""
        missing_deps = []
        missing_keys = []
        
        # Check for required environment variables
        if config.CHAT_PROVIDER == "openai":
            if not os.getenv("OPENAI_KEY"):
                missing_keys.append("OPENAI_KEY")
        elif config.CHAT_PROVIDER == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                missing_keys.append("GOOGLE_API_KEY")
        
        # Check for required audio directories
        required_dirs = [
            "./audio_data/wake_word_models",
            "./audio_data/opener_audio"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"📁 Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
        
        # Report missing dependencies
        if missing_deps:
            print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        if missing_keys:
            print(f"\n❌ Missing API keys: {', '.join(missing_keys)}")
            print("Set the required environment variables")
            return False
        
        return True
    
    def initialize(self) -> bool:
        """Initialize the voice assistant system."""
        try:
            print("\n🔧 Initializing components...")
            
            # Check dependencies
            if not self._check_dependencies():
                return False
            
            # Create component orchestrator
            self.orchestrator = ComponentOrchestrator()
            
            # Initialize components
            if not self.orchestrator.initialize_components():
                print("❌ Failed to initialize components")
                return False
            
            print("✅ Components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the voice assistant system."""
        try:
            if not self.orchestrator:
                print("❌ System not initialized")
                return False
            
            print("\n🚀 Starting voice assistant...")
            
            # Start all components
            if not self.orchestrator.start_all():
                print("❌ Failed to start components")
                return False
            
            self.running = True
            print("✅ Voice assistant started successfully\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start system: {e}")
            return False
    
    def run_forever(self):
        """Run the voice assistant until interrupted."""
        if not self.running:
            print("❌ System not started")
            return
        
        try:
            # Print usage instructions
            if config.CONVERSATION_MODE == "wake_word":
                print("💡 Say the wake word to start a conversation")
            elif config.CONVERSATION_MODE == "interactive":
                print("💡 Press Enter to start a conversation")
            elif config.CONVERSATION_MODE == "continuous":
                print("💡 Continuous listening mode - speak naturally")
            
            print("💡 Press Ctrl+C to stop the system\n")
            
            # Run the orchestrator
            self.orchestrator.run_forever()
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the voice assistant system."""
        if self.running:
            print("🛑 Stopping voice assistant...")
            
            if self.orchestrator:
                self.orchestrator.stop_all()
            
            self.running = False
            print("✅ Voice assistant stopped")
    
    def get_status(self) -> dict:
        """Get current system status."""
        if not self.orchestrator:
            return {"status": "not_initialized"}
        
        return self.orchestrator.get_system_status()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RasPi Smart Home Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operational Modes:
  wake_word    - Wait for wake word, then start conversation (default)
  continuous   - Always listening, no wake word needed
  interactive  - Manual conversation triggers (press Enter)

Examples:
  python main.py                                    # Use default config
  python main.py --mode continuous                  # Continuous mode
  python main.py --no-wake-word --no-tts          # Disable features
  python main.py --debug --chat-provider gemini    # Debug with Gemini
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["wake_word", "continuous", "interactive"],
        help="Conversation mode (default: from config)"
    )
    
    parser.add_argument(
        "--chat-provider",
        choices=["openai", "gemini"],
        help="AI chat provider (default: from config)"
    )
    
    parser.add_argument(
        "--wake-word", 
        action="store_true", 
        dest="wake_word_enabled",
        help="Enable wake word detection"
    )
    
    parser.add_argument(
        "--no-wake-word", 
        action="store_false", 
        dest="wake_word_enabled",
        help="Disable wake word detection"
    )
    
    parser.add_argument(
        "--tts", 
        action="store_true", 
        dest="tts_enabled",
        help="Enable text-to-speech"
    )
    
    parser.add_argument(
        "--no-tts", 
        action="store_false", 
        dest="tts_enabled",
        help="Disable text-to-speech"
    )
    
    parser.add_argument(
        "--aec", 
        action="store_true", 
        dest="aec_enabled",
        help="Enable acoustic echo cancellation"
    )
    
    parser.add_argument(
        "--no-aec", 
        action="store_false", 
        dest="aec_enabled",
        help="Disable acoustic echo cancellation"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--no-auto-restart", 
        action="store_false", 
        dest="auto_restart",
        help="Disable automatic component restart"
    )
    
    parser.add_argument(
        "--no-startup-sound", 
        action="store_false", 
        dest="startup_sound",
        help="Disable startup sound"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show system status and exit"
    )
    
    return parser.parse_args()


def build_config_overrides(args) -> dict:
    """Build configuration overrides from command line arguments."""
    overrides = {}
    
    # Map command line arguments to config variables
    arg_mapping = {
        'mode': 'CONVERSATION_MODE',
        'chat_provider': 'CHAT_PROVIDER',
        'wake_word_enabled': 'WAKE_WORD_ENABLED',
        'tts_enabled': 'TTS_ENABLED',
        'aec_enabled': 'AEC_ENABLED',
        'debug': 'DEBUG_MODE',
        'auto_restart': 'AUTO_RESTART',
        'startup_sound': 'STARTUP_SOUND'
    }
    
    for arg_name, config_name in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_name] = value
    
    return overrides


def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Build configuration overrides
        config_overrides = build_config_overrides(args)
        
        # Create and initialize the application
        app = VoiceAssistantApp(config_overrides)
        
        # Show system information
        app._print_system_info()
        
        # Handle status request
        if args.status:
            if app.initialize():
                status = app.get_status()
                print(f"\n📊 System Status:")
                print(f"   Running: {status.get('running', False)}")
                print(f"   Mode: {status.get('mode', 'unknown')}")
                print(f"   Components: {len(status.get('components', {}))}")
                for name, comp_status in status.get('components', {}).items():
                    state = comp_status.get('state', 'unknown')
                    healthy = comp_status.get('healthy', False)
                    print(f"     {name}: {state} {'✓' if healthy else '✗'}")
            return
        
        # Initialize the system
        if not app.initialize():
            print("❌ Failed to initialize voice assistant")
            sys.exit(1)
        
        # Start the system
        if not app.start():
            print("❌ Failed to start voice assistant")
            sys.exit(1)
        
        # Run forever
        app.run_forever()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()