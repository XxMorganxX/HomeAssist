#!/usr/bin/env python3
"""
RasPi Smart Home Voice Assistant - Main Entry Point

Unified entry point for the voice assistant system with configurable components.
Supports multiple operational modes: wake_word, continuous, and interactive.
Includes integrated MCP tool server for external tool access.
"""

import os
import sys
import time
import signal
import argparse
import threading
from pathlib import Path
from typing import Optional

# Suppress common warnings from dependencies
from core.suppress_warnings import suppress_common_warnings

from core.state_management.statemanager import StateManager

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import config
    from core.components import ComponentOrchestrator
    from mcp_server.server import MCPServer
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class VoiceAssistantApp:
    """Main application class for the RasPi Smart Home Voice Assistant with integrated MCP server."""
    
    def __init__(self, config_overrides: Optional[dict] = None, enable_mcp_server: bool = True):
        """
        Initialize the voice assistant application.
        
        Args:
            config_overrides: Optional configuration overrides
            enable_mcp_server: Whether to start the integrated MCP server
        """
        self.orchestrator: Optional[ComponentOrchestrator] = None
        self.mcp_server: Optional[MCPServer] = None
        self.mcp_server_thread: Optional[threading.Thread] = None
        self.enable_mcp_server = enable_mcp_server
        self.running = False
        

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🏠 RasPi Smart Home Voice Assistant")
        if enable_mcp_server:
            print("🔧 With Integrated MCP Tool Server")
        print("=" * 50)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        
        # Check if this is the second Ctrl+C (force exit)
        if hasattr(self, '_shutdown_requested'):
            print("🛑 Force exit requested - terminating immediately")
            import os
            os._exit(1)
        
        self._shutdown_requested = True
        
        # Save any active conversations before shutdown
        try:
            self._save_active_conversations()
            self._conversations_saved = True  # Mark as saved to prevent duplicate saving
        except Exception as e:
            print(f"⚠️ Error saving conversations: {e}")
        
        # Always try to shutdown gracefully first
        try:
            self.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        # If we're not in the main running loop, force exit
        if not self.running:
            print("🛑 Force exit")
            import os
            os._exit(1)
    
    def _save_active_conversations(self):
        """Save any active conversations to database before shutdown."""
        print("💾 Checking for active conversations to save...")
        
        try:
            # Check if orchestrator exists
            if not self.orchestrator:
                print("ℹ️ No orchestrator found - system may not have been fully initialized")
                return
            
            # Check if conversation handler exists
            if not hasattr(self.orchestrator, 'conversation_handler'):
                print("ℹ️ Orchestrator has no conversation_handler attribute")
                return
            
            conversation_handler = self.orchestrator.conversation_handler
            if not conversation_handler:
                print("ℹ️ No conversation handler instance found")
                return
            
            # Check if chatbot exists
            if not hasattr(conversation_handler, 'chatbot'):
                print("ℹ️ Conversation handler has no chatbot attribute")
                return
            
            chatbot = conversation_handler.chatbot
            if not chatbot:
                print("ℹ️ No chatbot instance found")
                return
            
            # Check if conversation exists
            if not hasattr(chatbot, 'conversation'):
                print("ℹ️ Chatbot has no conversation attribute")
                return
            
            conversation = chatbot.conversation
            if not conversation:
                print("ℹ️ No conversation instance found")
                return
            
            # Try to get messages (excluding system prompt)
            try:
                messages = conversation.get_chat_minus_sys_prompt()
                if messages and len(messages) > 0:
                    print(f"💾 Saving active voice conversation ({len(messages)} messages)...")
                    chatbot.send_to_db(messages)
                    print("✅ Active voice conversation saved to database")
                else:
                    print("ℹ️ No active voice conversation messages to save")
            except Exception as msg_error:
                print(f"⚠️ Error getting conversation messages: {msg_error}")
            
        except Exception as e:
            print(f"⚠️ Error saving active conversations: {e}")
            import traceback
            if config.DEBUG_MODE:
                traceback.print_exc()
    
    def _print_system_info(self):
        """Print system configuration and status."""
        print("\n📋 System Configuration:")
        print(f"   Mode: {config.CONVERSATION_MODE}")
        print(f"   Wake Word: {'✓' if config.WAKE_WORD_ENABLED else '✗'}")
        print(f"   Terminal Word: {'✓' if config.TERMINAL_WORD_ENABLED else '✗'}")
        print(f"   TTS: {'✓' if config.TTS_ENABLED else '✗'}")
        print(f"   AEC: {'✓' if config.AEC_ENABLED else '✗'}")
        print(f"   Debug Mode: {'✓' if config.DEBUG_MODE else '✗'}")
        print(f"   Auto Restart: {'✓' if config.AUTO_RESTART else '✗'}")
        print(f"   MCP Server: {'✓' if self.enable_mcp_server else '✗'}")
        
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
        
        if config.WAKE_WORD_ENABLED or config.TERMINAL_WORD_ENABLED:
            print(f"\n🗣️ Wake Word Configuration:")
            if config.WAKE_WORD_ENABLED:
                print(f"   Wake Model: {config.WAKE_WORD_MODEL} (threshold: {config.WAKE_WORD_THRESHOLD})")
            if config.TERMINAL_WORD_ENABLED:
                print(f"   Terminal Phrases: {', '.join(config.TERMINAL_PHRASES)} (via transcription)")
    
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

    def _initialize_mcp_server(self) -> bool:
        """Initialize the MCP server."""
        if not self.enable_mcp_server:
            return True
            
        try:
            print("🔧 Initializing MCP server...")
            self.mcp_server = MCPServer()
            
            # Print MCP server status
            status = self.mcp_server.get_server_status()
            print(f"   Tools available: {status['tools_available']}")
            print(f"   Tool names: {', '.join(status['tool_names'])}")
            
            print("✅ MCP server initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ MCP server initialization failed: {e}")
            return False

    def _start_mcp_server_background(self):
        """Start MCP server in background thread."""
        if not self.mcp_server:
            return
            
        def run_mcp_server():
            try:
                print("🚀 Starting MCP server in background...")
                # Note: In a real implementation, this would start the MCP protocol handler
                # For now, it just keeps the server instance alive and ready for tool calls
                while self.running:
                    time.sleep(1)
                print("🛑 MCP server background thread stopped")
            except Exception as e:
                print(f"❌ MCP server background error: {e}")
        
        self.mcp_server_thread = threading.Thread(target=run_mcp_server, daemon=True)
        self.mcp_server_thread.start()
        print("✅ MCP server running in background")

    def initialize(self) -> bool:
        """Initialize the voice assistant system."""
        try:
            print("\n🔧 Initializing components...")
            
            # Check dependencies
            if not self._check_dependencies():
                return False
            
            # Initialize MCP server first
            if not self._initialize_mcp_server():
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
            
            # Start MCP server in background
            if self.mcp_server:
                self._start_mcp_server_background()
            
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
    
    def run_terminal_test_mode(self):
        """Run in terminal test mode for testing MCP tools via text input."""
        if not self.mcp_server:
            print("❌ MCP server not available")
            return
        
        print("\n🧪 Terminal Test Mode - MCP Tools Testing")
        print("=" * 50)
        
        # Get available tools
        status = self.mcp_server.get_server_status()
        tools = status['tool_names']
        
        print(f"Available tools: {', '.join(tools)}")
        print("\nExamples:")
        print("  - 'turn on the living room lights'")
        print("  - 'play some music on spotify'")
        print("  - 'set the lighting scene to movie'")
        print("  - 'what's the current spotify user?'")
        print("  - 'pause the music'")
        print("\nType 'quit', 'exit', or 'q' to exit test mode")
        print("Type 'tools' to see available tools again")
        print("Type 'status' to see system status")
        print("Press Ctrl+C to exit at any time")
        print("-" * 50)
        
        # Initialize the streaming chatbot for tool processing
        try:
            from core.streaming_chatbot import ToolEnabledStreamingChatbot
            chatbot = ToolEnabledStreamingChatbot()
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            return
        
        # Set a flag for test mode to handle signals differently
        original_running = self.running
        self.running = False  # Prevent the main signal handler from shutting down everything
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Exiting terminal test mode")
                        break
                    elif user_input.lower() == 'tools':
                        print(f"Available tools: {', '.join(tools)}")
                        continue
                    elif user_input.lower() == 'status':
                        status = self.mcp_server.get_server_status()
                        print(f"🔧 Server Status:")
                        print(f"   Tools available: {status['tools_available']}")
                        print(f"   Tool names: {', '.join(status['tool_names'])}")
                        continue
                    
                    # Process the input through the chatbot
                    print("🤖 Assistant: ", end="", flush=True)
                    
                    response_text = chatbot.process_text_input(user_input)
                    print(response_text)
                    
                except EOFError:
                    # Handle Ctrl+D gracefully
                    print("\n👋 Exiting terminal test mode")
                    break
                except Exception as e:
                    print(f"\n❌ Error processing input: {e}")
                    continue
                    
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully and save conversation
            print("\n\n💾 Saving test conversation before exit...")
            try:
                # Save the current test conversation
                messages = chatbot.conversation.get_chat_minus_sys_prompt()
                if messages and len(messages) > 0:  # Any messages without system prompt
                    chatbot.send_to_db(messages)
                    print("✅ Test conversation saved to database")
                else:
                    print("ℹ️ No test conversation to save")
            except Exception as save_error:
                print(f"⚠️ Error saving test conversation: {save_error}")
            
            print("👋 Exiting terminal test mode")
        finally:
            # Restore the original running state
            self.running = original_running

    def run_forever(self):
        """Run the voice assistant until interrupted."""
        if not self.running:
            print("❌ System not started")
            return
    
        self.state_manager = StateManager()
        
        
        try:
            # Print usage instructions
            if config.CONVERSATION_MODE == "wake_word":
                self.state_manager.set("chat_controlled_state", "asleep")
                print("💡 Say the wake word to start a conversation")
            elif config.CONVERSATION_MODE == "interactive":
                self.state_manager.set("chat_controlled_state", "listening (active)")
                print("💡 Press Enter to start a conversation")
            elif config.CONVERSATION_MODE == "continuous":
                self.state_manager.set("chat_controlled_state", "listening (quiet)")
                print("💡 Continuous listening mode - speak naturally")
            
            if self.mcp_server:
                tools = self.mcp_server.get_server_status()['tool_names']
                print(f"🔧 Available tools: {', '.join(tools)}")
            
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
        import time
        start_time = time.time()
        print("🛑 Stopping voice assistant...")
        
        # Set running to False first to stop any loops
        self.running = False
        
        # Save any active conversations before stopping components (failsafe)
        if not hasattr(self, '_conversations_saved'):
            try:
                print("💾 Final check for unsaved conversations...")
                self._save_active_conversations()
                self._conversations_saved = True
            except Exception as e:
                print(f"⚠️ Error in final conversation save: {e}")
        
        # Stop orchestrator with timeout protection
        if self.orchestrator:
            try:
                print("🛑 Stopping orchestrator...")
                orchestrator_start = time.time()
                self.orchestrator.stop_all()
                orchestrator_time = time.time() - orchestrator_start
                print(f"   Orchestrator stopped in {orchestrator_time:.2f}s")
            except Exception as e:
                print(f"⚠️ Error stopping orchestrator: {e}")
        
        # Stop MCP server with timeout
        if self.mcp_server_thread and self.mcp_server_thread.is_alive():
            print("🛑 Stopping MCP server...")
            mcp_start = time.time()
            try:
                timeout = 0.3 if config.FAST_SHUTDOWN else 1.0
                self.mcp_server_thread.join(timeout=timeout)
                mcp_time = time.time() - mcp_start
                if self.mcp_server_thread.is_alive():
                    print(f"⚠️ MCP server thread did not stop gracefully after {mcp_time:.2f}s")
                else:
                    print(f"   MCP server stopped in {mcp_time:.2f}s")
            except Exception as e:
                print(f"⚠️ Error stopping MCP server: {e}")
        
        total_time = time.time() - start_time
        print(f"✅ Voice assistant stopped in {total_time:.2f}s")

    def get_mcp_server(self) -> Optional[MCPServer]:
        """Get the MCP server instance for external access."""
        return self.mcp_server


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
    
    parser.add_argument(
        "--no-mcp-server",
        action="store_false",
        dest="enable_mcp_server",
        help="Disable the integrated MCP tool server"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in terminal test mode for testing MCP tools via text input"
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
        enable_mcp_server = getattr(args, 'enable_mcp_server', True)
        app = VoiceAssistantApp(config_overrides, enable_mcp_server)
        
        # Show system information
        app._print_system_info()
        
        # Handle status request
        if args.status:
            if app.initialize():
                print(f"\n📊 System Status:")
                print(f"   Voice Assistant: {'✓' if app.orchestrator else '✗'}")
                print(f"   MCP Server: {'✓' if app.mcp_server else '✗'}")
                if app.mcp_server:
                    mcp_status = app.mcp_server.get_server_status()
                    print(f"   Available Tools: {mcp_status['tools_available']}")
                    for tool_name in mcp_status['tool_names']:
                        print(f"     - {tool_name}")
            return
        
        # Handle test mode
        if args.test:
            if not app.initialize():
                print("❌ Failed to initialize voice assistant")
                sys.exit(1)
            
            # Run in test mode (no need to start full system)
            app.run_terminal_test_mode()
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