"""
MCP Server bootstrap and main server implementation.
Coordinates tool discovery, core services, and MCP protocol handling.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug Python environment
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.path[:3]}...")

# Test kasa import
try:
    import kasa
    logger.info(f"Kasa module found at: {kasa.__file__}")
except ImportError as e:
    logger.error(f"Cannot import kasa: {e}")
    logger.info(f"Site packages: {[p for p in sys.path if 'site-packages' in p]}")

from mcp_server.tool_registry import ToolRegistry
from mcp_server.base_tool import CoreServices

# Import core modules with fallback for missing dependencies
try:
    from core.speech_services import SpeechServices, ConversationManager
    SPEECH_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Speech services not available: {e}")
    SpeechServices = None
    ConversationManager = None
    SPEECH_SERVICES_AVAILABLE = False

try:
    import config
    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Config not available: {e}")
    config = None
    CONFIG_AVAILABLE = False


class MCPServer:
    """
    Main MCP Server that coordinates tools and core services.
    Provides a framework for extensible tool creation.
    """
    
    def __init__(self, config_module=None):
        """
        Initialize MCP server.
        
        Args:
            config_module: Configuration module (defaults to project config)
        """
        self.config = config_module or config
        self.tool_registry = ToolRegistry()
        self.core_services = None
        self.available_tools: Dict[str, Any] = {}
        
        # Initialize core services
        self._initialize_core_services()
        
        # Discover and register tools
        self._discover_tools()
        
    def _initialize_core_services(self):
        """Initialize core services that tools can access."""
        try:
            # Initialize speech services if available and API key is set
            api_key = os.getenv("OPENAI_KEY")
            speech_services = None
            conversation_manager = None
            
            if SPEECH_SERVICES_AVAILABLE and api_key and CONFIG_AVAILABLE:
                speech_services = SpeechServices(
                    api_key=api_key,
                    whisper_model=self.config.WHISPER_MODEL,
                    chat_model=self.config.RESPONSE_MODEL
                )
                conversation_manager = ConversationManager(self.config.SYSTEM_PROMPT)
                logger.info("Speech services initialized")
            else:
                if not SPEECH_SERVICES_AVAILABLE:
                    logger.warning("Speech services not available - missing dependencies")
                elif not api_key:
                    logger.warning("No OPENAI_KEY found - speech services unavailable")
                elif not CONFIG_AVAILABLE:
                    logger.warning("Config not available - using defaults")
            
            # Initialize audio processor configuration (with defaults if config unavailable)
            audio_processor_config = {}
            if CONFIG_AVAILABLE:
                audio_processor_config = {
                    'sample_rate': self.config.SAMPLE_RATE,
                    'frame_ms': self.config.FRAME_MS,
                    'vad_mode': self.config.VAD_MODE,
                    'silence_end_sec': self.config.SILENCE_END_SEC,
                    'max_utterance_sec': self.config.MAX_UTTERANCE_SEC
                }
            else:
                # Provide defaults
                audio_processor_config = {
                    'sample_rate': 16000,
                    'frame_ms': 30,
                    'vad_mode': 2,
                    'silence_end_sec': 0.7,
                    'max_utterance_sec': 15
                }
            
            # Create core services container
            self.core_services = CoreServices(
                audio_processor=audio_processor_config,
                speech_services=speech_services,
                conversation_manager=conversation_manager,
                config=self.config if CONFIG_AVAILABLE else None
            )
            
            logger.info("Core services initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize core services: {e}")
            # Don't raise - create minimal core services
            self.core_services = CoreServices(
                audio_processor={},
                speech_services=None,
                conversation_manager=None,
                config=None
            )
            logger.warning("Created minimal core services")
    
    def _discover_tools(self):
        """Discover and validate all available tools."""
        try:
            discovered = self.tool_registry.discover_tools()
            logger.info(f"Discovered {len(discovered)} tool modules")
            
            # Validate tools and build available tools dict
            for tool_name in self.tool_registry.get_available_tools():
                if self.tool_registry.validate_tool(tool_name):
                    schema = self.tool_registry.get_tool_schema(tool_name)
                    self.available_tools[tool_name] = {
                        'name': tool_name,
                        'schema': schema,
                        'status': 'available'
                    }
                    logger.info(f"Tool '{tool_name}' registered and validated")
                else:
                    logger.error(f"Tool '{tool_name}' failed validation")
                    
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools with their schemas.
        
        Returns:
            List of tool information dictionaries
        """
        return list(self.available_tools.values())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool_instance = self.tool_registry.get_tool_instance(tool_name, self.core_services)
        return tool_instance.get_info()
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        try:
            tool_instance = self.tool_registry.get_tool_instance(tool_name, self.core_services)
            result = tool_instance.safe_execute(params)
            
            logger.info(f"Tool '{tool_name}' executed successfully: {result.get('success', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute tool '{tool_name}': {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def reload_tools(self) -> Dict[str, Any]:
        """
        Reload all tools (useful for development).
        
        Returns:
            Reload status information
        """
        try:
            old_count = len(self.available_tools)
            self.available_tools.clear()
            
            reloaded = self.tool_registry.reload_tools()
            self._discover_tools()
            
            new_count = len(self.available_tools)
            
            logger.info(f"Tools reloaded: {old_count} -> {new_count}")
            
            return {
                "success": True,
                "old_count": old_count,
                "new_count": new_count,
                "reloaded_modules": reloaded
            }
            
        except Exception as e:
            logger.error(f"Tool reload failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status.
        
        Returns:
            Server status information
        """
        return {
            "status": "running",
            "tools_available": len(self.available_tools),
            "tool_names": list(self.available_tools.keys()),
            "core_services": {
                "speech_services": self.core_services.speech_services is not None,
                "conversation_manager": self.core_services.conversation_manager is not None,
                "audio_processor": self.core_services.audio_processor is not None,
                "config": self.core_services.config is not None
            }
        }
    
    def start(self):
        """Start the MCP server and keep it running."""
        logger.info("MCP Server starting...")
        logger.info(f"Available tools: {list(self.available_tools.keys())}")
        logger.info("Server ready to handle requests")
        
        print("\n=== MCP Server Running ===")
        print(f"Tools loaded: {len(self.available_tools)}")
        print(f"Available tools: {', '.join(self.available_tools.keys())}")
        print("\nThe server is now running. Press Ctrl+C to stop.")
        print("\nTo use this server:")
        print("1. Run examples/interactive_tool_chat.py in another terminal")
        print("2. Or use the HTTP server: python mcp_server/run_server.py")
        
        try:
            # Keep the server running
            while True:
                time.sleep(1)
                
                # Optional: Auto-reload tools every 30 seconds for development
                # Uncomment the following lines to enable auto-reload:
                # if not hasattr(self, '_last_reload'):
                #     self._last_reload = time.time()
                # elif time.time() - self._last_reload > 30:
                #     logger.info("Auto-reloading tools...")
                #     self.reload_tools()
                #     self._last_reload = time.time()
                
        except KeyboardInterrupt:
            print("\n\nShutting down MCP Server...")
            logger.info("MCP Server stopped by user")
            return


def main():
    """Entry point for running the MCP server."""
    try:
        server = MCPServer()
        
        # Print server status
        status = server.get_server_status()
        print("\n=== MCP Server Status ===")
        print(f"Status: {status['status']}")
        print(f"Tools available: {status['tools_available']}")
        print(f"Tool names: {', '.join(status['tool_names'])}")
        
        # Print core services status
        print("\n=== Core Services ===")
        for service, available in status['core_services'].items():
            print(f"{service}: {'✓' if available else '✗'}")
        
        # List all tools with schemas
        print("\n=== Available Tools ===")
        for tool in server.list_tools():
            print(f"- {tool['name']}: {tool['schema'].get('description', 'No description')}")
        
        print("\n=== Server Ready ===")
        print("MCP server initialized and ready to handle requests")
        
        # Start server (in a real implementation, this would start the MCP protocol handler)
        server.start()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()