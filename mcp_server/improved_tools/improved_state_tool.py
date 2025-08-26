"""
Improved State Management Tool using ImprovedBaseTool.

This tool manages system state and user preferences via voice commands with
enhanced parameter descriptions and type safety.
"""

import sys
sys.path.insert(0, '../..')

from config import LOG_TOOLS
from core.state_management.statemanager import StateManager
from mcp_server.improved_base_tool import ImprovedBaseTool
try:
    import config
except ImportError:
    # Fallback for MCP server context
    config = None
from typing import Dict, Any, Literal


class ImprovedStateTool(ImprovedBaseTool):
    """Enhanced tool for managing system state and user preferences via voice commands."""
    
    name = "improved_state_manager"
    description = "Manage system state including user preferences, current settings, and application state. Use this tool when users want to change their Spotify account, lighting scene, volume level, or do-not-disturb status. This tool provides centralized control over all chat-controllable system settings."
    version = "1.0.1"
    
    def __init__(self):
        """Initialize the improved state management tool."""
        super().__init__()
        self.state_manager = StateManager()
        
        # Get valid state types dynamically from the state manager
        self.valid_state_types = list(self.state_manager.state.get("chat_controlled_state", {}).keys())
        
        # If empty, use fallback from config or defaults
        if not self.valid_state_types:
            if config and hasattr(config, 'ALL_CHAT_CONTROLLED_STATES'):
                self.valid_state_types = config.ALL_CHAT_CONTROLLED_STATES
            else:
                self.valid_state_types = ["current_spotify_user", "lighting_scene", "volume_level", "do_not_disturb"]
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with enhanced parameter descriptions.
        
        Returns:
            Comprehensive JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "state_type": {
                    "type": "string",
                    "description": "The type of system state to control. Options: 'current_spotify_user' (change active Spotify account between Morgan/Spencer), 'lighting_scene' (set mood/party/movie/all_on/all_off), 'volume_level' (adjust system volume 0-100), 'do_not_disturb' (enable/disable DND mode). Choose based on what the user wants to change.",
                    "enum": self.valid_state_types
                },
                "new_state": {
                    "type": "string", 
                    "description": "The new value to set for the specified state type. For 'current_spotify_user': use 'Morgan' or 'Spencer'. For 'lighting_scene': use 'mood', 'party', 'movie', 'all_on', or 'all_off'. For 'volume_level': use number as string '0'-'100'. For 'do_not_disturb': use 'true' or 'false'. Always match the value to the state_type being changed."
                }
            },
            "required": ["state_type", "new_state"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the state management tool.
        
        Args:
            params: Tool parameters containing state_type and new_state
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # Extract and validate parameters
            state_type = params.get("state_type")
            new_state = params.get("new_state")
            
            if not state_type or not new_state:
                return {
                    "success": False,
                    "error": "Missing required parameters: state_type and new_state are both required",
                    "valid_state_types": self.valid_state_types
                }

            # Validate state type
            if state_type not in self.valid_state_types:
                return {
                    "success": False,
                    "error": f"Invalid state_type '{state_type}'. Valid options: {self.valid_state_types}",
                    "valid_state_types": self.valid_state_types
                }
            if LOG_TOOLS:
                # Route logging to stderr via the logger
                self.logger.info("Executing Tool: State -- %s", params)
            
            # Perform state change
            self.state_manager.set(state_type, new_state)
            self.log_info(f"Successfully changed {state_type} to {new_state}")
            
            # Get current state for confirmation
            current_value = self.state_manager.get(state_type)
            
            return {
                "success": True,
                "message": f"Successfully set {state_type} to {new_state}",
                "state_type": state_type,
                "previous_state": current_value if current_value != new_state else "unknown",
                "new_state": new_state,
                "timestamp": self.state_manager.get_last_updated(state_type) if hasattr(self.state_manager, 'get_last_updated') else None
            }
            
        except Exception as e:
            self.log_error(f"Failed to set state: {e}")
            return {
                "success": False,
                "error": f"Failed to set {params.get('state_type', 'unknown')}: {str(e)}",
                "state_type": params.get("state_type"),
                "attempted_new_state": params.get("new_state")
            }
    
    def get_current_state(self, state_type: str = None) -> Dict[str, Any]:
        """
        Get current state values.
        
        Args:
            state_type: Optional specific state to get, or None for all states
            
        Returns:
            Dictionary with current state values
        """
        try:
            if state_type:
                if state_type in self.valid_state_types:
                    return {
                        "success": True,
                        "state_type": state_type,
                        "current_value": self.state_manager.get(state_type)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Invalid state_type '{state_type}'",
                        "valid_state_types": self.valid_state_types
                    }
            else:
                # Get all states
                all_states = {}
                for st in self.valid_state_types:
                    all_states[st] = self.state_manager.get(st)
                    
                return {
                    "success": True,
                    "all_states": all_states
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get state: {str(e)}"
            }