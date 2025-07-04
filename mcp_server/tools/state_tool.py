from core.state_management.statemanager import StateManager
from mcp_server.base_tool import BaseTool
import config


class StateTool(BaseTool):
    """Tool for managing system state and user preferences via voice commands"""
    
    name = "state_manager"
    description = "Manage system state including user preferences, current settings, and application state"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.state_manager = StateManager()
        self.valid_state_types = [k for k in self.state_manager.state["chat_controlled_state"].keys()]
        
    def get_info(self):
        return {
            "name": "StateTool",
            f"description": f"""Control the state of the autonomous system when prompted by the user. 
            This tool can use user specifications to change the {self.valid_state_types} system_types. 
            Should be called when the user asks to change the spotify user account, or when asking to change the lighting scene.""",
            "version": "1.0.0"
        }


    def get_schema(self):
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "state_type": {
                    "type": "string",
                    "description": "State that is being controlled by the user",
                    "enum": self.valid_state_types
                },
                "new_state": {
                    "type": "string",
                    "description": "New state value to set",
                    "enum": config.ALL_CHAT_CONTROLLED_STATES
                }
            },
            "required": ["state_type", "new_state"],
            "description": self.get_info()["description"]
        }
    
    def execute(self, params):
        state_type = params["state_type"]
        new_state = params["new_state"]
        
        try:

            self.state_manager.set(state_type, new_state)
            self.log_info(f"Set {state_type} to {new_state}")
            
            return {
                "success": True,
                "message": f"Successfully set {state_type} to {new_state}",
                "state_type": state_type,
                "new_state": new_state
            }
            
        except Exception as e:
            self.log_error(f"Failed to set state: {e}")
            return {
                "success": False,
                "error": f"Failed to set {state_type}: {str(e)}"
            }
    
    
