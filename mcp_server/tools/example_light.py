"""
Example smart light control tool demonstrating clean schema definition.
Shows how schemas should be defined directly in the tool class.
"""

from typing import Dict, Any
from mcp_server.base_tool import BaseTool


class SmartLightTool(BaseTool):
    """Tool for controlling smart home lights with in-class schema definition."""
    
    name = "smart_light"
    description = "Control smart home lights (brightness, color, on/off)"
    version = "1.0.0"
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Schema defined right in the tool - clean and maintainable!
        No external dependencies, no separate files to manage.
        """
        return {
            "type": "object",
            "description": "Control smart home lighting",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["on", "off", "dim", "brighten", "set_color"],
                    "description": "Action to perform on the lights"
                },
                "location": {
                    "type": "string",
                    "description": "Room or area (e.g., 'bedroom', 'kitchen', 'all')",
                    "default": "living room"
                },
                "brightness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Brightness level as percentage (0-100)"
                },
                "color": {
                    "type": "string",
                    "pattern": "^#[0-9A-Fa-f]{6}$",
                    "description": "Hex color code (e.g., #FF0000 for red)",
                    "examples": ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]
                },
                "duration": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Transition duration in seconds",
                    "default": 1.0
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the light control command."""
        action = params["action"]
        location = params.get("location", "living room")
        
        # Simulate smart light control
        result = {
            "success": True,
            "action": action,
            "location": location,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        if action == "on":
            brightness = params.get("brightness", 100)
            result.update({
                "status": "on",
                "brightness": brightness,
                "message": f"Turned on {location} lights at {brightness}% brightness"
            })
            
        elif action == "off":
            result.update({
                "status": "off",
                "message": f"Turned off {location} lights"
            })
            
        elif action in ["dim", "brighten"]:
            current_brightness = 50  # Simulate current brightness
            change = params.get("brightness", 20)
            
            if action == "dim":
                new_brightness = max(0, current_brightness - change)
            else:  # brighten
                new_brightness = min(100, current_brightness + change)
                
            result.update({
                "status": "adjusted",
                "brightness": new_brightness,
                "message": f"{action.title()}ed {location} lights to {new_brightness}%"
            })
            
        elif action == "set_color":
            color = params.get("color", "#FFFFFF")
            result.update({
                "status": "color_changed",
                "color": color,
                "message": f"Set {location} lights to color {color}"
            })
            
        return result