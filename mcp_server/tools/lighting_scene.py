import asyncio
from kasa import Discover
from config import LIGHT_ONE_IP
from mcp_server.base_tool import BaseTool
from typing import Dict, Any
from core.state_management.statemanager import StateManager


class LightingSceneTool(BaseTool):
    """Controls multiple lights based on predefined scenes"""
    name = "lighting_scene"
    description = "Sets a lighting scene that controls multiple lights at once"
    version = "1.0.0"
    
    # Define scenes with their light configurations
    SCENES = {
        "movie": {
            "lights": ["Light 1"],
            "actions": ["off"]
        },
        "party": {
            "lights": ["Light 1", "Light 2"],
            "actions": ["on", "on"]
        },
        "mood": {
            "lights": ["Light 1"],
            "actions": ["dim"]  # Could implement dimming if supported
        },
        "all_on": {
            "lights": ["Light 1", "Light 2"],
            "actions": ["on", "on"]
        },
        "all_off": {
            "lights": ["Light 1", "Light 2"], 
            "actions": ["off", "off"]
        }
    }
    
    def __init__(self):
        super().__init__()
        self.clients = {}
        self.state_manager = StateManager()
        
    async def _init_client(self, light_name: str):
        """Initialize a client for a specific light"""
        if light_name not in self.clients:
            if light_name == "Light 1":
                self.clients[light_name] = await Discover.discover_single(
                    LIGHT_ONE_IP,
                    username="morgannstuart@gmail.com",
                    password="ithaca-home-2025"
                )
            # Add more lights as needed
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "description": "The lighting scene to activate",
                    "enum": list(self.SCENES.keys())
                }
            },
            "required": ["scene"],
            "description": self.get_info()["description"]
        }
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        return asyncio.run(self.set_scene(params))
    
    async def set_scene(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set the lighting scene"""
        scene_name = params.get("scene")
        self.log_info(f"Setting lighting scene to: {scene_name}")
        
        if scene_name not in self.SCENES:
            return {"success": False, "error": f"Unknown scene: {scene_name}"}
        
        scene = self.SCENES[scene_name]
        results = []
        
        # Process each light in the scene
        for light_name, action in zip(scene["lights"], scene["actions"]):
            try:
                if light_name == "Light 1":
                    await self._init_client(light_name)
                    client = self.clients[light_name]
                    
                    if action == "on":
                        await client.turn_on()
                    elif action == "off":
                        await client.turn_off()
                    elif action == "dim":
                        await client.turn_on()
                        # If dimming is supported: await client.set_brightness(50)
                    
                    await client.update()
                    results.append(f"{light_name}: {action}")
                    
                elif light_name == "Light 2":
                    results.append(f"{light_name}: not yet configured")
                    
            except Exception as e:
                results.append(f"{light_name}: error - {str(e)}")
        
        # Update state
        self.state_manager.set("lighting_scene", scene_name)
        
        return {
            "success": True,
            "scene": scene_name,
            "results": results
        }