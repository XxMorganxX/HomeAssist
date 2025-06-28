import asyncio
from kasa import Discover
from config import LIGHT_ONE_IP
from mcp_server.base_tool import BaseTool, CoreServices
from typing import Dict, Any, Optional


class LightOffTool(BaseTool):
    """Turns off the light"""
    name = "light_off"
    description = "Called when the user wants to turn off the light"
    version = "1.0.0"
    
    def __init__(self, core_services: Optional[CoreServices]):
        super().__init__(core_services)
        self.client = None
        
    async def _init_client(self):
        """Initialize the client if not already done"""
        if self.client is None:
            self.client = await Discover.discover_single(
                LIGHT_ONE_IP, 
                username="morgannstuart@gmail.com", 
                password="ithaca-home-2025"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "description": "The room the light is in"
                },
                "light_name": {
                    "type": "string",
                    "description": "The name of the light"
                }
            },
            "required": ["room", "light_name"]
        }
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        return asyncio.run(self.turn_off(params))
    
    async def turn_off(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Turn off the light"""
        room = params.get("room")
        light_name = params.get("light_name")
        self.log_info(f"Turning off {light_name} in {room}")
        
        await self._init_client()
        await self.client.turn_off()
        await self.client.update()
        
        return {"message": "success", "status": "light turned off"}