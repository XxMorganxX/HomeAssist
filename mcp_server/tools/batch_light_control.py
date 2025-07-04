import asyncio
from kasa import Discover
from config import LIGHT_ROOM_MAPPING
from mcp_server.base_tool import BaseTool
from typing import Dict, Any, List, Optional


class BatchLightControlTool(BaseTool):
    """Controls lights by name or by room"""
    name = "batch_light_control"
    description = "Control lights. When user says 'Light 1' or 'Light 2', use light_name ONLY. When user mentions a room like 'living room', use room ONLY. Never ask for room when light_name is given."
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.clients = {}
        self.light_config = LIGHT_ROOM_MAPPING
        
    async def _init_client(self, light_name: str):
        """Initialize a client for a specific light"""
        if light_name not in self.clients:
            light_info = self.light_config["lights"].get(light_name)
            if light_info and light_info["ip"] and light_info["credentials"]:
                self.clients[light_name] = await Discover.discover_single(
                    light_info["ip"],
                    username=light_info["credentials"]["username"],
                    password=light_info["credentials"]["password"]
                )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": "List of commands. For 'turn on Light 1', use: [{\"light_name\": \"Light 1\", \"action\": \"on\"}]. Do NOT include room.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "light_name": {
                                "type": "string",
                                "enum": list(self.light_config["lights"].keys()),
                                "description": "The specific light to control (e.g., 'Light 1'). Use this when user mentions a specific light name."
                            },
                            "action": {
                                "type": "string",
                                "enum": ["on", "off", "toggle"],
                                "description": "The action to perform"
                            },
                            "room": {
                                "type": "string",
                                "enum": list(self.light_config["rooms"].keys()),
                                "description": "Optional: Only provide if user mentions a room name, not needed for specific lights"
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            "required": ["commands"],
            "description": self.description
        }
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        return asyncio.run(self.control_lights(params))
    
    async def control_lights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control multiple lights"""
        commands = params.get("commands", [])
        
        # Expand room-based commands into individual light commands
        expanded_commands = []
        for cmd in commands:
            if "room" in cmd and "light_name" not in cmd:
                # Room-based command: expand to all lights in room
                room = cmd["room"]
                lights_in_room = self.light_config["rooms"].get(room, [])
                for light_name in lights_in_room:
                    expanded_commands.append({
                        "light_name": light_name,
                        "action": cmd["action"],
                        "room": room
                    })
            else:
                # Direct light command
                expanded_commands.append(cmd)
        
        self.log_info(f"Processing {len(expanded_commands)} light commands (from {len(commands)} original commands)")
        
        # Execute all commands concurrently
        tasks = []
        for cmd in expanded_commands:
            tasks.append(self._control_single_light(cmd))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        
        return {
            "success": success_count == len(expanded_commands),
            "total": len(expanded_commands),
            "successful": success_count,
            "results": results
        }
    
    async def _control_single_light(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Control a single light"""
        light_name = command.get("light_name")
        action = command.get("action")
        room = command.get("room", "")
        
        # Get light configuration
        light_info = self.light_config["lights"].get(light_name)
        
        if not light_info:
            return {
                "success": False,
                "light": light_name,
                "error": f"Unknown light: {light_name}"
            }
        
        if not light_info["ip"] or not light_info["credentials"]:
            return {
                "success": False,
                "light": light_name,
                "room": room or light_info["room"],
                "error": "Light not yet configured"
            }
        
        try:
            await self._init_client(light_name)
            client = self.clients[light_name]
            
            if action == "on":
                await client.turn_on()
            elif action == "off":
                await client.turn_off()
            elif action == "toggle":
                await client.update()
                if client.is_on:
                    await client.turn_off()
                else:
                    await client.turn_on()
            
            await client.update()
            return {
                "success": True,
                "light": light_name,
                "action": action,
                "room": room or light_info["room"],
                "status": "on" if client.is_on else "off"
            }
                
        except Exception as e:
            return {
                "success": False,
                "light": light_name,
                "room": room or light_info["room"],
                "error": str(e)
            }