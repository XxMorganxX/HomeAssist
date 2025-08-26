"""
Improved Batch Light Control Tool using ImprovedBaseTool.

This tool controls smart lights by name or room with enhanced parameter descriptions,
better validation, and comprehensive lighting control options.
"""

import asyncio
try:
    # Preferred import
    from kasa import Discover  # type: ignore
except Exception:
    try:
        # Some environments expose the package as `Kasa`
        from Kasa import Discover  # type: ignore
    except Exception:
        Discover = None  # type: ignore
import sys
sys.path.insert(0, '../..')

try:
    from config import LIGHT_ROOM_MAPPING
except ImportError:
    # Fallback configuration for MCP server context
    LIGHT_ROOM_MAPPING = {
        "lights": {
            "Light 1": {"ip": "192.168.1.100", "room": "living_room", "credentials": {"username": None, "password": None}},
            "Light 2": {"ip": "192.168.1.101", "room": "bedroom", "credentials": {"username": None, "password": None}}
        },
        "rooms": {"living_room": ["Light 1"], "bedroom": ["Light 2"], "all": ["Light 1", "Light 2"]}
    }
from mcp_server.improved_base_tool import ImprovedBaseTool
from typing import Dict, Any, List, Optional, Literal
from config import LOG_TOOLS

class ImprovedBatchLightControlTool(ImprovedBaseTool):
    """Enhanced tool for controlling smart lights with detailed command specifications."""
    
    name = "improved_batch_light_control"
    description = "Control Kasa smart lights by specific light name or room. When user mentions a specific light like 'Light 1' or 'Light 2', use light_name parameter ONLY. When user mentions a room like 'living room' or 'bedroom', use room parameter ONLY. Never ask for room when light_name is provided. Supports multiple simultaneous commands for efficient batch operations."
    version = "1.0.1"
    
    def __init__(self):
        """Initialize the improved batch light control tool."""
        super().__init__()
        self.clients = {}
        self.light_config = LIGHT_ROOM_MAPPING
        
        # Extract available lights and rooms for validation
        self.available_lights = list(self.light_config.get("lights", {}).keys())
        self.available_rooms = list(self.light_config.get("rooms", {}).keys())
        
    async def _init_client(self, light_name: str):
        """Initialize a Kasa client for a specific light."""
        if light_name not in self.clients:
            light_info = self.light_config["lights"].get(light_name)
            if light_info and light_info.get("ip"):
                try:
                    if Discover is None:
                        raise RuntimeError(
                            "python-kasa not available. Install with: pip install python-kasa"
                        )
                    # Handle both authenticated and non-authenticated lights
                    credentials = light_info.get("credentials", {})
                    if credentials.get("username") and credentials.get("password"):
                        self.clients[light_name] = await Discover.discover_single(
                            light_info["ip"],
                            username=credentials["username"],
                            password=credentials["password"]
                        )
                    else:
                        self.clients[light_name] = await Discover.discover_single(
                            light_info["ip"]
                        )
                except Exception as e:
                    self.logger.error(f"Failed to initialize client for {light_name}: {e}")
                    raise
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with detailed command descriptions.
        
        Returns:
            Comprehensive JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": f"List of lighting commands to execute. Each command controls either a specific light or all lights in a room. Available lights: {self.available_lights}. Available rooms: {self.available_rooms}. Examples: For 'turn on Light 1' use [{{'light_name': 'Light 1', 'action': 'on'}}]. For 'dim living room lights' use [{{'room': 'living_room', 'action': 'dim', 'brightness': 30}}]. Never mix light_name and room in the same command.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "light_name": {
                                "type": "string",
                                "enum": self.available_lights,
                                "description": f"Specific light to control. Use when user mentions exact light names like 'Light 1' or 'Light 2'. Available lights: {', '.join(self.available_lights)}. Do NOT use this with room parameter."
                            },
                            "room": {
                                "type": "string", 
                                "enum": self.available_rooms,
                                "description": f"Room containing lights to control. Use when user mentions rooms like 'living room', 'bedroom', or 'all'. Available rooms: {', '.join(self.available_rooms)}. This will control all lights in the specified room. Do NOT use this with light_name parameter."
                            },
                            "action": {
                                "type": "string",
                                "enum": ["on", "off", "toggle", "dim", "brighten"],
                                "description": "Action to perform on the light(s). 'on' turns lights on at full brightness, 'off' turns lights off, 'toggle' switches current state, 'dim' reduces brightness (requires brightness parameter), 'brighten' increases brightness (requires brightness parameter)."
                            },
                        },
                        "required": ["action"],
                        "oneOf": [
                            {"required": ["light_name"]},
                            {"required": ["room"]}
                        ]
                    },
                    "minItems": 1
                }
            },
            "required": ["commands"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the batch light control commands.
        
        Args:
            params: Tool parameters containing commands array
            
        Returns:
            Dictionary containing execution results for all commands
        """
        try:
            commands = params.get("commands", [])
            
            if not commands:
                return {
                    "success": False,
                    "error": "No commands provided. At least one command is required.",
                    "available_lights": self.available_lights,
                    "available_rooms": self.available_rooms
                }
            
            if LOG_TOOLS:
                # Use logger so logs appear in the agent terminal and avoid stdout clashes on stdio
                self.logger.info("Executing Tool: Light Control -- %s", params)
            
            # Validate commands before execution
            validation_errors = self._validate_commands(commands)
            if validation_errors:
                return {
                    "success": False,
                    "error": "Command validation failed",
                    "validation_errors": validation_errors,
                    "available_lights": self.available_lights,
                    "available_rooms": self.available_rooms
                }
            
            # Execute commands asynchronously
            results = asyncio.run(self._execute_commands_async(commands))
            
            # Analyze results
            successful_commands = [r for r in results if r.get("success")]
            failed_commands = [r for r in results if not r.get("success")]
            
            return {
                "success": len(failed_commands) == 0,  # Success if no failures
                "total_commands": len(commands),
                "successful_commands": len(successful_commands),
                "failed_commands": len(failed_commands),
                "results": results,
                "summary": self._create_execution_summary(results),
                "lights_affected": self._get_affected_lights(commands),
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing batch light control: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "total_commands": len(params.get("commands", [])),
                "successful_commands": 0,
                "failed_commands": len(params.get("commands", []))
            }
    
    def _validate_commands(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Validate all commands before execution."""
        errors = []
        
        for i, cmd in enumerate(commands):
            cmd_errors = []
            
            # Check for required action
            if "action" not in cmd:
                cmd_errors.append("Missing required 'action' parameter")
            elif cmd["action"] not in ["on", "off", "toggle", "dim", "brighten"]:
                cmd_errors.append(f"Invalid action '{cmd['action']}'")
            
            # Check for exactly one of light_name or room
            has_light = "light_name" in cmd
            has_room = "room" in cmd
            
            if not has_light and not has_room:
                cmd_errors.append("Must specify either 'light_name' or 'room'")
            elif has_light and has_room:
                cmd_errors.append("Cannot specify both 'light_name' and 'room' in the same command")
            
            # Validate light_name if present
            if has_light:
                light_name = cmd["light_name"]
                if light_name not in self.available_lights:
                    cmd_errors.append(f"Unknown light '{light_name}'. Available: {self.available_lights}")
            
            # Validate room if present
            if has_room:
                room = cmd["room"]
                if room not in self.available_rooms:
                    cmd_errors.append(f"Unknown room '{room}'. Available: {self.available_rooms}")
            
            # Validate brightness parameter
            if cmd.get("action") in ["dim", "brighten"]:
                if "brightness" not in cmd:
                    cmd_errors.append(f"Action '{cmd['action']}' requires 'brightness' parameter")
                elif not isinstance(cmd["brightness"], int) or not (1 <= cmd["brightness"] <= 100):
                    cmd_errors.append("Brightness must be an integer between 1 and 100")
            
            # Add command-specific errors
            if cmd_errors:
                errors.append(f"Command {i+1}: {'; '.join(cmd_errors)}")
        
        return errors
    
    async def _execute_commands_async(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute all commands asynchronously."""
        tasks = []
        for i, cmd in enumerate(commands):
            task = self._execute_single_command(cmd, i)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_command(self, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Execute a single light control command."""
        try:
            # Determine target lights
            if "light_name" in cmd:
                target_lights = [cmd["light_name"]]
            else:  # room specified
                room = cmd["room"]
                target_lights = self.light_config["rooms"].get(room, [])
            
            if not target_lights:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"No lights found for target: {cmd.get('light_name') or cmd.get('room')}",
                    "command": cmd
                }
            
            # Execute action on each target light
            light_results = []
            for light_name in target_lights:
                try:
                    await self._init_client(light_name)
                    client = self.clients.get(light_name)
                    
                    if not client:
                        light_results.append({
                            "light": light_name,
                            "success": False,
                            "error": "Failed to initialize client"
                        })
                        continue
                    
                    # Perform the action
                    await self._perform_light_action(client, cmd, light_name)
                    
                    light_results.append({
                        "light": light_name,
                        "success": True,
                        "action": cmd["action"],
                        "brightness": cmd.get("brightness"),
                        "color_temp": cmd.get("color_temp")
                    })
                    
                except Exception as e:
                    light_results.append({
                        "light": light_name,
                        "success": False,
                        "error": str(e)
                    })
            
            # Determine overall command success
            successful_lights = [r for r in light_results if r["success"]]
            command_success = len(successful_lights) > 0
            
            return {
                "success": command_success,
                "command_index": cmd_index,
                "command": cmd,
                "target_lights": target_lights,
                "lights_controlled": len(successful_lights),
                "total_lights": len(target_lights),
                "light_results": light_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "command_index": cmd_index,
                "command": cmd,
                "error": str(e)
            }
    
    async def _perform_light_action(self, client, cmd: Dict[str, Any], light_name: str):
        """Perform the specified action on a light client."""
        action = cmd["action"]
        
        if action == "on":
            await client.turn_on()
        elif action == "off":
            await client.turn_off()
        elif action == "toggle":
            if client.is_on:
                await client.turn_off()
            else:
                await client.turn_on()
        elif action in ["dim", "brighten"]:
            brightness = cmd.get("brightness", 50)
            await client.turn_on()
            if hasattr(client, 'set_brightness'):
                await client.set_brightness(brightness)
            else:
                self.logger.warning(f"Light {light_name} does not support brightness control")
        
        # Set color temperature if specified
        if "color_temp" in cmd and hasattr(client, 'set_color_temp'):
            try:
                await client.set_color_temp(cmd["color_temp"])
            except Exception as e:
                self.logger.warning(f"Failed to set color temperature for {light_name}: {e}")
    
    def _create_execution_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the execution results."""
        total_lights_affected = 0
        actions_performed = {}
        
        for result in results:
            if result.get("success") and isinstance(result, dict):
                total_lights_affected += result.get("lights_controlled", 0)
                action = result.get("command", {}).get("action")
                if action:
                    actions_performed[action] = actions_performed.get(action, 0) + 1
        
        return {
            "total_lights_affected": total_lights_affected,
            "actions_performed": actions_performed,
            "execution_status": "completed"
        }
    
    def _get_affected_lights(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Get list of all lights that would be affected by the commands."""
        affected = set()
        
        for cmd in commands:
            if "light_name" in cmd:
                affected.add(cmd["light_name"])
            elif "room" in cmd:
                room_lights = self.light_config["rooms"].get(cmd["room"], [])
                affected.update(room_lights)
        
        return sorted(list(affected))
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except:
            return "unknown"