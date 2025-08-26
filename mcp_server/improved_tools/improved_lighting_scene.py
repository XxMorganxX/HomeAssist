"""
Improved Lighting Scene Tool using ImprovedBaseTool.

This tool controls multiple lights based on predefined scenes with enhanced
parameter descriptions and comprehensive scene management.
"""

import asyncio
try:
    from kasa import Discover  # type: ignore
except Exception:
    try:
        from Kasa import Discover  # type: ignore
    except Exception:
        Discover = None  # type: ignore
import sys
sys.path.insert(0, '../..')

try:
    from config import LIGHT_ONE_IP
except ImportError:
    # Fallback for MCP server context
    LIGHT_ONE_IP = "192.168.1.100"
from mcp_server.improved_base_tool import ImprovedBaseTool
from typing import Dict, Any, Literal
from core.state_management.statemanager import StateManager


class ImprovedLightingSceneTool(ImprovedBaseTool):
    """Enhanced tool for controlling multiple lights with predefined scenes."""
    
    name = "improved_lighting_scene"
    description = "Control multiple smart lights simultaneously using predefined lighting scenes. Each scene automatically configures all relevant lights for specific activities or moods. Use this when users want to set an overall lighting atmosphere rather than controlling individual lights."
    version = "1.0.1"
    
    # Enhanced scene definitions with detailed descriptions
    SCENES = {
        "movie": {
            "description": "Dim lighting perfect for watching movies or TV shows. Turns off main lights to reduce screen glare while maintaining some ambient lighting.",
            "lights": ["Light 1"],
            "actions": ["off"],
            "mood": "cinematic"
        },
        "party": {
            "description": "Bright, energetic lighting for social gatherings and celebrations. All lights at full brightness to create a lively atmosphere.",
            "lights": ["Light 1", "Light 2"],
            "actions": ["on", "on"],
            "mood": "energetic"
        },
        "mood": {
            "description": "Soft, warm lighting for relaxation and intimate settings. Creates a cozy, romantic atmosphere with dimmed lights.",
            "lights": ["Light 1"],
            "actions": ["dim"],
            "brightness": 25,
            "mood": "romantic"
        },
        "work": {
            "description": "Bright, focused lighting optimal for work, reading, or detailed tasks. Provides clear, comfortable illumination to reduce eye strain.",
            "lights": ["Light 1", "Light 2"],
            "actions": ["on", "on"],
            "brightness": 90,
            "mood": "productive"
        },
        "sleep": {
            "description": "Very dim, warm lighting for bedtime preparation. Minimal lighting that won't disrupt circadian rhythms.",
            "lights": ["Light 2"],
            "actions": ["dim"],
            "brightness": 10,
            "mood": "restful"
        },
        "all_on": {
            "description": "Turn on all lights at full brightness. Maximum illumination for cleaning, security, or when full visibility is needed.",
            "lights": ["Light 1", "Light 2"],
            "actions": ["on", "on"],
            "mood": "utilitarian"
        },
        "all_off": {
            "description": "Turn off all lights completely. Complete darkness for sleep, leaving the house, or energy conservation.",
            "lights": ["Light 1", "Light 2"], 
            "actions": ["off", "off"],
            "mood": "off"
        }
    }
    
    def __init__(self):
        """Initialize the improved lighting scene tool."""
        super().__init__()
        self.clients = {}
        self.state_manager = StateManager()
        
        # Extract available scenes for validation
        self.available_scenes = list(self.SCENES.keys())
        
    async def _init_client(self, light_name: str):
        """Initialize a Kasa client for a specific light."""
        if light_name not in self.clients:
            try:
                if Discover is None:
                    raise RuntimeError(
                        "python-kasa not available. Install with: pip install python-kasa"
                    )
                if light_name == "Light 1":
                    self.clients[light_name] = await Discover.discover_single(
                        LIGHT_ONE_IP,
                        username="morgannstuart@gmail.com",
                        password="1234567890"
                    )
                elif light_name == "Light 2":
                    # Add Light 2 configuration when available
                    self.logger.warning(f"Light 2 configuration not yet implemented")
                else:
                    self.logger.error(f"Unknown light: {light_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize client for {light_name}: {e}")
                raise
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with detailed scene descriptions.
        
        Returns:
            Comprehensive JSON schema dictionary
        """
        # Create enhanced scene descriptions for the enum
        scene_descriptions = {}
        for scene_name, scene_config in self.SCENES.items():
            lights_involved = ", ".join(scene_config["lights"])
            scene_descriptions[scene_name] = f"{scene_config['description']} (Controls: {lights_involved})"
        
        return {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "description": f"The lighting scene to activate. Each scene controls multiple lights with predefined settings: {', '.join([f'{k} - {v['description']}' for k, v in self.SCENES.items()])}",
                    "enum": self.available_scenes
                },
                "override_brightness": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Optional brightness override (1-100) to customize the scene's default brightness. Only applies to scenes that turn lights on. Use this when user wants a specific brightness level different from the scene's default."
                },
                "transition_duration": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Duration in seconds for gradual scene transition. Creates smooth lighting changes instead of instant switching. Useful for mood scenes or when lights are currently on."
                },
                "save_current_as_custom": {
                    "type": "boolean",
                    "description": "Whether to save the current lighting state before applying the new scene. Allows restoration of previous lighting configuration later.",
                    "default": False
                }
            },
            "required": ["scene"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the lighting scene control.
        
        Args:
            params: Tool parameters containing scene and optional overrides
            
        Returns:
            Dictionary containing execution results
        """
        try:
            scene_name = params.get("scene")
            
            if not scene_name:
                return {
                    "success": False,
                    "error": "Missing required parameter: scene",
                    "available_scenes": self.available_scenes,
                    "scene_descriptions": {k: v["description"] for k, v in self.SCENES.items()}
                }
            
            if scene_name not in self.SCENES:
                return {
                    "success": False,
                    "error": f"Unknown scene '{scene_name}'",
                    "available_scenes": self.available_scenes,
                    "scene_descriptions": {k: v["description"] for k, v in self.SCENES.items()}
                }
            
            # Get scene configuration
            scene_config = self.SCENES[scene_name]
            override_brightness = params.get("override_brightness")
            transition_duration = params.get("transition_duration", 0)
            save_current = params.get("save_current_as_custom", False)
            
            # Save current state if requested
            if save_current:
                self._save_current_state()
            
            # Execute scene asynchronously
            result = asyncio.run(self._execute_scene_async(
                scene_config, 
                override_brightness, 
                transition_duration
            ))
            
            # Update state manager with current scene
            try:
                self.state_manager.set("lighting_scene", scene_name)
            except Exception as e:
                self.logger.warning(f"Failed to update state manager: {e}")
            
            # Enhance result with scene information
            result.update({
                "scene_name": scene_name,
                "scene_description": scene_config["description"],
                "scene_mood": scene_config.get("mood", "unknown"),
                "lights_controlled": scene_config["lights"],
                "override_brightness": override_brightness,
                "transition_duration": transition_duration,
                "timestamp": self._get_current_timestamp()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing lighting scene: {e}")
            return {
                "success": False,
                "error": f"Scene execution failed: {str(e)}",
                "scene_name": params.get("scene", "unknown")
            }
    
    async def _execute_scene_async(self, scene_config: Dict[str, Any], override_brightness: int = None, transition_duration: int = 0) -> Dict[str, Any]:
        """Execute a lighting scene asynchronously."""
        lights = scene_config["lights"]
        actions = scene_config["actions"]
        default_brightness = scene_config.get("brightness", 100)
        
        # Use override brightness if provided
        brightness = override_brightness if override_brightness is not None else default_brightness
        
        results = []
        
        # Execute actions on each light
        for i, light_name in enumerate(lights):
            try:
                await self._init_client(light_name)
                client = self.clients.get(light_name)
                
                if not client:
                    results.append({
                        "light": light_name,
                        "success": False,
                        "error": "Failed to initialize client"
                    })
                    continue
                
                # Get the action for this light
                action = actions[i] if i < len(actions) else actions[0]
                
                # Perform the action
                await self._perform_scene_action(
                    client, 
                    light_name, 
                    action, 
                    brightness, 
                    transition_duration
                )
                
                results.append({
                    "light": light_name,
                    "success": True,
                    "action": action,
                    "brightness": brightness if action != "off" else 0
                })
                
            except Exception as e:
                results.append({
                    "light": light_name,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze results
        successful_lights = [r for r in results if r["success"]]
        failed_lights = [r for r in results if not r["success"]]
        
        return {
            "success": len(successful_lights) > 0,  # Success if at least one light worked
            "total_lights": len(lights),
            "successful_lights": len(successful_lights),
            "failed_lights": len(failed_lights),
            "light_results": results,
            "execution_summary": {
                "lights_controlled": [r["light"] for r in successful_lights],
                "brightness_applied": brightness,
                "transition_used": transition_duration > 0
            }
        }
    
    async def _perform_scene_action(self, client, light_name: str, action: str, brightness: int, transition_duration: int):
        """Perform a specific action on a light for scene control."""
        if action == "on":
            await client.turn_on()
            if hasattr(client, 'set_brightness') and brightness < 100:
                await client.set_brightness(brightness)
        elif action == "off":
            await client.turn_off()
        elif action == "dim":
            await client.turn_on()
            if hasattr(client, 'set_brightness'):
                await client.set_brightness(brightness)
            else:
                self.logger.warning(f"Light {light_name} does not support dimming")
        elif action == "toggle":
            if client.is_on:
                await client.turn_off()
            else:
                await client.turn_on()
                if hasattr(client, 'set_brightness') and brightness < 100:
                    await client.set_brightness(brightness)
        
        # Handle transition duration (basic implementation)
        if transition_duration > 0:
            # Simple transition simulation - more sophisticated implementations
            # would gradually change brightness over the duration
            await asyncio.sleep(min(transition_duration / 10, 1))
    
    def _save_current_state(self):
        """Save the current lighting state as a custom scene."""
        try:
            # This would save current light states to restore later
            # Implementation depends on available client state reading capabilities
            self.logger.info("Current lighting state saved (placeholder implementation)")
        except Exception as e:
            self.logger.warning(f"Failed to save current state: {e}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except:
            return "unknown"
    
    def get_available_scenes(self) -> Dict[str, Any]:
        """
        Get information about all available scenes.
        
        Returns:
            Dictionary with scene information
        """
        return {
            "available_scenes": self.available_scenes,
            "scene_details": {
                name: {
                    "description": config["description"],
                    "mood": config.get("mood", "unknown"),
                    "lights_controlled": config["lights"],
                    "brightness": config.get("brightness", "default")
                }
                for name, config in self.SCENES.items()
            },
            "total_scenes": len(self.SCENES)
        }