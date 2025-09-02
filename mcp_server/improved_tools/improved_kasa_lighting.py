"""
Improved Kasa Lighting Tool

Single tool that supports two interaction styles:
1) Direct control of individual lights (on/off/toggle/dim/brighten/color temp)
2) Scene application across a room or selected lights

Backed by core.kasa_lighting_client.KasaLightingClient for robust control and debugging.
"""

import sys
sys.path.insert(0, '../..')

from typing import Dict, Any, List, Optional

from mcp_server.improved_base_tool import ImprovedBaseTool

try:
    from core.kasa_lighting_client import KasaLightingClient
except Exception as e:
    # Fallback import path when running from MCP server root
    import os
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from core.kasa_lighting_client import KasaLightingClient  # type: ignore

try:
    import config  # type: ignore
    LIGHT_ROOM_MAPPING = getattr(config, 'LIGHT_ROOM_MAPPING', {})
except Exception:
    LIGHT_ROOM_MAPPING = {"lights": {}, "rooms": {}}


class ImprovedKasaLightingTool(ImprovedBaseTool):
    """Control Kasa smart lights directly or by scene with clear, validated parameters."""

    name = "improved_kasa_lighting"
    description = (
        "Control Kasa smart lights in two ways: "
        "(1) Direct device control (on/off/toggle/dim/brighten/color temperature) for a single light, or "
        "(2) Apply a lighting scene across a room or selected lights. "
        "When the user mentions a specific light like 'Light 1', use direct control. "
        "When they ask for 'movie' or 'work' mood, use scene mode."
    )
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.client = KasaLightingClient()
        self.light_config = LIGHT_ROOM_MAPPING or {"lights": {}, "rooms": {}}
        self.available_lights: List[str] = list(self.light_config.get("lights", {}).keys())
        self.available_rooms: List[str] = list(self.light_config.get("rooms", {}).keys())
        self.available_scenes: List[str] = ["movie", "work", "mood", "reading", "party", "relax"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "interaction": {
                    "type": "string",
                    "enum": ["direct", "scene"],
                    "description": (
                        "Choose 'direct' to control a single light with actions like on/off/dim. "
                        "Choose 'scene' to apply a preset across a room or selected lights."
                    ),
                },
                "light_name": {
                    "type": "string",
                    "enum": self.available_lights,
                    "description": (
                        f"Specific light to control (direct mode). Available: {', '.join(self.available_lights) if self.available_lights else 'none configured'}. "
                        "Use only one target: either light_name or room."
                    ),
                },
                "room": {
                    "type": "string",
                    "enum": self.available_rooms,
                    "description": (
                        f"Room name for scene target or to resolve lights (optional). Available rooms: {', '.join(self.available_rooms) if self.available_rooms else 'none configured'}."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": (
                        "Direct mode action: 'on' or 'off' only."
                    ),
                },
                "scene_name": {
                    "type": "string",
                    "enum": self.available_scenes,
                    "description": (
                        "Scene to apply (scene mode). Presets: movie, work, mood, reading, party, relax. "
                        "Scenes set reasonable brightness and color temperature if supported."
                    ),
                },
                "light_names": {
                    "type": "array",
                    "items": {"type": "string", "enum": self.available_lights},
                    "description": (
                        "Optional list of specific lights for scene mode. If omitted, the 'room' field is used."
                    ),
                },
            },
            "required": ["interaction"],
            "allOf": [
                {
                    "if": {"properties": {"interaction": {"const": "direct"}}},
                    "then": {
                        "required": ["action", "light_name"],
                    },
                },
                {
                    "if": {"properties": {"interaction": {"const": "scene"}}},
                    "then": {
                        "required": ["scene_name"],
                        "anyOf": [
                            {"required": ["room"]},
                            {"required": ["light_names"]},
                        ],
                    },
                },
            ],
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            interaction: str = params.get("interaction", "").strip().lower()
            if interaction not in ("direct", "scene"):
                return {"success": False, "error": "interaction must be 'direct' or 'scene'"}

            if interaction == "direct":
                action = (params.get("action") or "").strip().lower()
                light_name = params.get("light_name")
                if not action or not light_name:
                    return {"success": False, "error": "direct mode requires 'action' and 'light_name'"}

                if action == "on":
                    result = self.client.turn_on(light_name)
                elif action == "off":
                    result = self.client.turn_off(light_name)
                else:
                    return {"success": False, "error": f"Unknown action '{action}'. Allowed: on, off"}

                return {"success": bool(result.get("success")), "result": result}

            # scene interaction
            scene_name = params.get("scene_name")
            room = params.get("room")
            light_names = params.get("light_names")
            # Only on/off supported: for a scene request, turn on all target lights
            targets: List[str] = []
            if light_names:
                targets = [n for n in light_names if n in self.light_config.get('lights', {})]
            elif room:
                targets = list(self.light_config.get('rooms', {}).get(room, []))
            if not targets:
                return {"success": False, "error": "No target lights resolved for scene"}
            per_light = []
            for name in targets:
                per_light.append(self.client.turn_on(name))
            overall = any(r.get('success') for r in per_light)
            return {"success": overall, "result": {"scene": scene_name, "targets": targets, "results": per_light}}

        except Exception as e:
            return {"success": False, "error": f"Lighting control failed: {e}"}

