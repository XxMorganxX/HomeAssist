"""
Cursor Composer Control Tool using BaseTool.

This tool sends prompts to Cursor's Composer interface via clipboard + AppleScript.
NOTE: Only available on macOS - will not register on other platforms.
"""

import platform
import subprocess
from typing import Dict, Any
from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS

# Check if we're on macOS
IS_MACOS = platform.system() == "Darwin"


class CursorComposerTool(BaseTool):
    """
    Tool to send coding prompts to Cursor's Composer interface.
    
    This tool only works on macOS and uses:
    - Clipboard to store the prompt (more reliable than typing)
    - AppleScript to activate Cursor and paste the prompt
    
    The prompt is pasted but NOT submitted - user reviews and presses Enter.
    """
    
    name = "cursor_composer"
    description = (
        "Send a coding request to Cursor's Composer for multi-file edits. "
        "Use when the user wants to write, refactor, or modify code in their codebase. "
        "The prompt will be typed into Cursor's Composer interface (Cmd+I) for review. "
        "Only works on macOS."
    )
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Cursor Composer tool.
        
        Args:
            config: Configuration dictionary (currently unused)
        """
        super().__init__()
        config = config or {}
        
        if not IS_MACOS:
            self.logger.warning("CursorComposerTool is only available on macOS")
    
    @staticmethod
    def is_available() -> bool:
        """Check if this tool is available on the current platform."""
        return IS_MACOS
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool.
        
        Returns:
            JSON schema dictionary
        """
        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The coding task to send to Cursor Composer. "
                        "Be specific about what files, functions, or changes are needed. "
                        "This will be pasted into Cursor's Composer for review before execution."
                    )
                }
            },
            "required": ["prompt"]
        }
        return schema
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool to send a prompt to Cursor's Composer.
        
        Args:
            params: Tool parameters containing:
                - prompt: The coding task/request to send
                
        Returns:
            Dictionary with success status and details
        """
        try:
            # Check platform
            if not IS_MACOS:
                return {
                    "success": False,
                    "error": "This tool is only available on macOS",
                    "platform": platform.system()
                }
            
            # Extract parameters
            prompt = params.get("prompt", "").strip()
            
            # Validate inputs
            if not prompt:
                return {
                    "success": False,
                    "error": "Prompt cannot be empty"
                }
            
            if LOG_TOOLS:
                self.logger.info(f"Executing Tool: Cursor Composer -- Sending prompt ({len(prompt)} chars)")
            
            # Send the prompt to Cursor
            success = self._send_to_cursor_composer(prompt)
            
            if success:
                return {
                    "success": True,
                    "prompt_sent": prompt,
                    "method": "Clipboard + AppleScript",
                    "info": "Prompt sent to Cursor Composer. Review and press Enter to execute."
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to send prompt to Cursor Composer",
                    "suggestion": "Ensure Cursor is installed and running"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing Cursor Composer tool: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _send_to_cursor_composer(self, prompt: str) -> bool:
        """
        Send a prompt to Cursor's Composer via clipboard and AppleScript.
        
        Args:
            prompt: The coding prompt to send
        
        Returns:
            True if prompt was sent successfully, False otherwise
        """
        import tempfile
        import os
        
        try:
            # Step 1: Copy prompt to clipboard using pbcopy
            process = subprocess.run(
                ["pbcopy"],
                input=prompt.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=5
            )
            
            # Step 2: Use AppleScript to activate Cursor, open Composer, and paste
            script = '''
tell application "Cursor" to activate
delay 0.3
tell application "System Events"
    -- Open Composer with Cmd+I
    keystroke "i" using command down
    delay 0.3
    -- Paste from clipboard with Cmd+V
    keystroke "v" using command down
end tell
'''
            
            # Write script to temp file to avoid shell escaping issues
            with tempfile.NamedTemporaryFile(mode='w', suffix='.applescript', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            try:
                subprocess.run(
                    ["osascript", script_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                self.logger.info("Prompt sent to Cursor Composer")
                return True
            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except Exception:
                    pass
                    
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to send to Cursor: {e.stderr if hasattr(e, 'stderr') else e}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Cursor control timed out")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending to Cursor: {e}")
            return False


# Only export the tool class if we're on macOS
# This prevents registration on non-macOS platforms
if IS_MACOS:
    __all__ = ['CursorComposerTool']
else:
    __all__ = []

