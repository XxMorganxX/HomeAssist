"""
Send SMS/iMessage Tool using BaseTool.

This tool sends text messages via iMessage on macOS.
NOTE: Only available on macOS - will not register on other platforms.
"""

import platform
import subprocess
from typing import Dict, Any
from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS

# Check if we're on macOS
IS_MACOS = platform.system() == "Darwin"


class SendSMSTool(BaseTool):
    """
    Tool to send SMS/iMessage notifications via macOS Messages app.
    
    This tool only works on macOS and requires:
    - Messages app to be set up with iMessage
    - The recipient must be reachable via iMessage
    """
    
    name = "send_sms"
    description = (
        "Send a text message (iMessage) to the user's phone. "
        "Use this when the user wants to send themselves a reminder, notification, or important message that the user wants available on their mobile device. "
        "Messages are sent to the configured phone number only, which is the user's phone number. "
        "Only works on macOS with iMessage configured."
    )
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SMS tool.
        
        Args:
            config: Configuration dictionary containing:
                - default_phone_number: Default recipient phone number
        """
        super().__init__()
        config = config or {}
        self.default_phone_number = config.get("default_phone_number", "")
        
        if not IS_MACOS:
            self.logger.warning("SendSMSTool is only available on macOS")
    
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
                "message": {
                    "type": "string",
                    "description": (
                        "The text message to send. Keep it concise and clear. "
                        "This will be sent as an iMessage to your configured phone number."
                    )
                }
            },
            "required": ["message"]
        }
        return schema
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the SMS tool to send a message.
        
        Args:
            params: Tool parameters containing:
                - message: The text message to send
                - phone_number: Optional recipient phone number
                
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
            message = params.get("message", "").strip()
            
            # Always use the configured default phone number (security restriction)
            phone_number = self.default_phone_number
            
            # Validate inputs
            if not message:
                return {
                    "success": False,
                    "error": "Message cannot be empty"
                }
            
            if not phone_number:
                return {
                    "success": False,
                    "error": "SMS tool not configured with a phone number"
                }
            
            # Validate phone number format (basic check)
            if not phone_number.startswith("+"):
                return {
                    "success": False,
                    "error": "Configured phone number invalid (must include country code)"
                }
            
            if LOG_TOOLS:
                self.logger.info(f"Executing Tool: SMS -- Sending to {phone_number[:6]}...")
            
            # Send the message
            success = self._send_imessage(message, phone_number)
            
            if success:
                return {
                    "success": True,
                    "message_sent": message,
                    "recipient": phone_number,
                    "method": "iMessage",
                    "info": "Message sent successfully via iMessage"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to send message via iMessage",
                    "recipient": phone_number,
                    "suggestion": "Ensure Messages app is configured and recipient uses iMessage"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing SMS tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "recipient": params.get("phone_number", "unknown")
            }
    
    def _send_imessage(self, message: str, phone_number: str) -> bool:
        """
        Send an iMessage via AppleScript.
        
        Args:
            message: The text message to send
            phone_number: Recipient phone number (with country code)
        
        Returns:
            True if message was sent successfully, False otherwise
        """
        import tempfile
        import os
        
        # Escape special characters for AppleScript string
        # In AppleScript, backslash and double-quote need escaping
        escaped_message = message.replace('\\', '\\\\').replace('"', '\\"')
        
        script = f'''tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{phone_number}" of targetService
    send "{escaped_message}" to targetBuddy
    delay 1
    quit
end tell
'''
        
        try:
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
                    timeout=30  # 30 second timeout
                )
                self.logger.info(f"Message sent to {phone_number}")
                return True
            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except Exception:
                    pass
                    
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to send message: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Message sending timed out")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message: {e}")
            return False


# Only export the tool class if we're on macOS
# This prevents registration on non-macOS platforms
if IS_MACOS:
    __all__ = ['SendSMSTool']
else:
    __all__ = []
