"""
Clipboard Read Tool using BaseTool.

This tool reads the current system clipboard contents, allowing the AI
to work with text the user has copied.
"""

import subprocess
import platform
from typing import Dict, Any

from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS


class ClipboardTool(BaseTool):
    """Tool for reading the system clipboard contents."""
    
    name = "read_clipboard"
    description = (
        "Read the current contents of the user's system clipboard. "
        "Use this when the user asks you to read, summarize, explain, "
        "or work with something they've copied. Returns the clipboard text content."
    )
    version = "1.0.0"
    
    def __init__(self):
        """Initialize the clipboard tool."""
        super().__init__()
        self._is_macos = platform.system() == "Darwin"
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool.
        
        Returns:
            JSON schema dictionary defining the tool's interface
        """
        return {
            "type": "object",
            "properties": {
                "max_length": {
                    "type": "integer",
                    "description": (
                        "Maximum number of characters to return from the clipboard. "
                        "Content longer than this will be truncated. Default is 8000 characters."
                    ),
                    "default": 8000
                }
            },
            "required": []
        }
    
    def _read_clipboard_macos(self) -> str:
        """Read clipboard contents on macOS using pbpaste."""
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("Clipboard read timed out")
        except FileNotFoundError:
            raise RuntimeError("pbpaste command not found - are you on macOS?")
        except Exception as e:
            raise RuntimeError(f"Failed to read clipboard: {e}")
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the clipboard read tool.
        
        Args:
            params: Tool parameters containing optional max_length
            
        Returns:
            Dictionary containing clipboard contents and metadata
        """
        try:
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Clipboard -- %s", params)
            
            # Check platform support
            if not self._is_macos:
                return {
                    "success": False,
                    "error": "Clipboard reading is only supported on macOS",
                    "platform": platform.system()
                }
            
            # Get max length parameter
            max_length = params.get("max_length", 8000)
            
            # Read clipboard
            content = self._read_clipboard_macos()
            
            # Handle empty clipboard
            if not content:
                return {
                    "success": True,
                    "content": "",
                    "is_empty": True,
                    "message": "The clipboard is empty - nothing has been copied."
                }
            
            # Calculate original length before truncation
            original_length = len(content)
            was_truncated = original_length > max_length
            
            # Truncate if needed
            if was_truncated:
                content = content[:max_length]
                truncation_message = f"Content truncated from {original_length} to {max_length} characters."
            else:
                truncation_message = None
            
            return {
                "success": True,
                "content": content,
                "is_empty": False,
                "character_count": len(content),
                "original_length": original_length,
                "was_truncated": was_truncated,
                "truncation_message": truncation_message
            }
            
        except Exception as e:
            self.logger.error("Failed to read clipboard: %s", e)
            return {
                "success": False,
                "error": f"Failed to read clipboard: {str(e)}"
            }
