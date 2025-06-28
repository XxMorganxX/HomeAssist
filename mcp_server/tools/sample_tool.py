"""
Sample MCP tool demonstrating proper tool structure.
This tool shows how to inherit from BaseTool and implement required methods.
"""

import asyncio
from typing import Dict, Any
from mcp_server.base_tool import BaseTool


class SampleTool(BaseTool):
    """Sample tool for demonstration purposes."""
    
    name = "sample_tool"
    description = "A sample tool that demonstrates proper MCP tool structure"
    version = "1.0.0"
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A message to process"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of times to repeat the message",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["message"],
            "description": self.description
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the sample tool."""
        message = params.get("message", "")
        count = params.get("count", 1)
        
        # Log the execution
        self.log_info(f"Processing message: '{message}' x{count}")
        
        # Process the message
        result = []
        for i in range(count):
            result.append(f"{i+1}: {message}")
        
        # Return the result
        return {
            "processed_message": message,
            "count": count,
            "results": result,
            "total_length": sum(len(r) for r in result)
        }


# Alternative: You can also define the tool class with a different name
# and set the name attribute, like this:
class EchoTool(BaseTool):
    """Simple echo tool that returns the input."""
    
    name = "echo"
    description = "Echoes back the input message with optional formatting"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                },
                "uppercase": {
                    "type": "boolean",
                    "description": "Whether to return text in uppercase",
                    "default": False
                }
            },
            "required": ["text"],
            "description": self.description
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        uppercase = params.get("uppercase", False)
        
        result_text = text.upper() if uppercase else text
        
        self.log_info(f"Echoing: '{text}' -> '{result_text}'")
        
        return {
            "original": text,
            "result": result_text,
            "was_uppercased": uppercase
        } 