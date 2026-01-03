"""
System Information Tool using BaseTool.

This tool provides information about the assistant's internal structure,
capabilities, and configuration by reading project documentation.
"""

from mcp_server.config import LOG_TOOLS
from mcp_server.base_tool import BaseTool
from typing import Dict, Any
from pathlib import Path


class SystemInfoTool(BaseTool):
    """Tool for providing information about the assistant's internal structure and capabilities."""
    
    name = "system_info"
    description = (
        "Get information about how this voice assistant works internally. "
        "Use this tool when the user asks about the assistant's architecture, capabilities, "
        "how it was built, project structure, configuration options, available tools, "
        "memory systems, troubleshooting, or any questions about the assistant itself. "
        "Returns documentation about the assistant's design and features."
    )
    version = "1.0.0"
    
    def __init__(self):
        """Initialize the system info tool."""
        super().__init__()
        
        # Get project root (two levels up from tools directory)
        self.project_root = Path(__file__).parent.parent.parent
        
        # README file paths
        self.readme_paths = {
            "main": self.project_root / "README.md",
            "framework": self.project_root / "assistant_framework" / "README.md",
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": (
                        "Optional: specific section of documentation to retrieve. "
                        "Options: 'overview' (general project info and quick start), "
                        "'structure' (project structure and file organization), "
                        "'tools' (available MCP tools and smart home capabilities), "
                        "'memory' (memory systems - persistent and vector), "
                        "'config' (configuration options and settings), "
                        "'framework' (detailed assistant framework architecture), "
                        "'troubleshooting' (common issues and solutions), "
                        "'all' (complete documentation). "
                        "Default is 'all' which returns comprehensive information."
                    ),
                    "enum": ["overview", "structure", "tools", "memory", "config", "framework", "troubleshooting", "all"],
                    "default": "all"
                }
            },
            "required": []
        }
    
    def _read_readme(self, path: Path) -> str:
        """Read a README file and return its contents."""
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
            else:
                return f"[File not found: {path}]"
        except Exception as e:
            return f"[Error reading {path}: {str(e)}]"
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content."""
        lines = content.split('\n')
        result_lines = []
        in_section = False
        current_level = 0
        
        # Define section mappings
        section_headers = {
            "overview": ["# HomeAssist", "## Quick Start"],
            "structure": ["## Project Structure", "## Architecture Overview"],
            "tools": ["## MCP Tools", "## Supported Providers"],
            "memory": ["## Memory Systems"],
            "config": ["## Configuration"],
            "troubleshooting": ["## Troubleshooting", "## Error Handling"],
        }
        
        target_headers = section_headers.get(section_name, [])
        
        for line in lines:
            # Check if this is a header
            if line.startswith('#'):
                # Count header level
                level = len(line) - len(line.lstrip('#'))
                
                # Check if this is our target section
                for header in target_headers:
                    if line.strip().startswith(header.lstrip('#').strip()) or header in line:
                        in_section = True
                        current_level = level
                        result_lines.append(line)
                        break
                else:
                    # Different header - check if we should exit the section
                    if in_section and level <= current_level:
                        in_section = False
                    elif in_section:
                        result_lines.append(line)
            elif in_section:
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip()
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the system info tool.
        
        Args:
            params: Tool parameters (optional section filter)
            
        Returns:
            Dictionary containing system documentation
        """
        try:
            section = params.get("section", "all")
            
            if LOG_TOOLS:
                self.logger.info("Executing Tool: SystemInfo -- section=%s", section)
            
            # Read both README files
            main_readme = self._read_readme(self.readme_paths["main"])
            framework_readme = self._read_readme(self.readme_paths["framework"])
            
            if section == "all":
                # Return everything
                return {
                    "success": True,
                    "message": "Complete system documentation retrieved",
                    "main_documentation": main_readme,
                    "framework_documentation": framework_readme,
                    "summary": (
                        "HomeAssist V2 is a fully local, always-listening voice assistant. "
                        "It uses wake word detection, real-time transcription, GPT-4 for responses, "
                        "and text-to-speech output. It can control smart home devices, play Spotify, "
                        "check calendars, get weather, and more. It has persistent memory for facts "
                        "and preferences, plus vector memory for semantic search of past conversations."
                    )
                }
            
            elif section == "framework":
                # Return framework-specific docs
                return {
                    "success": True,
                    "message": "Framework architecture documentation retrieved",
                    "framework_documentation": framework_readme,
                    "summary": (
                        "The assistant framework uses a provider-based architecture with interfaces "
                        "for transcription, response generation, TTS, context management, and wake word detection. "
                        "Providers can be swapped by changing configuration."
                    )
                }
            
            else:
                # Extract specific section from main README
                extracted = self._extract_section(main_readme, section)
                
                # Also check framework README for additional context
                framework_extracted = self._extract_section(framework_readme, section)
                
                if not extracted and not framework_extracted:
                    # Fallback to returning all if section not found
                    return {
                        "success": True,
                        "message": f"Section '{section}' not found as distinct section, returning relevant documentation",
                        "main_documentation": main_readme,
                        "note": "The requested section may be integrated throughout the documentation"
                    }
                
                result = {
                    "success": True,
                    "message": f"Documentation section '{section}' retrieved",
                    "section": section,
                }
                
                if extracted:
                    result["main_documentation"] = extracted
                if framework_extracted:
                    result["framework_documentation"] = framework_extracted
                    
                return result
            
        except Exception as e:
            self.logger.error("Failed to retrieve system info: %s", e)
            return {
                "success": False,
                "error": f"Failed to retrieve system documentation: {str(e)}"
            }

