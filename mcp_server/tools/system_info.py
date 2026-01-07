"""
System Information Tool using BaseTool.

This tool provides information about the assistant's internal structure,
capabilities, and configuration by reading the developer documentation.
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
    version = "2.0.0"
    
    # Section mappings to DEV_README.md headers
    SECTION_HEADERS = {
        "overview": ["## Architecture Overview", "## Table of Contents"],
        "architecture": ["## Architecture Overview", "### Directory Structure", "### Execution Flow"],
        "providers": ["## Provider Pattern"],
        "orchestrator": ["## Orchestrator & State Machine"],
        "audio": ["## Audio Pipeline"],
        "tools": ["## MCP Tool System"],
        "memory": ["## Memory Systems"],
        "config": ["## Configuration System"],
        "models": ["## Data Models"],
        "scheduled": ["## Scheduled Jobs"],
        "errors": ["## Error Handling"],
        "development": ["## Common Development Tasks"],
        "performance": ["## Performance Considerations", "## Key Implementation Details"],
    }
    
    def __init__(self):
        """Initialize the system info tool."""
        super().__init__()
        
        # Get project root (two levels up from tools directory)
        self.project_root = Path(__file__).parent.parent.parent
        
        # DEV_README path
        self.dev_readme_path = self.project_root / "DEV_README.md"
    
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
                        "Options: 'overview' (project summary and table of contents), "
                        "'architecture' (directory structure and execution flow), "
                        "'providers' (interface system and factory pattern), "
                        "'orchestrator' (state machine and component lifecycle), "
                        "'audio' (transcription, TTS, barge-in detection), "
                        "'tools' (MCP tool system and how to add tools), "
                        "'memory' (persistent memory, vector memory, conversation context), "
                        "'config' (configuration system and presets), "
                        "'models' (data structures and types), "
                        "'scheduled' (background jobs and briefings), "
                        "'errors' (error handling and recovery), "
                        "'development' (running, testing, debugging tips), "
                        "'performance' (latency optimization, memory usage), "
                        "'all' (complete documentation). "
                        "Default is 'overview' for general questions, use 'all' only if specifically requested."
                    ),
                    "enum": [
                        "overview", "architecture", "providers", "orchestrator", "audio",
                        "tools", "memory", "config", "models", "scheduled", "errors",
                        "development", "performance", "all"
                    ],
                    "default": "overview"
                }
            },
            "required": []
        }
    
    def _read_dev_readme(self) -> str:
        """Read the DEV_README file and return its contents."""
        try:
            if self.dev_readme_path.exists():
                return self.dev_readme_path.read_text(encoding="utf-8")
            else:
                return f"[DEV_README.md not found at {self.dev_readme_path}]"
        except Exception as e:
            return f"[Error reading DEV_README.md: {str(e)}]"
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content."""
        lines = content.split('\n')
        result_lines = []
        in_section = False
        current_level = 0
        
        target_headers = self.SECTION_HEADERS.get(section_name, [])
        
        for line in lines:
            # Check if this is a header
            if line.startswith('#'):
                # Count header level
                level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                # Check if this is our target section
                matched = False
                for header in target_headers:
                    header_clean = header.lstrip('#').strip()
                    if header_text == header_clean or header_text.startswith(header_clean):
                        matched = True
                        break
                
                if matched:
                    in_section = True
                    current_level = level
                    result_lines.append(line)
                elif in_section:
                    # Different header - check if we should exit the section
                    if level <= current_level:
                        # Same or higher level header, stop collecting
                        in_section = False
                    else:
                        # Subsection, keep collecting
                        result_lines.append(line)
            elif in_section:
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip()
    
    def _get_quick_summary(self) -> str:
        """Get a quick summary of the assistant."""
        return (
            "HomeAssist V2 is a fully local, always-listening voice assistant built in Python. "
            "Key components:\n"
            "• Wake word detection (OpenWakeWord, process-isolated)\n"
            "• Real-time transcription (AssemblyAI or OpenAI Whisper)\n"
            "• GPT-4o Realtime for responses with MCP tool calling\n"
            "• Text-to-speech (Piper, Google, Chatterbox)\n"
            "• Three-tier memory: conversation context, persistent facts, vector search\n"
            "• Smart home tools: Spotify, Kasa lights, Google Calendar, weather, SMS, etc.\n"
            "• Scheduled briefings: email summaries, news digests, calendar reminders\n\n"
            "The codebase uses a provider pattern with swappable implementations for each component."
        )
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the system info tool.
        
        Args:
            params: Tool parameters (optional section filter)
            
        Returns:
            Dictionary containing system documentation
        """
        try:
            section = params.get("section", "overview")
            
            if LOG_TOOLS:
                self.logger.info("Executing Tool: SystemInfo -- section=%s", section)
            
            # Read DEV_README
            dev_readme = self._read_dev_readme()
            
            if "[DEV_README.md not found" in dev_readme or "[Error reading" in dev_readme:
                return {
                    "success": False,
                    "error": dev_readme,
                    "fallback_summary": self._get_quick_summary()
                }
            
            if section == "all":
                # Return complete documentation
                return {
                    "success": True,
                    "message": "Complete developer documentation retrieved",
                    "documentation": dev_readme,
                    "summary": self._get_quick_summary()
                }
            
            elif section == "overview":
                # Return architecture overview with quick summary
                extracted = self._extract_section(dev_readme, "overview")
                architecture = self._extract_section(dev_readme, "architecture")
                
                return {
                    "success": True,
                    "message": "Architecture overview retrieved",
                    "summary": self._get_quick_summary(),
                    "architecture_overview": architecture if architecture else extracted,
                    "note": "Use specific section names for detailed info: providers, orchestrator, audio, tools, memory, config, etc."
                }
            
            else:
                # Extract specific section
                extracted = self._extract_section(dev_readme, section)
                
                if not extracted:
                    # Section not found, return overview with note
                    return {
                        "success": True,
                        "message": f"Section '{section}' not found as distinct section",
                        "summary": self._get_quick_summary(),
                        "available_sections": list(self.SECTION_HEADERS.keys()),
                        "note": "Try 'all' for complete documentation"
                    }
                
                return {
                    "success": True,
                    "message": f"Documentation section '{section}' retrieved",
                    "section": section,
                    "content": extracted
                }
            
        except Exception as e:
            self.logger.error("Failed to retrieve system info: %s", e)
            return {
                "success": False,
                "error": f"Failed to retrieve system documentation: {str(e)}",
                "fallback_summary": self._get_quick_summary()
            }
