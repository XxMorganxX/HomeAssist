#!/usr/bin/env python3
"""
Display the status of all MCP tools (enabled/disabled).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.tool_registry import ToolRegistry
from mcp_server.tools_config import ENABLED_TOOLS, DISABLED_TOOLS, is_tool_enabled

def main():
    """Show status of all tools."""
    print("=" * 60)
    print("MCP TOOL STATUS")
    print("=" * 60)
    print()
    
    # Create registry instance
    registry = ToolRegistry()
    
    # Get all tools status
    all_tools = registry.get_all_tools_status()
    
    print("DISCOVERED TOOLS:")
    print("-" * 40)
    for tool_name, enabled in sorted(all_tools.items()):
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"  {tool_name:<25} {status}")
    
    print()
    print("CONFIGURATION:")
    print("-" * 40)
    print(f"Tools in ENABLED_TOOLS: {len(ENABLED_TOOLS)}")
    print(f"Tools in DISABLED_TOOLS: {len(DISABLED_TOOLS)}")
    
    # Load tools to see which ones actually register
    print()
    print("LOADING TOOLS...")
    print("-" * 40)
    discovered = registry.discover_tools()
    available = registry.get_available_tools()
    
    print(f"Modules discovered: {len(discovered)}")
    print(f"Tools registered: {len(available)}")
    
    if available:
        print()
        print("REGISTERED TOOLS:")
        print("-" * 40)
        for tool_name in sorted(available):
            print(f"  ✅ {tool_name}")
    
    # Show disabled tools that were skipped
    skipped = set(all_tools.keys()) - set(available)
    if skipped:
        print()
        print("DISABLED TOOLS (not loaded):")
        print("-" * 40)
        for tool_name in sorted(skipped):
            if not is_tool_enabled(tool_name):
                print(f"  ❌ {tool_name}")
    
    print()
    print("=" * 60)
    print("To enable/disable tools, edit:")
    print("  mcp_server/tools_config.py")
    print("=" * 60)

if __name__ == "__main__":
    main()