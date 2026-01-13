"""
Tool Result Formatter for structured, readable console output.

Formats MCP tool calls and results in a clean, easy-to-read format
with proper indentation, visual separators, and smart truncation.
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime

# ANSI color codes (works in most terminals)
class Colors:
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# Configuration
MAX_VALUE_LENGTH = 200  # Max length for individual values before truncation
MAX_ARRAY_ITEMS = 10    # Max items to show in arrays before truncating
INDENT = "  "           # Indentation string


def _truncate(value: str, max_len: int = MAX_VALUE_LENGTH) -> str:
    """Truncate string with ellipsis if too long."""
    if len(value) <= max_len:
        return value
    return value[:max_len - 3] + "..."


def _format_value(value: Any, depth: int = 0) -> str:
    """Format a single value with proper indentation and type handling."""
    indent = INDENT * depth
    next_indent = INDENT * (depth + 1)
    
    if value is None:
        return f"{Colors.DIM}null{Colors.RESET}"
    
    if isinstance(value, bool):
        color = Colors.GREEN if value else Colors.RED
        return f"{color}{str(value).lower()}{Colors.RESET}"
    
    if isinstance(value, (int, float)):
        return f"{Colors.CYAN}{value}{Colors.RESET}"
    
    if isinstance(value, str):
        # Check if it's a long string
        if len(value) > MAX_VALUE_LENGTH:
            return f'"{_truncate(value)}"'
        return f'"{value}"'
    
    if isinstance(value, list):
        if not value:
            return "[]"
        
        if len(value) > MAX_ARRAY_ITEMS:
            items = value[:MAX_ARRAY_ITEMS]
            suffix = f"\n{next_indent}{Colors.DIM}... and {len(value) - MAX_ARRAY_ITEMS} more items{Colors.RESET}"
        else:
            items = value
            suffix = ""
        
        # For simple arrays, show inline
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in items):
            if len(str(items)) < 80:
                return str(items)
        
        lines = []
        for item in items:
            lines.append(f"{next_indent}{_format_value(item, depth + 1)}")
        
        return "[\n" + ",\n".join(lines) + suffix + f"\n{indent}]"
    
    if isinstance(value, dict):
        if not value:
            return "{}"
        
        lines = []
        for k, v in value.items():
            formatted_v = _format_value(v, depth + 1)
            lines.append(f"{next_indent}{Colors.BLUE}{k}{Colors.RESET}: {formatted_v}")
        
        return "{\n" + ",\n".join(lines) + f"\n{indent}}}"
    
    return str(value)


def format_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Format a tool call for console output.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        
    Returns:
        Formatted string for console display
    """
    lines = [
        "",
        f"{Colors.BOLD}┌─ 🔧 TOOL CALL ─────────────────────────────────{Colors.RESET}",
        f"{Colors.BOLD}│{Colors.RESET} Tool: {Colors.YELLOW}{tool_name}{Colors.RESET}",
    ]
    
    if arguments:
        lines.append(f"{Colors.BOLD}│{Colors.RESET} Arguments:")
        for key, value in arguments.items():
            formatted = _format_value(value, depth=1)
            # Handle multiline values
            if '\n' in formatted:
                lines.append(f"{Colors.BOLD}│{Colors.RESET}   {Colors.BLUE}{key}{Colors.RESET}:")
                for line in formatted.split('\n'):
                    lines.append(f"{Colors.BOLD}│{Colors.RESET}     {line}")
            else:
                lines.append(f"{Colors.BOLD}│{Colors.RESET}   {Colors.BLUE}{key}{Colors.RESET}: {formatted}")
    else:
        lines.append(f"{Colors.BOLD}│{Colors.RESET} Arguments: {Colors.DIM}(none){Colors.RESET}")
    
    lines.append(f"{Colors.BOLD}└────────────────────────────────────────────────{Colors.RESET}")
    
    return "\n".join(lines)


def format_tool_result(
    tool_name: str,
    result: Any,
    success: Optional[bool] = None,
    execution_time_ms: Optional[float] = None
) -> str:
    """
    Format a tool result for console output.
    
    Args:
        tool_name: Name of the tool that was called
        result: Result data (can be dict, str, or any JSON-serializable type)
        success: Whether the tool succeeded (auto-detected if None)
        execution_time_ms: Execution time in milliseconds
        
    Returns:
        Formatted string for console display
    """
    # Parse result if it's a string (might be JSON)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Auto-detect success from result
    if success is None and isinstance(result, dict):
        success = result.get('success', True)
        if result.get('error'):
            success = False
    
    # Choose status indicator
    if success is False:
        status = f"{Colors.RED}✗ FAILED{Colors.RESET}"
        border_color = Colors.RED
    else:
        status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}"
        border_color = Colors.GREEN
    
    # Build header
    lines = [
        "",
        f"{border_color}{Colors.BOLD}┌─ 📥 TOOL RESULT ───────────────────────────────{Colors.RESET}",
        f"{border_color}{Colors.BOLD}│{Colors.RESET} Tool: {Colors.YELLOW}{tool_name}{Colors.RESET}  Status: {status}",
    ]
    
    if execution_time_ms is not None:
        lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET} Time: {Colors.CYAN}{execution_time_ms:.1f}ms{Colors.RESET}")
    
    lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}")
    
    # Format the result content
    if isinstance(result, dict):
        # Extract and display key fields first
        priority_fields = ['success', 'error', 'message', 'location', 'mode']
        other_fields = {}
        
        for key, value in result.items():
            if key in priority_fields and value is not None:
                formatted = _format_value(value, depth=0)
                if key == 'error':
                    lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.RED}{key}{Colors.RESET}: {formatted}")
                elif key == 'success':
                    continue  # Already shown in status
                else:
                    lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.BLUE}{key}{Colors.RESET}: {formatted}")
            elif key not in priority_fields:
                other_fields[key] = value
        
        # Show remaining fields
        if other_fields:
            lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}")
            lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}── Data ──{Colors.RESET}")
            
            for key, value in other_fields.items():
                formatted = _format_value(value, depth=1)
                if '\n' in formatted:
                    lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}   {Colors.BLUE}{key}{Colors.RESET}:")
                    for line in formatted.split('\n'):
                        lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}     {line}")
                else:
                    # Truncate long single-line values
                    if len(formatted) > 100:
                        formatted = formatted[:97] + "..."
                    lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}   {Colors.BLUE}{key}{Colors.RESET}: {formatted}")
    
    elif isinstance(result, str):
        # Plain text result
        if len(result) > 500:
            display_result = result[:500] + f"\n{Colors.DIM}... ({len(result) - 500} more chars){Colors.RESET}"
        else:
            display_result = result
        
        for line in display_result.split('\n'):
            lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}   {line}")
    
    else:
        lines.append(f"{border_color}{Colors.BOLD}│{Colors.RESET}   {_format_value(result)}")
    
    lines.append(f"{border_color}{Colors.BOLD}└────────────────────────────────────────────────{Colors.RESET}")
    lines.append("")
    
    return "\n".join(lines)


def format_tool_error(tool_name: str, error: str, traceback_str: Optional[str] = None) -> str:
    """
    Format a tool error for console output.
    
    Args:
        tool_name: Name of the tool that failed
        error: Error message
        traceback_str: Optional traceback string
        
    Returns:
        Formatted string for console display
    """
    lines = [
        "",
        f"{Colors.RED}{Colors.BOLD}┌─ ❌ TOOL ERROR ─────────────────────────────────{Colors.RESET}",
        f"{Colors.RED}{Colors.BOLD}│{Colors.RESET} Tool: {Colors.YELLOW}{tool_name}{Colors.RESET}",
        f"{Colors.RED}{Colors.BOLD}│{Colors.RESET}",
        f"{Colors.RED}{Colors.BOLD}│{Colors.RESET} {Colors.RED}Error: {error}{Colors.RESET}",
    ]
    
    if traceback_str:
        lines.append(f"{Colors.RED}{Colors.BOLD}│{Colors.RESET}")
        lines.append(f"{Colors.RED}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}Traceback:{Colors.RESET}")
        for line in traceback_str.strip().split('\n')[-5:]:  # Show last 5 lines of traceback
            lines.append(f"{Colors.RED}{Colors.BOLD}│{Colors.RESET}   {Colors.DIM}{line}{Colors.RESET}")
    
    lines.append(f"{Colors.RED}{Colors.BOLD}└────────────────────────────────────────────────{Colors.RESET}")
    lines.append("")
    
    return "\n".join(lines)


# Convenience function for quick summary
def format_tool_summary(tool_name: str, success: bool, brief_result: str = "") -> str:
    """
    Format a brief one-line tool summary.
    
    Args:
        tool_name: Name of the tool
        success: Whether it succeeded
        brief_result: Optional brief result description
        
    Returns:
        Single-line summary
    """
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
    result_part = f" → {brief_result}" if brief_result else ""
    return f"{icon} {Colors.YELLOW}{tool_name}{Colors.RESET}{result_part}"

