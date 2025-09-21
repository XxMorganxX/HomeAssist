"""
Improved Calendar Data Tool using ImprovedBaseTool.

This tool provides comprehensive Google Calendar access with enhanced parameter
descriptions, better command validation, and detailed event management.
"""

import sys
sys.path.insert(0, '../..')

from core.calendar_component import CalendarComponent
from mcp_server.improved_base_tool import ImprovedBaseTool
from typing import Dict, Any, List
from datetime import datetime
from config import LOG_TOOLS
try:
    import config
except ImportError:
    # Fallback for MCP server context
    config = None


class ImprovedCalendarTool(ImprovedBaseTool):
    """Enhanced tool for accessing and managing Google Calendar events with comprehensive command support."""
    
    name = "improved_calendar_data"
    description = "Access Google Calendar for reading and creating events across multiple users and calendars. Supports viewing upcoming events, daily summaries, and event creation. CRITICAL RESTRICTION: Only query ONE user per request - never multiple users simultaneously. For day summaries, user MUST explicitly say 'today' in their request."
    version = "1.0.1"
    
    def __init__(self):
        """Initialize the improved calendar tool."""
        super().__init__()
        
        # Cache calendar instances for each user
        self.calendar_instances: Dict[str, CalendarComponent] = {}
        
        # Available users and calendar types
        self.available_users = ["morgan_personal", "morgan_school", "spencer"]
        self.available_actions = ["read", "write", "create_event"]
        self.available_read_types = ["next_events", "day_summary", "week_summary", "specific_date"]
        self.available_calendar_names = ["primary", "personal", "work", "school", "class"]
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with comprehensive command descriptions.
        
        Returns:
            Detailed JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": f"List of calendar commands to execute. CRITICAL: Only ONE user per request. Available users: {self.available_users}. For 'What classes does Spencer have today?' use [{{'read_or_write': 'read', 'user': 'spencer', 'read_type': 'day_summary', 'calendar_name': 'class'}}]. For 'What's my next meeting?' use [{{'read_or_write': 'read', 'user': 'morgan_personal', 'read_type': 'next_events', 'limit': 3}}].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "read_or_write": {
                                "type": "string",
                                "description": "Operation type. 'read' retrieves existing events and information, 'write' or 'create_event' adds new events to the calendar. Use 'read' for questions about schedules, upcoming events, or daily summaries.",
                                "enum": ["read", "write", "create_event"]
                            },
                            "user": {
                                "type": "string",
                                "description": "Which user's calendar to access. 'morgan_personal' for Morgan's personal calendar, 'morgan_school' for Morgan's academic calendar, 'spencer' for Spencer's calendar. Each user has different permissions and calendar access. NEVER query multiple users in a single request.",
                                "enum": self.available_users
                            },
                            "read_type": {
                                "type": "string",
                                "description": "Type of read operation when read_or_write is 'read'. 'next_events' gets upcoming events in chronological order, 'day_summary' gets all events for a specific day (requires user to explicitly say 'today'), 'week_summary' gets events for the current week, 'specific_date' gets events for a particular date (requires date parameter).",
                                "enum": self.available_read_types
                            },
                            "calendar_name": {
                                "type": "string",
                                "description": "Specific calendar to access within the user's account. 'primary' for main calendar, 'personal' for personal events, 'work' for work-related events, 'school'/'class' for academic events. Use 'class' when user asks about classes or academic schedule.",
                                "enum": self.available_calendar_names,
                                "default": "primary"
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Maximum number of events to return for 'next_events' read_type. Use 1-3 for quick checks ('next meeting'), 5-10 for daily planning, 20+ for comprehensive reviews. Default is 10.",
                                "default": 10
                            },
                            "date": {
                                "type": "string",
                                "description": "Specific date in YYYY-MM-DD format when read_type is 'specific_date' or for event creation. For 'today' queries, use current date. Required for specific_date read_type and event creation."
                            },
                            "event_title": {
                                "type": "string",
                                "description": "Title/summary of the event when creating new events (read_or_write is 'write' or 'create_event'). Be descriptive and clear, e.g., 'Team Meeting with Marketing', 'Doctor Appointment', 'CS 101 Lecture'."
                            },
                            "event_description": {
                                "type": "string",
                                "description": "Detailed description of the event when creating events. Include agenda, location details, preparation notes, or other relevant information."
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start time for event creation. Accepts 12-hour (e.g., '11pm', '11:30AM') or 24-hour HH:MM (e.g., '23:00'). The tool will normalize to 24-hour internally. Required for event creation."
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End time for event creation. Accepts 12-hour (e.g., '11:30pm') or 24-hour HH:MM (e.g., '23:30'). Must be after start_time."
                            },
                            "location": {
                                "type": "string",
                                "description": "Event location when creating events. Can be physical address, room number, or virtual meeting link. e.g., 'Conference Room A', '123 Main St', 'https://zoom.us/j/123456789'."
                            },
                            "attendees": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of attendee email addresses when creating events. Include all participants who should receive calendar invitations."
                            },
                            "include_past_events": {
                                "type": "boolean",
                                "description": "Whether to include past events in the results. Normally false to focus on upcoming events, but set to true when user asks about historical events or 'what did I have earlier today'.",
                                "default": False
                            },
                            "time_zone": {
                                "type": "string",
                                "description": "Time zone for event times. Uses system default if not specified. Format like 'America/New_York', 'Europe/London', etc."
                            }
                        },
                        "required": ["read_or_write", "user"],
                        "if": {
                            "properties": {"read_or_write": {"const": "read"}}
                        },
                        "then": {
                            "required": ["read_type"]
                        },
                        "else": {
                            "if": {
                                "properties": {"read_or_write": {"enum": ["write", "create_event"]}}
                            },
                            "then": {
                                "required": ["event_title", "date", "start_time", "end_time"]
                            }
                        }
                    },
                    "minItems": 1,
                    "maxItems": 5
                }
            },
            "required": ["commands"]
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the calendar operations.
        """
        try:
            commands = params.get("commands", [])
            # Normalize loose inputs (ISO datetimes, write_type alias, 12h times)
            commands = [self._normalize_command(cmd) for cmd in commands]
            if LOG_TOOLS:
                self.logger.info("Normalized Calendar Commands -- %s", commands)

            if not commands:
                return {
                    "success": False,
                    "error": "No commands provided. At least one command is required.",
                    "available_users": self.available_users,
                    "available_actions": self.available_actions
                }
            
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Calendar -- %s", params)
            
            users_in_request = set(cmd.get("user") for cmd in commands if cmd.get("user"))
            if len(users_in_request) > 1:
                return {
                    "success": False,
                    "error": f"CRITICAL VIOLATION: Only ONE user per request allowed. Found users: {list(users_in_request)}",
                    "available_users": self.available_users,
                    "restriction": "Single user per request only"
                }
            
            validation_errors = self._validate_commands(commands)
            if validation_errors:
                return {
                    "success": False,
                    "error": "Command validation failed",
                    "validation_errors": validation_errors,
                    "available_users": self.available_users,
                    "available_actions": self.available_actions
                }
            
            results = []
            for i, cmd in enumerate(commands):
                try:
                    result = self._execute_single_command(cmd, i)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "command_index": i,
                        "command": cmd,
                        "error": str(e)
                    })
            
            successful_commands = [r for r in results if r.get("success")]
            failed_commands = [r for r in results if not r.get("success")]
            
            return {
                "success": len(failed_commands) == 0,
                "total_commands": len(commands),
                "successful_commands": len(successful_commands),
                "failed_commands": len(failed_commands),
                "results": results,
                "user": list(users_in_request)[0] if users_in_request else "unknown",
                "timestamp": self._get_current_timestamp()
            }
        except Exception as e:
            self.logger.error(f"Error executing calendar operations: {e}")
            return {
                "success": False,
                "error": f"Calendar execution failed: {str(e)}",
                "total_commands": len(params.get("commands", [])),
                "successful_commands": 0,
                "failed_commands": len(params.get("commands", []))
            }

    def _validate_commands(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Validate all commands before execution."""
        errors = []
        
        for i, cmd in enumerate(commands):
            cmd_errors = []
            
            # Check required fields
            if "read_or_write" not in cmd:
                cmd_errors.append("Missing required 'read_or_write' parameter")
            elif cmd["read_or_write"] not in self.available_actions:
                cmd_errors.append(f"Invalid read_or_write '{cmd['read_or_write']}'. Must be one of: {self.available_actions}")
            
            if "user" not in cmd:
                cmd_errors.append("Missing required 'user' parameter")
            elif cmd["user"] not in self.available_users:
                cmd_errors.append(f"Invalid user '{cmd['user']}'. Must be one of: {self.available_users}")
            
            # Validate read operations
            if cmd.get("read_or_write") == "read":
                if "read_type" not in cmd:
                    cmd_errors.append("Read operations require 'read_type' parameter")
                elif cmd["read_type"] not in self.available_read_types:
                    cmd_errors.append(f"Invalid read_type '{cmd['read_type']}'. Must be one of: {self.available_read_types}")
                
                # Check for day_summary restrictions
                if cmd.get("read_type") == "day_summary":
                    # This is a business logic check that would ideally be validated against actual user input
                    pass  # In practice, you'd check if user actually said "today"
                
                # Check specific_date requirements
                if cmd.get("read_type") == "specific_date" and "date" not in cmd:
                    cmd_errors.append("read_type 'specific_date' requires 'date' parameter")
            
            # Validate write operations
            elif cmd.get("read_or_write") in ["write", "create_event"]:
                required_fields = ["event_title", "date", "start_time", "end_time"]
                for field in required_fields:
                    if field not in cmd:
                        cmd_errors.append(f"Event creation requires '{field}' parameter")
                
                # Validate time format
                if "start_time" in cmd and not self._is_valid_time_format(cmd["start_time"]):
                    cmd_errors.append("start_time must be in HH:MM format (24-hour)")
                
                if "end_time" in cmd and not self._is_valid_time_format(cmd["end_time"]):
                    cmd_errors.append("end_time must be in HH:MM format (24-hour)")
                
                # Validate date format
                if "date" in cmd and not self._is_valid_date_format(cmd["date"]):
                    cmd_errors.append("date must be in YYYY-MM-DD format")
            
            # Add command-specific errors
            if cmd_errors:
                errors.append(f"Command {i+1}: {'; '.join(cmd_errors)}")
        
        return errors
    
    def _execute_single_command(self, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Execute a single calendar command."""
        try:
            user = cmd["user"]
            action = cmd["read_or_write"]
            
            # Get calendar instance
            calendar_instance = self._get_calendar_instance(user)
            if not calendar_instance:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Failed to initialize calendar for user: {user}",
                    "command": cmd
                }
            
            # Execute based on action type
            if action == "read":
                return self._handle_read_command(calendar_instance, cmd, cmd_index)
            elif action in ["write", "create_event"]:
                return self._handle_write_command(calendar_instance, cmd, cmd_index)
            else:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Unknown action: {action}",
                    "command": cmd
                }
                
        except Exception as e:
            return {
                "success": False,
                "command_index": cmd_index,
                "command": cmd,
                "error": str(e)
            }
    
    def _handle_read_command(self, calendar_instance: CalendarComponent, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Handle calendar read operations."""
        read_type = cmd["read_type"]
        calendar_name = cmd.get("calendar_name", "primary")
        limit = cmd.get("limit", 10)
        include_past = cmd.get("include_past_events", False)
        
        try:
            if read_type == "next_events":
                events = calendar_instance.get_upcoming_events(
                    calendar_name=calendar_name,
                    max_results=limit,
                    include_past=include_past
                )
                
                return {
                    "success": True,
                    "command_index": cmd_index,
                    "command": cmd,
                    "read_type": read_type,
                    "events": events,
                    "event_count": len(events),
                    "calendar": calendar_name,
                    "user": cmd["user"]
                }
            
            elif read_type == "day_summary":
                # Use current date if not specified
                target_date = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
                events = calendar_instance.get_day_events(
                    date=target_date,
                    calendar_name=calendar_name,
                    include_past=include_past
                )
                
                return {
                    "success": True,
                    "command_index": cmd_index,
                    "command": cmd,
                    "read_type": read_type,
                    "date": target_date,
                    "events": events,
                    "event_count": len(events),
                    "calendar": calendar_name,
                    "user": cmd["user"]
                }
            
            elif read_type == "week_summary":
                events = calendar_instance.get_week_events(
                    calendar_name=calendar_name,
                    include_past=include_past
                )
                
                return {
                    "success": True,
                    "command_index": cmd_index,
                    "command": cmd,
                    "read_type": read_type,
                    "events": events,
                    "event_count": len(events),
                    "calendar": calendar_name,
                    "user": cmd["user"]
                }
            
            elif read_type == "specific_date":
                target_date = cmd["date"]
                events = calendar_instance.get_day_events(
                    date=target_date,
                    calendar_name=calendar_name,
                    include_past=include_past
                )
                
                return {
                    "success": True,
                    "command_index": cmd_index,
                    "command": cmd,
                    "read_type": read_type,
                    "date": target_date,
                    "events": events,
                    "event_count": len(events),
                    "calendar": calendar_name,
                    "user": cmd["user"]
                }
            
            else:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Unknown read_type: {read_type}",
                    "command": cmd
                }
                
        except Exception as e:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Read operation failed: {str(e)}",
                "command": cmd
            }
    
    def _handle_write_command(self, calendar_instance: CalendarComponent, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Handle calendar write operations."""
        try:
            event_data = {
                "title": cmd["event_title"],
                "description": cmd.get("event_description", ""),
                "date": cmd["date"],
                "start_time": cmd["start_time"],
                "end_time": cmd["end_time"],
                "location": cmd.get("location", ""),
                "attendees": cmd.get("attendees", []),
                "calendar_name": cmd.get("calendar_name", "primary"),
                "time_zone": cmd.get("time_zone")
            }
            created_event = calendar_instance.create_event(event_data)
            return {
                "success": True,
                "command_index": cmd_index,
                "command": cmd,
                "operation": "create_event",
                "created_event": created_event,
                "event_title": cmd["event_title"],
                "event_date": cmd["date"],
                "event_time": f"{cmd['start_time']} - {cmd['end_time']}",
                "calendar": cmd.get("calendar_name", "primary"),
                "user": cmd["user"]
            }
        except Exception as e:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Event creation failed: {str(e)}",
                "command": cmd
            }

    def _normalize_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize inputs for write/create_event commands.
        - Accept 'write_type': 'create_event' alias by mapping to read_or_write
        - Accept ISO datetimes in start_time/end_time and derive date/HH:MM
        - Accept HH:MM:SS by truncating to HH:MM
        - Accept 12h formats like '11pm'/'11:30am'
        """
        try:
            normalized = dict(cmd)
            # Map common title synonyms to event_title
            if not normalized.get("event_title"):
                for alt in ("event_name", "title", "summary", "name"):
                    if isinstance(normalized.get(alt), str) and normalized.get(alt).strip():
                        normalized["event_title"] = normalized[alt].strip()
                        break
            wt = normalized.get("write_type")
            if wt and not normalized.get("read_or_write"):
                normalized["read_or_write"] = "create_event" if wt == "create_event" else "write"
            for key in ("start_time", "end_time"):
                val = normalized.get(key)
                if not val or not isinstance(val, str):
                    continue
                parsed = self._parse_time_like(val)
                if parsed:
                    if not normalized.get("date") and parsed.get("date"):
                        normalized["date"] = parsed["date"]
                    if parsed.get("time"):
                        normalized[key] = parsed["time"]
            dval = normalized.get("date")
            if isinstance(dval, str) and "T" in dval:
                try:
                    iso = dval.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    normalized["date"] = dt.date().isoformat()
                except Exception:
                    try:
                        normalized["date"] = dval.split("T", 1)[0]
                    except Exception:
                        pass
            if not normalized.get("calendar_name"):
                normalized["calendar_name"] = "primary"
            return normalized
        except Exception:
            return cmd

    def _parse_time_like(self, value: str) -> Dict[str, Any]:
        """Parse various time string formats into {'date':..., 'time':...}."""
        try:
            v = value.strip().rstrip(".,;:!")  # tolerate trailing punctuation from speech
            if "T" in v or (len(v) >= 10 and v[:10].count("-") == 2):
                try:
                    iso = v.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    return {"date": dt.date().isoformat(), "time": dt.strftime("%H:%M")}
                except Exception:
                    pass
            import re as _re
            m = _re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", v)
            if m:
                hh = int(m.group(1))
                mm = int(m.group(2))
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    return {"time": f"{hh:02d}:{mm:02d}"}
            m2 = _re.match(r"^(\d{1,2})(?::(\d{2}))?\s*([ap]m)$", v, flags=_re.IGNORECASE)
            if m2:
                hh = int(m2.group(1))
                mm = int(m2.group(2) or 0)
                ap = m2.group(3).lower()
                if hh == 12:
                    hh = 0
                if ap == 'pm':
                    hh += 12
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    return {"time": f"{hh:02d}:{mm:02d}"}
            return {}
        except Exception:
            return {}
    
    def _get_calendar_instance(self, user: str) -> CalendarComponent:
        """Get or create calendar instance for user."""
        if user not in self.calendar_instances:
            try:
                self.calendar_instances[user] = CalendarComponent(user=user)
            except Exception as e:
                self.logger.error(f"Failed to create calendar instance for {user}: {e}")
                return None
        return self.calendar_instances[user]
    
    def _is_valid_time_format(self, time_str: str) -> bool:
        """Validate HH:MM time format."""
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Validate YYYY-MM-DD date format."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        try:
            return datetime.now().isoformat()
        except Exception:
            return "unknown"