"""
Calendar Data Tool using BaseTool.

This tool provides comprehensive Google Calendar access with enhanced parameter
descriptions, better command validation, and detailed event management.
"""

from mcp_server.base_tool import BaseTool
from typing import Dict, Any, List
from datetime import datetime
from mcp_server.config import LOG_TOOLS
try:
    from mcp_server import config
except ImportError:
    # Fallback for MCP server context
    config = None

# Import user config for dynamic user resolution
try:
    from mcp_server.user_config import get_calendar_users, get_default_calendar_user
except ImportError:
    # Fallback if user_config not available
    def get_calendar_users():
        return ["user_personal", "user_school"]
    def get_default_calendar_user():
        return "user_personal"

# Import calendar component with fallback
try:
    from mcp_server.clients.calendar_client import CalendarComponent
except ImportError:
    CalendarComponent = None


class CalendarTool(BaseTool):
    """Enhanced tool for accessing and managing Google Calendar events with comprehensive command support."""
    
    name = "calendar_data"
    version = "1.0.2"
    
    @property
    def description(self) -> str:
        """Dynamic description using configured default user."""
        default_user = get_default_calendar_user()
        return f"Access Google Calendar for reading and creating events across multiple users and calendars. Supports viewing upcoming events, daily summaries, and event creation. DEFAULTS: User defaults to '{default_user}', action defaults to 'read' (only use 'write'/'create_event' when user explicitly asks to add/create/schedule a new event). CRITICAL RESTRICTION: Only query ONE user per request - never multiple users simultaneously. For day summaries, user MUST explicitly say 'today' in their request."
    
    def __init__(self):
        """Initialize the calendar tool."""
        super().__init__()
        
        # Cache calendar instances for each user
        self.calendar_instances: Dict[str, CalendarComponent] = {}
        
        # Available users from config (dynamically loaded)
        self._available_users = None
        self.available_actions = ["read", "write", "create_event"]
        self.available_read_types = ["next_events", "day_summary", "week_summary", "specific_date"]
        self.available_calendar_names = ["primary", "personal", "work", "school", "class", "default", "homeassist", "assistant"]
    
    @property
    def available_users(self) -> List[str]:
        """Dynamically get available users from config."""
        if self._available_users is None:
            self._available_users = get_calendar_users()
        return self._available_users
    
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
                    "description": f"List of calendar commands to execute. DEFAULTS: user='{get_default_calendar_user()}', read_or_write='read'. CRITICAL: Only ONE user per request. Available users: {self.available_users}. For 'What's my next meeting?' use [{{'read_type': 'next_events', 'limit': 3}}] (defaults to primary user + read).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "read_or_write": {
                                "type": "string",
                                "description": "Operation type. Defaults to 'read'. 'read' retrieves existing events and information, 'write' or 'create_event' adds new events to the calendar. Use 'read' for questions about schedules, upcoming events, or daily summaries. Only use 'write'/'create_event' when user EXPLICITLY asks to add, create, or schedule a new event.",
                                "enum": ["read", "write", "create_event"],
                                "default": "read"
                            },
                            "user": {
                                "type": "string",
                                "description": f"Which user's calendar to access. Defaults to '{get_default_calendar_user()}'. Each user has different permissions and calendar access. NEVER query multiple users in a single request.",
                                "enum": self.available_users,
                                "default": get_default_calendar_user()
                            },
                            "read_type": {
                                "type": "string",
                                "description": "Type of read operation when read_or_write is 'read'. 'next_events' gets upcoming events in chronological order, 'day_summary' gets all events for a specific day (requires user to explicitly say 'today'), 'week_summary' gets events for the current week, 'specific_date' gets events for a particular date (requires date parameter).",
                                "enum": self.available_read_types
                            },
                            "calendar_name": {
                                "type": "string",
                                "description": "Specific calendar to access within the user's account. For READ operations, defaults to 'primary' which reads from ALL calendars. For WRITE/CREATE operations, defaults to 'homeassist' (the dedicated assistant calendar). Options: 'primary'/'default' (all calendars for reads), 'homeassist'/'assistant' (for assistant-created events), 'personal', 'work', 'school'/'class'. You usually don't need to specify this - it auto-selects based on read vs write.",
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
                                "description": "Specific date in YYYY-MM-DD format when read_type is 'specific_date' or for event creation. IMPORTANT: Always use the current year (check CURRENT CONTEXT in system prompt) unless user explicitly mentions a different year. For 'today' queries, use current date. For relative dates like 'next Tuesday' or 'January 15th', use the current or upcoming year, not past years. Required for specific_date read_type and event creation."
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
                        "required": [],  # user defaults to primary user, read_or_write defaults to read
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
        """Normalize inputs for calendar commands.
        - Default user to primary user unless otherwise specified
        - Default read_or_write to 'read' unless explicitly creating an event
        - Accept 'write_type': 'create_event' alias by mapping to read_or_write
        - Accept ISO datetimes in start_time/end_time and derive date/HH:MM
        - Accept HH:MM:SS by truncating to HH:MM
        - Accept 12h formats like '11pm'/'11:30am'
        """
        try:
            normalized = dict(cmd)
            
            # Default user to primary user unless otherwise specified
            if not normalized.get("user"):
                normalized["user"] = get_default_calendar_user()
            
            # Map common title synonyms to event_title
            if not normalized.get("event_title"):
                for alt in ("event_name", "title", "summary", "name"):
                    if isinstance(normalized.get(alt), str) and normalized.get(alt).strip():
                        normalized["event_title"] = normalized[alt].strip()
                        break
                
                # Handle nested event object (e.g., {"event": {"title": "..."}})
                if not normalized.get("event_title") and isinstance(normalized.get("event"), dict):
                    event_obj = normalized["event"]
                    for alt in ("title", "event_name", "summary", "name"):
                        if isinstance(event_obj.get(alt), str) and event_obj.get(alt).strip():
                            normalized["event_title"] = event_obj[alt].strip()
                            break
                    # Also extract other event fields from nested object
                    # Handle both 'start_time'/'end_time' and 'start'/'end' variants
                    if "start_time" in event_obj:
                        normalized["start_time"] = event_obj["start_time"]
                    elif "start" in event_obj:
                        normalized["start_time"] = event_obj["start"]
                    if "end_time" in event_obj:
                        normalized["end_time"] = event_obj["end_time"]
                    elif "end" in event_obj:
                        normalized["end_time"] = event_obj["end"]
                    if "date" in event_obj:
                        normalized["date"] = event_obj["date"]
                    if "description" in event_obj:
                        normalized["event_description"] = event_obj["description"]
                    if "location" in event_obj:
                        normalized["location"] = event_obj["location"]
            
            # Also handle top-level 'start'/'end' fields as aliases
            if not normalized.get("start_time") and normalized.get("start"):
                normalized["start_time"] = normalized.pop("start")
            if not normalized.get("end_time") and normalized.get("end"):
                normalized["end_time"] = normalized.pop("end")
            
            # Handle "action" field as alias for read_or_write
            if normalized.get("action") and not normalized.get("read_or_write"):
                action = normalized["action"]
                if action in ["create_event", "write"]:
                    normalized["read_or_write"] = action
                elif action == "read":
                    normalized["read_or_write"] = "read"
            
            wt = normalized.get("write_type")
            if wt and not normalized.get("read_or_write"):
                normalized["read_or_write"] = "create_event" if wt == "create_event" else "write"
            
            # Default to 'read' unless explicitly creating an event (has event_title or write action)
            if not normalized.get("read_or_write"):
                # Only use write/create_event if there's clear intent to create
                has_event_creation_fields = normalized.get("event_title") or normalized.get("event_name")
                if has_event_creation_fields:
                    normalized["read_or_write"] = "create_event"
                else:
                    normalized["read_or_write"] = "read"
            
            # Default read_type to 'next_events' for read operations without a read_type
            if normalized.get("read_or_write") == "read" and not normalized.get("read_type"):
                normalized["read_type"] = "next_events"
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
            
            # Set default calendar_name based on operation type
            # - READ operations: default to "primary" (reads all calendars)
            # - WRITE operations: default to "homeassist" (dedicated assistant calendar)
            if not normalized.get("calendar_name"):
                if normalized.get("read_or_write") in ["write", "create_event"]:
                    normalized["calendar_name"] = "homeassist"
                else:
                    normalized["calendar_name"] = "primary"
            
            # Map common calendar name aliases
            calendar_name_mapping = {
                "default": "primary",
                "main": "primary",
                "default_calendar": "primary",
                "assistant": "homeassist",  # alias for homeassist
            }
            if normalized.get("calendar_name") in calendar_name_mapping:
                normalized["calendar_name"] = calendar_name_mapping[normalized["calendar_name"]]
            
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