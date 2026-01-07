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
    version = "1.1.0"
    
    def __init__(self):
        """Initialize the calendar tool."""
        # Initialize instance variables BEFORE calling super().__init__()
        # (which accesses self.description which needs _available_calendars)
        self.calendar_instances: Dict[str, CalendarComponent] = {}
        self._available_calendars = None
        self.available_actions = ["read", "create_event"]
        self.available_read_types = ["next_events", "day_summary", "week_summary", "specific_date"]
        
        super().__init__()
    
    @property
    def description(self) -> str:
        """Dynamic description using configured default user."""
        calendars = self.available_calendars
        return f"Google Calendar access. ONE tool call per request. Calendars: {calendars}. READ: auto-reads from all. WRITE: 'calendars' array adds same event to multiple calendars in one call. IMPORTANT: For create_event, NEVER invent times - if user didn't specify start_time/end_time, OMIT those fields and the tool will tell you to ask the user."
    
    @property
    def available_calendars(self) -> List[str]:
        """Dynamically get available calendars from config."""
        if self._available_calendars is None:
            self._available_calendars = get_calendar_users()
        return self._available_calendars
    
    @property
    def available_users(self) -> List[str]:
        """Alias for available_calendars for backward compatibility."""
        return self.available_calendars
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with comprehensive command descriptions.
        
        Returns:
            Detailed JSON schema dictionary
        """
        # Build calendar options: individual calendars + "all" for reading
        calendar_options = self.available_calendars + ["all"]
        
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": "Exactly ONE command per tool call. Never call this tool multiple times for the same user request.",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "read_or_write": {
                                "type": "string",
                                "description": "'read' (default) for viewing events. 'create_event' only when user explicitly asks to add/create/schedule.",
                                "enum": ["read", "create_event"],
                                "default": "read"
                            },
                            "calendar": {
                                "type": "string",
                                "description": f"READ: defaults to 'all'. WRITE: ignored (use 'calendars').",
                                "enum": calendar_options
                            },
                            "calendars": {
                                "type": "array",
                                "items": {"type": "string", "enum": self.available_calendars},
                                "description": f"WRITE only: calendars to add event to. Defaults to ['morgan_personal']. Use array to add same event to multiple calendars: ['morgan_personal', 'homeassist']"
                            },
                            "read_type": {
                                "type": "string",
                                "description": "For read: 'next_events' (default), 'day_summary', 'week_summary', 'specific_date'.",
                                "enum": self.available_read_types,
                                "default": "next_events"
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Max events to return. Default 10.",
                                "default": 10
                            },
                            "date": {
                                "type": "string",
                                "description": "YYYY-MM-DD format. Required for 'specific_date' read and event creation. For create_event: ONLY provide if user specified a date. If user says 'today'/'tomorrow'/etc, convert to YYYY-MM-DD. If no date mentioned, OMIT this field."
                            },
                            "event_title": {
                                "type": "string",
                                "description": "Event title. Required for create_event."
                            },
                            "event_description": {
                                "type": "string",
                                "description": "Event description. Optional."
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start time (HH:MM or '3pm'). Required for create_event. CRITICAL: DO NOT provide this field unless the user explicitly stated a time. If user did not specify a time, OMIT this field entirely - the tool will return an error prompting you to ask the user."
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End time (HH:MM or '4pm'). Required for create_event. CRITICAL: DO NOT provide this field unless the user explicitly stated an end time. If not specified, OMIT this field - the tool will prompt you to ask the user."
                            },
                            "location": {
                                "type": "string",
                                "description": "Event location. Optional."
                            },
                            "attendees": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Attendee emails. Optional."
                            }
                        },
                        "required": []
                    }
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
                    "available_calendars": self.available_calendars,
                    "available_actions": self.available_actions
                }
            
            # Enforce: ONE command per tool call
            if len(commands) > 1:
                return {
                    "success": False,
                    "error": "Only ONE command per tool call. Do not make multiple calendar tool calls for the same request.",
                    "available_calendars": self.available_calendars
                }
            
            if LOG_TOOLS:
                self.logger.info("Executing Tool: Calendar -- %s", params)
            
            # Limit: Only ONE write command per call (to prevent duplicate events)
            write_commands = [c for c in commands if c.get("read_or_write") == "create_event"]
            if len(write_commands) > 1:
                return {
                    "success": False,
                    "error": "Only ONE write/create_event command allowed per call. To add an event to multiple calendars, use the 'calendars' array parameter instead of multiple commands.",
                    "available_calendars": self.available_calendars
                }
            
            validation_errors = self._validate_commands(commands)
            if validation_errors:
                # Check if errors are about missing info - provide actionable guidance
                missing_info_errors = [e for e in validation_errors if "MISSING_INFO:" in e]
                if missing_info_errors:
                    # Extract the user-friendly message
                    guidance = missing_info_errors[0].replace("Command 1: MISSING_INFO: ", "")
                    return {
                        "success": False,
                        "error": "Cannot create event - missing required information",
                        "action_required": guidance,
                        "hint": "Ask the user to provide this information before trying again."
                    }
                return {
                    "success": False,
                    "error": "Command validation failed",
                    "validation_errors": validation_errors,
                    "available_calendars": self.available_calendars,
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
            
            # Determine calendars used
            calendars_used = set()
            for cmd in commands:
                cal = cmd.get("calendar") or cmd.get("user")
                if cal:
                    calendars_used.add(cal)
            
            return {
                "success": len(failed_commands) == 0,
                "total_commands": len(commands),
                "successful_commands": len(successful_commands),
                "failed_commands": len(failed_commands),
                "results": results,
                "calendars": list(calendars_used),
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
        
        # Valid calendar options include "all" for read operations
        valid_calendars = self.available_calendars + ["all"]
        
        for i, cmd in enumerate(commands):
            cmd_errors = []
            
            # Check required fields
            if "read_or_write" not in cmd:
                cmd_errors.append("Missing required 'read_or_write' parameter")
            elif cmd["read_or_write"] not in self.available_actions:
                cmd_errors.append(f"Invalid read_or_write '{cmd['read_or_write']}'. Must be one of: {self.available_actions}")
            
            # Get calendar value (from 'calendar' or 'user' param)
            calendar = cmd.get("calendar") or cmd.get("user")
            
            if not calendar:
                cmd_errors.append("Missing required 'calendar' parameter")
            elif calendar not in valid_calendars:
                cmd_errors.append(f"Invalid calendar '{calendar}'. Must be one of: {valid_calendars}")
            elif calendar == "all" and cmd.get("read_or_write") in ["write", "create_event"]:
                cmd_errors.append("Cannot use 'all' for write operations. Specify a specific calendar to write to.")
            
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
            elif cmd.get("read_or_write") == "create_event":
                # Check for missing required fields with user-friendly prompts
                missing_fields = []
                if "event_title" not in cmd or not cmd["event_title"]:
                    missing_fields.append("event title/name")
                if "date" not in cmd or not cmd["date"]:
                    missing_fields.append("date")
                if "start_time" not in cmd or not cmd["start_time"]:
                    missing_fields.append("start time")
                if "end_time" not in cmd or not cmd["end_time"]:
                    missing_fields.append("end time")
                
                if missing_fields:
                    # Create user-friendly prompt to ask for missing info
                    if len(missing_fields) == 1:
                        cmd_errors.append(f"MISSING_INFO: Please ask the user for the {missing_fields[0]} of the event.")
                    else:
                        fields_str = ", ".join(missing_fields[:-1]) + f" and {missing_fields[-1]}"
                        cmd_errors.append(f"MISSING_INFO: Please ask the user for the {fields_str} of the event.")
                else:
                    # Validate time format only if times are provided
                    if not self._is_valid_time_format(cmd["start_time"]):
                        cmd_errors.append("INVALID_FORMAT: start_time must be in HH:MM format (e.g., '14:00' or '2pm')")
                    
                    if not self._is_valid_time_format(cmd["end_time"]):
                        cmd_errors.append("INVALID_FORMAT: end_time must be in HH:MM format (e.g., '15:00' or '3pm')")
                    
                    # Validate date format
                    if not self._is_valid_date_format(cmd["date"]):
                        cmd_errors.append("INVALID_FORMAT: date must be in YYYY-MM-DD format")
            
            # Add command-specific errors
            if cmd_errors:
                errors.append(f"Command {i+1}: {'; '.join(cmd_errors)}")
        
        return errors
    
    def _execute_single_command(self, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Execute a single calendar command."""
        try:
            calendar = cmd.get("calendar") or cmd.get("user")
            action = cmd["read_or_write"]
            
            # Write commands handle their own calendar instances (supports multiple calendars)
            if action in ["write", "create_event"]:
                return self._handle_write_command(cmd, cmd_index)
            
            # Handle "all" calendars for read operations
            if calendar == "all" and action == "read":
                return self._handle_read_all_calendars(cmd, cmd_index)
            
            # Get calendar instance for specific calendar (read operations)
            calendar_instance = self._get_calendar_instance(calendar)
            if not calendar_instance:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Failed to initialize calendar: {calendar}",
                    "command": cmd
                }
            
            # Execute read operation
            if action == "read":
                return self._handle_read_command(calendar_instance, cmd, cmd_index)
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
    
    def _handle_read_all_calendars(self, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Handle read operations that span all configured calendars."""
        read_type = cmd.get("read_type", "next_events")
        limit = cmd.get("limit", 10)
        include_past = cmd.get("include_past_events", False)
        
        all_events = []
        calendars_read = []
        errors = []
        
        # Get calendar IDs from config
        try:
            from mcp_server.config import CALENDAR_USERS
        except ImportError:
            CALENDAR_USERS = {}
        
        # Read from each configured calendar
        for cal_name in self.available_calendars:
            try:
                calendar_instance = self._get_calendar_instance(cal_name)
                if not calendar_instance:
                    errors.append(f"Failed to initialize {cal_name}")
                    continue
                
                calendars_read.append(cal_name)
                
                # Get the specific calendar_id for this calendar from config
                # Service accounts need the actual calendar ID, not "primary"
                cal_config = CALENDAR_USERS.get(cal_name, {})
                calendar_id = cal_config.get("calendar_id", "primary")
                
                if read_type == "next_events":
                    events = calendar_instance.get_upcoming_events(
                        calendar_name=calendar_id,
                        max_results=limit,
                        include_past=include_past
                    )
                elif read_type == "day_summary":
                    target_date = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
                    events = calendar_instance.get_day_events(
                        date=target_date,
                        calendar_name=calendar_id,
                        include_past=include_past
                    )
                elif read_type == "week_summary":
                    events = calendar_instance.get_week_events(
                        calendar_name=calendar_id,
                        include_past=include_past
                    )
                elif read_type == "specific_date":
                    target_date = cmd["date"]
                    events = calendar_instance.get_day_events(
                        date=target_date,
                        calendar_name=calendar_id,
                        include_past=include_past
                    )
                else:
                    events = []
                
                # Add source calendar info to each event
                for event in events:
                    if isinstance(event, dict):
                        event["source_calendar"] = cal_name
                
                all_events.extend(events)
                
            except Exception as e:
                errors.append(f"{cal_name}: {str(e)}")
        
        # Sort all events by start time
        def get_event_sort_key(event):
            if isinstance(event, dict):
                # Try to get start time from various formats
                start = event.get("start_time") or event.get("start") or ""
                if isinstance(start, dict):
                    start = start.get("dateTime") or start.get("date") or ""
                return str(start)
            return str(event)
        
        all_events.sort(key=get_event_sort_key)
        
        # Limit total results
        limited_events = all_events[:limit]
        
        result = {
            "success": True,
            "command_index": cmd_index,
            "command": cmd,
            "read_type": read_type,
            "events": limited_events,
            "event_count": len(limited_events),
            "total_events_found": len(all_events),
            "calendars_read": calendars_read,
            "calendar": "all"
        }
        
        if errors:
            result["warnings"] = errors
        
        if read_type in ["day_summary", "specific_date"]:
            result["date"] = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        return result
    
    def _handle_read_command(self, calendar_instance: CalendarComponent, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Handle calendar read operations for a single calendar."""
        read_type = cmd["read_type"]
        calendar = cmd.get("calendar") or cmd.get("user")
        limit = cmd.get("limit", 10)
        include_past = cmd.get("include_past_events", False)
        
        # Get the specific calendar_id from config (service accounts need this)
        try:
            from mcp_server.config import CALENDAR_USERS
            cal_config = CALENDAR_USERS.get(calendar, {})
            calendar_id = cal_config.get("calendar_id", "primary")
        except ImportError:
            calendar_id = "primary"
        
        try:
            if read_type == "next_events":
                events = calendar_instance.get_upcoming_events(
                    calendar_name=calendar_id,
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
                    "calendar": calendar
                }
            
            elif read_type == "day_summary":
                # Use current date if not specified
                target_date = cmd.get("date", datetime.now().strftime("%Y-%m-%d"))
                events = calendar_instance.get_day_events(
                    date=target_date,
                    calendar_name=calendar_id,
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
                    "calendar": calendar
                }
            
            elif read_type == "week_summary":
                events = calendar_instance.get_week_events(
                    calendar_name=calendar_id,
                    include_past=include_past
                )
                
                return {
                    "success": True,
                    "command_index": cmd_index,
                    "command": cmd,
                    "read_type": read_type,
                    "events": events,
                    "event_count": len(events),
                    "calendar": calendar
                }
            
            elif read_type == "specific_date":
                target_date = cmd["date"]
                events = calendar_instance.get_day_events(
                    date=target_date,
                    calendar_name=calendar_id,
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
                    "calendar": calendar
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
    
    def _handle_write_command(self, cmd: Dict[str, Any], cmd_index: int) -> Dict[str, Any]:
        """Handle calendar write operations. Supports adding ONE event to MULTIPLE calendars."""
        try:
            from mcp_server.config import CALENDAR_USERS
        except ImportError:
            CALENDAR_USERS = {}
        
        # Get target calendars - prefer 'calendars' array, fallback to 'calendar' string
        target_calendars = cmd.get("calendars")
        if not target_calendars:
            single_cal = cmd.get("calendar") or cmd.get("user") or "morgan_personal"
            target_calendars = [single_cal]
        
        # Validate all calendars exist
        for cal in target_calendars:
            if cal not in CALENDAR_USERS:
                return {
                    "success": False,
                    "command_index": cmd_index,
                    "error": f"Unknown calendar: {cal}. Available: {list(CALENDAR_USERS.keys())}",
                    "command": cmd
                }
        
        # Create event on each target calendar
        created_events = []
        errors = []
        
        for cal_name in target_calendars:
            try:
                # Get calendar instance for this calendar
                cal_instance = self._get_calendar_instance(cal_name)
                if not cal_instance:
                    errors.append(f"Failed to initialize {cal_name}")
                    continue
                
                # Get the calendar_id from config
                cal_config = CALENDAR_USERS.get(cal_name, {})
                calendar_id = cal_config.get("calendar_id", "primary")
                
                event_data = {
                    "title": cmd["event_title"],
                    "description": cmd.get("event_description", ""),
                    "date": cmd["date"],
                    "start_time": cmd["start_time"],
                    "end_time": cmd["end_time"],
                    "location": cmd.get("location", ""),
                    "attendees": cmd.get("attendees", []),
                    "calendar_name": calendar_id,
                    "time_zone": cmd.get("time_zone")
                }
                
                created = cal_instance.create_event(event_data)
                created_events.append({
                    "calendar": cal_name,
                    "event_id": created.get("id"),
                    "link": created.get("htmlLink")
                })
                
            except Exception as e:
                errors.append(f"{cal_name}: {str(e)}")
        
        if not created_events:
            return {
                "success": False,
                "command_index": cmd_index,
                "error": f"Failed to create event on any calendar: {errors}",
                "command": cmd
            }
        
        result = {
            "success": True,
            "command_index": cmd_index,
            "command": cmd,
            "operation": "create_event",
            "event_title": cmd["event_title"],
            "event_date": cmd["date"],
            "event_time": f"{cmd['start_time']} - {cmd['end_time']}",
            "created_on_calendars": created_events,
            "calendars_count": len(created_events)
        }
        
        if errors:
            result["warnings"] = errors
        
        return result

    def _normalize_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize inputs for calendar commands.
        - Handle 'calendar' param (new) and 'user' param (legacy) - prefer 'calendar'
        - Default calendar based on operation:
          - READ: defaults to "all" (read from all calendars)
          - WRITE: defaults to "morgan_personal"
        - Default read_or_write to 'read' unless explicitly creating an event
        - Accept 'write_type': 'create_event' alias by mapping to read_or_write
        - Accept ISO datetimes in start_time/end_time and derive date/HH:MM
        - Accept HH:MM:SS by truncating to HH:MM
        - Accept 12h formats like '11pm'/'11:30am'
        """
        try:
            normalized = dict(cmd)
            
            # Handle 'calendar' and 'user' params - prefer 'calendar', fallback to 'user'
            # NOTE: Default calendar is set LATER based on read vs write operation
            if normalized.get("calendar"):
                normalized["user"] = normalized["calendar"]
            elif normalized.get("user"):
                normalized["calendar"] = normalized["user"]
            # else: calendar default will be set after determining read/write
            
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
                    # Note: These might be strings OR dicts like:
                    #   {"dateTime": "2026-01-07T16:30:00"} 
                    #   {"date": "2026-01-07", "time": "16:30"}
                    for key in ("start_time", "start"):
                        if key in event_obj:
                            val = event_obj[key]
                            if isinstance(val, dict):
                                # Extract time from {"date": "...", "time": "..."} format
                                if "time" in val:
                                    normalized["start_time"] = val["time"]
                                    if "date" in val and not normalized.get("date"):
                                        normalized["date"] = val["date"]
                                    break
                                val = val.get("dateTime") or val.get("date") or ""
                            if isinstance(val, str) and val:
                                normalized["start_time"] = val
                                break
                    for key in ("end_time", "end"):
                        if key in event_obj:
                            val = event_obj[key]
                            if isinstance(val, dict):
                                # Extract time from {"date": "...", "time": "..."} format
                                if "time" in val:
                                    normalized["end_time"] = val["time"]
                                    break
                                val = val.get("dateTime") or val.get("date") or ""
                            if isinstance(val, str) and val:
                                normalized["end_time"] = val
                                break
                    if "date" in event_obj and not normalized.get("date"):
                        val = event_obj["date"]
                        if isinstance(val, dict):
                            val = val.get("dateTime") or val.get("date") or ""
                        if isinstance(val, str):
                            normalized["date"] = val
                    if "description" in event_obj:
                        normalized["event_description"] = event_obj["description"]
                    if "location" in event_obj:
                        normalized["location"] = event_obj["location"]
            
            # Also handle top-level 'start'/'end' fields as aliases
            # Note: These might be strings OR dicts like:
            #   {"dateTime": "2026-01-07T16:30:00"}
            #   {"date": "2026-01-07", "time": "16:30"}
            if not normalized.get("start_time") and normalized.get("start"):
                val = normalized.pop("start")
                if isinstance(val, dict):
                    if "time" in val:
                        normalized["start_time"] = val["time"]
                        if "date" in val and not normalized.get("date"):
                            normalized["date"] = val["date"]
                        val = None  # Already handled
                    else:
                        val = val.get("dateTime") or val.get("date") or ""
                if isinstance(val, str) and val:
                    normalized["start_time"] = val
            if not normalized.get("end_time") and normalized.get("end"):
                val = normalized.pop("end")
                if isinstance(val, dict):
                    if "time" in val:
                        normalized["end_time"] = val["time"]
                        val = None  # Already handled
                    else:
                        val = val.get("dateTime") or val.get("date") or ""
                if isinstance(val, str) and val:
                    normalized["end_time"] = val
            
            # Handle "action" field as alias for read_or_write
            if normalized.get("action") and not normalized.get("read_or_write"):
                action = normalized["action"]
                if action in ["create_event", "write"]:
                    normalized["read_or_write"] = action
                elif action == "read":
                    normalized["read_or_write"] = "read"
            
            wt = normalized.get("write_type")
            if wt and not normalized.get("read_or_write"):
                normalized["read_or_write"] = "create_event"
            
            # Normalize "write" to "create_event" (write is deprecated alias)
            if normalized.get("read_or_write") == "write":
                normalized["read_or_write"] = "create_event"
            
            # Default to 'read' unless explicitly creating an event (has event_title or write action)
            if not normalized.get("read_or_write"):
                # Only use write/create_event if there's clear intent to create
                has_event_creation_fields = normalized.get("event_title") or normalized.get("event_name")
                if has_event_creation_fields:
                    normalized["read_or_write"] = "create_event"
                else:
                    normalized["read_or_write"] = "read"
            
            # NOW set default calendar based on operation type (if not already specified)
            # - READ: defaults to "all" (read from all calendars combined)
            # - WRITE: defaults to "morgan_personal"
            if not normalized.get("calendar") and not normalized.get("user"):
                if normalized.get("read_or_write") in ["write", "create_event"]:
                    normalized["calendar"] = "morgan_personal"
                    normalized["user"] = "morgan_personal"
                else:
                    normalized["calendar"] = "all"
                    normalized["user"] = "all"
            
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
            # Handle case where date is a dict like {"dateTime": "..."} or {"date": "..."}
            if isinstance(dval, dict):
                dval = dval.get("dateTime") or dval.get("date") or ""
                if isinstance(dval, str):
                    normalized["date"] = dval
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
        if not isinstance(time_str, str):
            return False
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Validate YYYY-MM-DD date format."""
        if not isinstance(date_str, str):
            return False
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