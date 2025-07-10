from core.calendar_component import CalendarComponent
from mcp_server.base_tool import BaseTool
from typing import Dict, Any, List
from datetime import datetime, timezone
import config


class CalendarTool(BaseTool):
    """Tool for accessing and managing Google Calendar events via voice commands"""
    
    name = "calendar_data"
    description = "Read and write events to Google Calendar for Morgan and Spencer"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        # Cache calendar instances for each user
        self.calendar_instances: Dict[str, CalendarComponent] = {}
        
    def get_info(self):
        return {
            "name": "Calendar Tool",
            "description": """Access Google Calendar for reading and creating events. 
            Supports multiple users (morgan_personal, morgan_school, spencer) and actions like:
            - View next/latest events
            - Get daily summaries
            - Create new events (coming soon)
            Examples:
            - "What's my next meeting?"
            - "What classes does Spencer have today?"
            - "Show me Morgan's events for today"
            """,
            "version": "1.0.0"
        }


    def get_schema(self):
        """Return the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": """
                    List of commands.
                    For 'what classes does spencer have today' use {\"read_or_write\": \"read\", \"user\": \"spencer\", \"read_type\": \"day_summary\", \"calendar_name\":\"class\" }. 
                    If read_or_write = \"read\" then require read_type. 
                    """,
                    "items": {
                        "type": "object",
                        "properties": {
                            "read_or_write":{
                                "type": "string",
                                "description": "Whether the user is looking to read (view events already on the calendar) from the calendar or write (create a new event) to the calendar",
                                "enum": ["read", "write"]
                            },
                            "user":{
                                "type": "string",
                                "description": "Which user the calendar request is about. If not specified by user, ask until they provide a valid one.",
                                "enum": ["morgan_personal","morgan_school", "spencer"]
                            },
                            "read_type": {
                                "type": "string",
                                "enum": ["latest_event", "next_event", "day_summary"],
                                "description": "The type of read request being made. Only relevant is the read_or_write param is read"
                            },
                            "calendar_name": {
                                "type": "string",
                                "enum": ["all", "class", "general", "tasks"],
                                "description": "Which calendar type the user is looking to read or write to. (i.e. whether the user wants to view general events or only their class events)"
                            },
                        },
                        "required": ["read_or_write", "user"]
                    }
                }
            },
            "required": ["commands"],
            "description": self.description
        }
    
    def _get_calendar(self, user: str) -> CalendarComponent:
        """Get or create calendar instance for specified user."""
        if user not in self.calendar_instances:
            try:
                self.calendar_instances[user] = CalendarComponent(user=user)
                self.log_info(f"Created calendar instance for user: {user}")
            except Exception as e:
                self.log_error(f"Failed to create calendar for {user}: {e}")
                raise
        return self.calendar_instances[user]
    
    def _handle_read(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read operations for calendar events."""
        user = command.get("user")
        read_type = command.get("read_type")
        calendar_name = command.get("calendar_name", "all")
        
        try:
            # Get calendar instance for user
            calendar = self._get_calendar(user)
            
            # Handle different read types
            if read_type == "latest_event":
                event = calendar.get_last_event()
                if event:
                    return {
                        "success": True,
                        "message": f"Last event: {event['summary']} on {event['start_date']} at {event['start_time']}",
                        "event": event,
                        "user": user
                    }
                else:
                    return {
                        "success": True,
                        "message": f"No past events found for {user}",
                        "user": user
                    }
                    
            elif read_type == "next_event":
                event = calendar.get_next_event()
                if event:
                    return {
                        "success": True,
                        "message": f"Next event: {event['summary']} on {event['start_date']} at {event['start_time']}",
                        "event": event,
                        "user": user
                    }
                else:
                    return {
                        "success": True,
                        "message": f"No upcoming events found for {user}",
                        "user": user
                    }
                    
            elif read_type == "day_summary":
                # Get today's events
                events = calendar.get_todays_events()
                
                if events:
                    # Filter by calendar name if specified
                    if calendar_name != "all":
                        events = [e for e in events if calendar_name.lower() in e.get('calendar_name', '').lower()]
                    
                    if events:
                        # Create summary message
                        summary = calendar.get_events_summary(len(events))
                        return {
                            "success": True,
                            "message": summary,
                            "events": events,
                            "count": len(events),
                            "user": user
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"No {calendar_name} events today for {user}",
                            "user": user
                        }
                else:
                    return {
                        "success": True,
                        "message": f"No events scheduled for today for {user}",
                        "user": user
                    }
                    
            else:
                return {
                    "success": False,
                    "error": f"Unknown read_type: {read_type}",
                    "user": user
                }
                
        except Exception as e:
            self.log_error(f"Read operation failed for {user}: {e}")
            return {
                "success": False,
                "error": f"Failed to read calendar: {str(e)}",
                "user": user
            }
    
    def _handle_write(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle write operations for calendar events."""
        # TODO: Implement event creation
        return {
            "success": False,
            "error": "Write operations not yet implemented",
            "user": command.get("user")
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calendar operations based on commands."""
        if config.DEBUG_MODE:
            print(params)
            
        commands = params.get("commands", [])
        
        if not commands:
            return {
                "success": False,
                "error": "No commands provided"
            }
        
        results = []
        
        for command in commands:
            read_or_write = command.get("read_or_write")
            
            if read_or_write == "read":
                result = self._handle_read(command)
            elif read_or_write == "write":
                result = self._handle_write(command)
            else:
                result = {
                    "success": False,
                    "error": f"Invalid operation: {read_or_write}. Must be 'read' or 'write'"
                }
            
            results.append(result)
        
        # Return single result if only one command, otherwise return all
        if len(results) == 1:
            return results[0]
        else:
            return {
                "success": all(r.get("success", False) for r in results),
                "results": results,
                "message": "Processed multiple calendar commands"
            }