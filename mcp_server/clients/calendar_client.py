from __future__ import print_function
import os
from pathlib import Path
import sys
from typing import Optional, Dict, List, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timezone, timedelta

# Project setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Prefer MCP server config when running tools; fall back to project root config
try:
    from mcp_server import config as config  # type: ignore
except Exception:
    try:
        import config  # type: ignore
    except Exception:
        # Minimal fallback config if neither is available
        class config:  # type: ignore
            DEBUG_MODE = False
            CALENDAR_USERS = {}
            CALENDAR_SCOPES = [
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/calendar.events",
            ]
            DEFAULT_TIME_ZONE = None

class CalendarComponent:
    """
    Google Calendar component with proper credentials management.
    
    Each instance is bound to a specific user that must be explicitly provided.
    To work with a different user, create a new CalendarComponent instance.
    """
    
    def __init__(self, user: str):
        """
        Initialize calendar component for a specific user.
        
        Args:
            user: User name ('morgan_personal' or 'morgan_school' or 'spencer'). Must be explicitly specified.
        
        Note:
            Once created, the user cannot be changed. Create a new instance for a different user.
        """
        # User is immutable once set and must be explicitly provided
        self._user = user
        self.creds: Optional[Credentials] = None
        self.service = None
        self.error_message: Optional[str] = None
        
        # Validate user
        if not self._validate_user():
            users = list(getattr(config, 'CALENDAR_USERS', {}).keys())
            raise ValueError(f"Invalid user '{self.user}'. Must be one of: {users}")
        
        # Initialize credentials and service
        self._initialize_credentials()
    
    @property
    def user(self) -> str:
        """Get the user associated with this calendar component."""
        return self._user

    def get_calendar_mappings(self):
        """Get calendar ID->name and name->ID mappings from Google Calendar API."""
        try:
            self._maybe_refresh_credentials()
            if not self.service:
                return {}, {}
            
            # Get all calendars for this user
            calendar_list = self.service.calendarList().list().execute()
            
            id_to_name = {}
            name_to_id = {}
            
            for cal in calendar_list.get('items', []):
                cal_id = cal['id']
                cal_name = cal.get('summary', cal_id)  # Use summary as the display name
                
                id_to_name[cal_id] = cal_name
                name_to_id[cal_name] = cal_id
            
            return id_to_name, name_to_id
            
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"⚠️ Error getting calendar mappings: {e}")
            return {}, {}

    def calendar_id_to_name(self, cal_id) -> str:
        """Convert calendar ID to human-readable name."""
        id_to_name, _ = self.get_calendar_mappings()
        return id_to_name.get(cal_id, cal_id)  # Return ID if name not found

    def calendar_name_to_id(self, name) -> str:
        """Convert calendar name to calendar ID (case-insensitive)."""
        _, name_to_id = self.get_calendar_mappings()
        # Try exact match first
        if name in name_to_id:
            return name_to_id[name]
        # Try case-insensitive match
        name_lower = name.lower()
        for cal_name, cal_id in name_to_id.items():
            if cal_name.lower() == name_lower:
                return cal_id
        # Return the name as-is if not found (will be used as ID directly)
        return name

    def get_morning_datetime(self) -> datetime:
        """Get the morning datetime for the user."""
        return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'

    def _validate_user(self) -> bool:
        """Validate that the user is configured."""
        return self._user in getattr(config, 'CALENDAR_USERS', {})
    
    def clear_credentials(self):
        """Clear stored credentials to force re-authentication with new permissions."""
        try:
            user_config = self._get_user_config()
            token_path = user_config["token"]
            if os.path.exists(token_path):
                os.remove(token_path)
                print(f"✅ Cleared credentials for {self.user}. Next run will request new authentication.")
                return True
            else:
                print(f"ℹ️ No credentials found for {self.user}")
                return False
        except Exception as e:
            print(f"❌ Error clearing credentials: {e}")
            return False
    
    def _get_user_config(self) -> Dict[str, str]:
        """Get configuration for the current user."""
        users_cfg = getattr(config, 'CALENDAR_USERS', {})
        return users_cfg[self._user]
    
    def _initialize_credentials(self) -> bool:
        """Initialize Google credentials for the user."""
        try:
            user_config = self._get_user_config()
            token_path = user_config["token"]
            client_secret_path = user_config["client_secret"]
            
            # Ensure credentials directory exists (derive from token path)
            os.makedirs(str(Path(token_path).parent), exist_ok=True)
            
            # Load existing token if available
            if os.path.exists(token_path):
                try:
                    scopes = getattr(
                        config,
                        "CALENDAR_SCOPES",
                        [
                            "https://www.googleapis.com/auth/calendar.readonly",
                            "https://www.googleapis.com/auth/calendar.events",
                        ],
                    )
                    self.creds = Credentials.from_authorized_user_file(token_path, scopes)
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"📅 Loaded existing calendar credentials for {self.user}")

                    # If the existing token does not contain a refresh token, proactively
                    # upgrade to offline access so we can refresh automatically.
                    if self.creds and not getattr(self.creds, "refresh_token", None):
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"♻️  Upgrading {self.user} credentials to offline access (refresh token)...")
                        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
                        self.creds = flow.run_local_server(
                            port=0,
                            access_type='offline',
                            prompt='consent'
                        )
                        with open(token_path, 'w') as token_file:
                            token_file.write(self.creds.to_json())
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"💾 Saved upgraded offline credentials for {self.user} to {token_path}")
                except Exception as e:
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"⚠️ Failed to load existing token: {e}")
                    self.creds = None
            
            # Check if credentials need refresh or reauthorization
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    # Refresh expired credentials
                    try:
                        self.creds.refresh(Request())
                        # Save the refreshed credentials immediately
                        with open(token_path, 'w') as token_file:
                            token_file.write(self.creds.to_json())
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"🔄 Refreshed and saved calendar credentials for {self.user}")
                    except Exception as e:
                        if getattr(config, "DEBUG_MODE", False):
                            print(f"⚠️ Failed to refresh credentials: {e}")
                        self.creds = None
                
                # If still no valid credentials, need to reauthorize
                if not self.creds or not self.creds.valid:
                    if not os.path.exists(client_secret_path):
                        raise FileNotFoundError(f"Client secret file not found: {client_secret_path}")
                    
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"🔐 Starting OAuth flow for {self.user}...")
                    
                    scopes = getattr(
                        config,
                        "CALENDAR_SCOPES",
                        [
                            "https://www.googleapis.com/auth/calendar.readonly",
                            "https://www.googleapis.com/auth/calendar.events",
                        ],
                    )
                    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
                    self.creds = flow.run_local_server(
                        port=0, 
                        access_type='offline',
                        prompt='consent'
                    )
                
                # Save updated credentials
                with open(token_path, 'w') as token_file:
                    token_file.write(self.creds.to_json())
                    if getattr(config, "DEBUG_MODE", False):
                        print(f"💾 Saved calendar credentials for {self.user} to {token_path}")

            # Build calendar service with discovery cache disabled
            # This prevents the "file_cache is only supported with oauth2client<4.0.0" message
            self.service = build('calendar', 'v3', credentials=self.creds, cache_discovery=False)
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"✅ Calendar service initialized for {self.user}")
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            print(f"❌ Failed to initialize calendar credentials for {self.user}: {e}")
            return False

    def _maybe_refresh_credentials(self) -> None:
        """Refresh credentials if expired and persist the updated token file."""
        try:
            if not self.creds:
                return
            if self.creds.expired and getattr(self.creds, "refresh_token", None):
                if getattr(config, "DEBUG_MODE", False):
                    print(f"🔄 Refreshing expired credentials for {self.user}...")
                self.creds.refresh(Request())
                token_path = self._get_user_config()["token"]
                with open(token_path, 'w') as token_file:
                    token_file.write(self.creds.to_json())
                if getattr(config, "DEBUG_MODE", False):
                    print(f"💾 Saved refreshed credentials for {self.user} to {token_path}")
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"⚠️ Failed to refresh credentials for {self.user}: {e}")
    
    def get_events(self, num_events: int = 10, time_min: str = None, time_max: str = None, calendar_id: str = 'primary') -> List[Dict]:
        """
        Get calendar events.
        
        Args:
            num_events: Maximum number of events to retrieve
            time_min: Minimum time for events (ISO format). If None, uses current time
            time_max: Maximum time for events (ISO format). If None, no upper limit
            calendar_id: Calendar ID to query (default: 'primary')
            
        Returns:
            List of event dictionaries
        """
        try:
            # Ensure credentials are fresh and service is available
            self._maybe_refresh_credentials()
            if not self.service:
                self._initialize_credentials()
                if not self.service:
                    print(f"❌ Calendar service not initialized for {self.user}")
                    return []
            
            # If specific calendar_id is requested, only query that calendar
            if calendar_id != 'primary':
                calendars_to_query = [{'id': calendar_id}]
            else:
                # Query all calendars
                calendar_list = self.service.calendarList().list().execute()
                calendars_to_query = calendar_list.get('items', [])
            
            all_events = []
            
            for cal in calendars_to_query:
                cal_id = cal['id']
                
                now = datetime.now(timezone.utc).isoformat() if time_min is None else time_min
                
                events_result = self.service.events().list(
                    calendarId=cal_id,
                    timeMin=now,
                    timeMax=time_max,
                    maxResults=num_events,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                # Add calendar_id to each event
                events = events_result.get('items', [])
                for event in events:
                    event['calendar_id'] = cal_id
                
                all_events.extend(events)
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"📅 Retrieved {len(all_events)} events from {len(calendars_to_query)} calendars for {self.user}")
            
            # Sort by start time using proper datetime parsing
            def get_event_start_time(event):
                """Parse event start time for proper sorting."""
                start = event.get('start', {})
                dt_str = start.get('dateTime') or start.get('date')
                if not dt_str:
                    return datetime.max.replace(tzinfo=timezone.utc)
                try:
                    # Handle full datetime (e.g., "2026-01-03T12:00:00-05:00")
                    if 'T' in dt_str:
                        # Parse ISO format with timezone
                        dt_str = dt_str.replace('Z', '+00:00')
                        return datetime.fromisoformat(dt_str)
                    else:
                        # All-day event (just date "2026-01-03")
                        # Set to midnight UTC for comparison
                        return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    return datetime.max.replace(tzinfo=timezone.utc)
            
            all_events.sort(key=get_event_start_time)
            
            # Limit to requested number of events
            limited_events = all_events[:num_events]
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"📅 Returning {len(limited_events)} events (limited from {len(all_events)} total)")
            
            return limited_events
            
        except HttpError as e:
            error_msg = f"Google Calendar API error: {e}"
            self.error_message = error_msg
            print(f"❌ {error_msg}")
            return []
        except Exception as e:
            error_msg = f"Failed to get calendar events: {e}"
            self.error_message = error_msg
            print(f"❌ {error_msg}")
            return []
    
    def get_formatted_events(self, num_events: int = 10, time_min: str = None) -> List[Dict]:
        """Get events with formatting applied."""
        raw_events = self.get_events(num_events, time_min)
        return [self.format_event(event) for event in raw_events]
    
    def format_event(self, event: Dict) -> Dict[str, str]:
        """
        Format event data for easier consumption.
        
        Args:
            event: Raw event dictionary from Google Calendar API
            
        Returns:
            Formatted event dictionary with standardized fields
        """
        try:
            # Handle start time
            start_raw = event['start'].get('dateTime', event['start'].get('date'))
            start_parts = start_raw.split("T")
            start_date = start_parts[0]
            start_time = start_parts[1].split("-")[0] if len(start_parts) > 1 else "All Day"
            
            # Handle end time
            end_raw = event['end'].get('dateTime', event['end'].get('date'))
            end_parts = end_raw.split("T")
            end_date = end_parts[0]
            end_time = end_parts[1].split("-")[0] if len(end_parts) > 1 else "All Day"
            
            return {
                'id': event.get('id', ''),
                'calendar_name': self.calendar_id_to_name(event.get('calendar_id', '')),
                'summary': event.get('summary', 'No Title'),
                'description': event.get('description', ''),
                'start_date': start_date,
                'start_time': start_time,
                'end_date': end_date,
                'end_time': end_time,
                'location': event.get('location', ''),
                'status': event.get('status', ''),
                'all_day': 'dateTime' not in event['start']
            }
            
        except Exception as e:
            if getattr(config, "DEBUG_MODE", False):
                print(f"⚠️ Error formatting event: {e}")
            return {
                'id': event.get('id', ''),
                'calendar_name': 'Unknown',
                'summary': event.get('summary', 'Error formatting event'),
                'description': '',
                'start_date': 'Unknown',
                'start_time': 'Unknown',
                'end_date': 'Unknown', 
                'end_time': 'Unknown',
                'location': '',
                'status': '',
                'all_day': False
            }
    
    def display_events_details(self, events_list: List[Dict]) -> None:
        """
        Display event details in a readable format.
        
        Args:
            events_list: List of event dictionaries (raw or formatted)
        """
        if not events_list:
            print("📅 No events to display")
            return
        
        print(f"📅 Displaying {len(events_list)} events for {self.user}:")
        print("-" * 80)
        
        for event in events_list:
            if 'start_date' in event:
                formatted = event
            else:
                formatted = self.format_event(event)
            
            print(f"📌 {formatted['summary']}")
            print(f"   📅 Calendar: {formatted['calendar_name']}")
            print(f"   📅 {formatted['start_date']} {formatted['start_time']} - {formatted['end_time']}")
            if formatted['location']:
                print(f"   📍 {formatted['location']}")
            if formatted['description']:
                print(f"   📝 {formatted['description'][:100]}{'...' if len(formatted['description']) > 100 else ''}")
            print()
    
    def get_todays_events(self) -> List[Dict]:
        """
        Get today's events with proper error handling.
        
        Returns:
            List of formatted events for today
        """
        try:
            # Get start of today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            time_min = today_start.isoformat() + 'Z'
            
            # Get events for today
            all_events = self.get_events(num_events=50, time_min=time_min)
            
            # Filter for today only
            today_str = datetime.now().strftime('%Y-%m-%d')
            todays_events = []
            
            for event in all_events:
                formatted = self.format_event(event)
                if formatted['start_date'] == today_str:
                    todays_events.append(formatted)
                elif formatted['start_date'] > today_str:
                    # Events are sorted by start time, so we can break here
                    break
            
            if getattr(config, "DEBUG_MODE", False):
                print(f"📅 Found {len(todays_events)} events for today")
            
            return todays_events
            
        except Exception as e:
            error_msg = f"Failed to get today's events: {e}"
            self.error_message = error_msg
            print(f"❌ {error_msg}")
            return []
    
    def get_next_event(self) -> Optional[Dict]:
        """
        Get and display the next upcoming event.
        
        Returns:
            Formatted event dictionary or None if no upcoming events
        """
        try:
            events = self.get_events(num_events=1)
            if events:
                formatted_event = self.format_event(events[0])                
                return formatted_event
            else:
                print(f"📅 No upcoming events found for {self.user}")
                return None
            
        except Exception as e:
            error_msg = f"Failed to get next event: {e}"
            self.error_message = error_msg
            print(f"❌ {error_msg}")
            return None
    
    def get_last_event(self) -> Optional[Dict]:
        """
        Get and display the most recent event whose start time is before now.
        
        Returns:
            Formatted event dictionary or None if no past events
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get recent events from a few days back to current time
            past_start_time = (current_time - timedelta(days=7)).isoformat().replace('+00:00', '') + 'Z'
            current_time_str = current_time.isoformat().replace('+00:00', '') + 'Z'
            
            # Get events up to current time
            events = self.get_events(
                num_events=100, 
                time_min=past_start_time, 
                time_max=current_time_str
            )
            
            if events:
                # Find the most recent event that has already started
                most_recent_past_event = None
                for event in reversed(events):  # Go backwards to find most recent
                    event_start = event['start'].get('dateTime', event['start'].get('date'))
                    # Parse the event start time
                    if 'T' in event_start:
                        event_start_dt = datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                    else:
                        # All-day event
                        event_start_dt = datetime.fromisoformat(event_start + 'T00:00:00+00:00')
                    
                    if event_start_dt < current_time:
                        most_recent_past_event = event
                        break
                
                if most_recent_past_event:
                    formatted_event = self.format_event(most_recent_past_event)
                    return formatted_event
                else:
                    print(f"📅 No past events found for {self.user}")
                    return None
            else:
                print(f"📅 No events found for {self.user}")
                return None
            
        except Exception as e:
            error_msg = f"Failed to get last event: {e}"
            self.error_message = error_msg
            print(f"❌ {error_msg}")
            return None
    
    def get_events_summary(self, num_events: int = 5) -> str:
        """
        Get a text summary of upcoming events suitable for voice responses.
        
        Args:
            num_events: Number of events to include in summary
            
        Returns:
            Text summary of events
        """
        try:
            events = self.get_formatted_events(num_events)
            
            if not events:
                return "You have no upcoming events in your calendar."
            
            if len(events) == 1:
                event = events[0]
                return f"Your next event is '{event['summary']}' on {event['start_date']} at {event['start_time']}."
            
            summary_parts = [f"You have {len(events)} upcoming events:"]
            
            for i, event in enumerate(events[:3], 1):  # Limit to first 3 for voice
                time_desc = event['start_time'] if event['start_time'] != 'All Day' else 'all day'
                summary_parts.append(f"{i}. '{event['summary']}' on {event['start_date']} at {time_desc}")
            
            if len(events) > 3:
                summary_parts.append(f"...and {len(events) - 3} more events.")
            
            return " ".join(summary_parts)
            
        except Exception:
            return "I'm having trouble accessing your calendar right now."

    def display_all_calendar_types(self):
        self._maybe_refresh_credentials()
        calendar_list = self.service.calendarList().list().execute()

        for cal in calendar_list['items']:
            cal_id = cal['id']
            print(self.calendar_id_to_name(cal_id))
    
    # Methods expected by calendar_data tool
    def get_upcoming_events(self, calendar_name: str = "primary", max_results: int = 10, include_past: bool = False) -> List[Dict]:
        """Get upcoming events for the calendar tool."""
        try:
            # Map calendar name to calendar ID if needed
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            
            # Use existing get_events method
            time_min = None if include_past else datetime.now(timezone.utc).isoformat()
            events = self.get_events(num_events=max_results, time_min=time_min, calendar_id=calendar_id)
            
            return events
        except Exception:
            return []
    
    def get_day_events(self, date: str, calendar_name: str = "primary", include_past: bool = False) -> List[Dict]:
        """Get events for a specific day for the calendar tool."""
        try:
            # Parse the date string (YYYY-MM-DD format)
            target_date = datetime.strptime(date, "%Y-%m-%d")
            
            # Set time range for the day
            start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            end_time = start_time.replace(hour=23, minute=59, second=59)
            
            # Map calendar name to calendar ID if needed
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            
            events = self.get_events(
                num_events=50,  # Get more events for a full day
                time_min=start_time.isoformat(),
                time_max=end_time.isoformat(),
                calendar_id=calendar_id
            )
            
            return events
        except Exception:
            return []
    
    def get_week_events(self, calendar_name: str = "primary", include_past: bool = False) -> List[Dict]:
        """Get events for the current week for the calendar tool."""
        try:
            # Get start and end of current week
            now = datetime.now(timezone.utc)
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            # Map calendar name to calendar ID if needed
            calendar_id = calendar_name if calendar_name == "primary" else calendar_name
            
            time_min = start_of_week.isoformat() if include_past else datetime.now(timezone.utc).isoformat()
            
            events = self.get_events(
                num_events=100,  # Get more events for a full week
                time_min=time_min,
                time_max=end_of_week.isoformat(),
                calendar_id=calendar_id
            )
            
            return events
        except Exception:
            return []
    
    def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new calendar event for the calendar tool."""
        try:
            self._maybe_refresh_credentials()
            if not self.service:
                raise Exception("Calendar service not initialized")
            
            # Build event object from event_data
            start_datetime = f"{event_data['date']}T{event_data['start_time']}:00"
            end_datetime = f"{event_data['date']}T{event_data['end_time']}:00"
            
            # Apply timezone if provided (use timeZone field for IANA IDs),
            # otherwise use default from config if present.
            time_zone = event_data.get('time_zone') or getattr(config, 'DEFAULT_TIME_ZONE', None)
            if time_zone and isinstance(time_zone, str) and '/' in time_zone:
                event_body = {
                    'summary': event_data['title'],
                    'description': event_data.get('description', ''),
                    'start': {
                        'dateTime': start_datetime,
                        'timeZone': time_zone,
                    },
                    'end': {
                        'dateTime': end_datetime,
                        'timeZone': time_zone,
                    },
                }
            else:
                # Default to UTC when no valid IANA zone is provided
                start_datetime += "Z"
                end_datetime += "Z"
                event_body = {
                    'summary': event_data['title'],
                    'description': event_data.get('description', ''),
                    'start': {
                        'dateTime': start_datetime,
                    },
                    'end': {
                        'dateTime': end_datetime,
                    },
                }
            
            # Add location if provided
            if event_data.get('location'):
                event_body['location'] = event_data['location']
            
            # Add attendees if provided
            if event_data.get('attendees'):
                event_body['attendees'] = [{'email': email} for email in event_data['attendees']]
            
            # Get calendar ID - try to resolve name to ID
            requested_calendar = event_data.get('calendar_name', 'primary')
            calendar_id = 'primary'  # Default fallback
            
            if requested_calendar != 'primary':
                # Map calendar name to ID if possible
                try:
                    resolved = self.calendar_name_to_id(requested_calendar)
                    # Verify the calendar exists by checking if we found an actual ID
                    _, name_to_id = self.get_calendar_mappings()
                    # Check if resolved ID exists in our known calendars
                    id_to_name, _ = self.get_calendar_mappings()
                    if resolved in id_to_name or resolved in name_to_id.values():
                        calendar_id = resolved
                    else:
                        # Calendar doesn't exist, fallback to primary
                        print(f"⚠️ Calendar '{requested_calendar}' not found, using primary")
                except Exception:
                    pass  # Stick with primary fallback
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body
            ).execute()
            
            return {
                'id': created_event.get('id'),
                'htmlLink': created_event.get('htmlLink'),
                'summary': created_event.get('summary'),
                'start': created_event.get('start'),
                'end': created_event.get('end'),
            }
            
        except Exception as e:
            raise Exception(f"Failed to create event: {str(e)}")



def main():
    """Main function for standalone testing."""
    # Enable debug mode for testing
    if hasattr(config, 'DEBUG_MODE'):
        config.DEBUG_MODE = True
    
    print("📅 Google Calendar Component Test")
    print("=" * 50)
    
    calendar = CalendarComponent(user="morgan_personal")
    calendar.display_all_calendar_types()
    print("=" * 50)
    calendar.display_events_details(calendar.get_events(num_events=10))



if __name__ == '__main__':
    main()
