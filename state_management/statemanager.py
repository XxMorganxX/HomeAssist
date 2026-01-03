import json
import os
import uuid
from datetime import datetime
from pathlib import Path

# Get the absolute path to the state management directory
STATE_DIR = Path(__file__).parent.absolute()
STATE_FILE = STATE_DIR / "app_state.json"


def _prompt_initial_setup() -> dict:
    """
    Prompt user for essential setup information on first boot.
    Returns a dictionary with the initial state structure.
    """
    print("\n" + "=" * 60)
    print("🏠 HOMEASSIST FIRST-TIME SETUP")
    print("=" * 60)
    print("\nWelcome! Let's configure your assistant.\n")
    
    # Get primary user name
    while True:
        primary_user = input("👤 Enter your name (primary user): ").strip()
        if primary_user:
            break
        print("   Name cannot be empty. Please try again.")
    
    # Get default Spotify user (optional)
    print(f"\n📻 Spotify Integration (press Enter to skip)")
    spotify_user = input(f"   Default Spotify user [{primary_user}]: ").strip()
    if not spotify_user:
        spotify_user = primary_user
    
    # Default lighting scene
    print(f"\n💡 Lighting Preferences")
    print("   Options: mood, party, movie, all_on, all_off")
    lighting_scene = input("   Default lighting scene [all_on]: ").strip().lower()
    if lighting_scene not in ["mood", "party", "movie", "all_on", "all_off"]:
        lighting_scene = "all_on"
    
    # Default volume
    print(f"\n🔊 Default system volume (0-100)")
    volume_input = input("   Volume level [50]: ").strip()
    try:
        volume_level = int(volume_input) if volume_input else 50
        volume_level = max(0, min(100, volume_level))
    except ValueError:
        volume_level = 50
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! Starting HomeAssist...")
    print("=" * 60 + "\n")
    
    # Build initial state structure
    return {
        "user_state": {
            "primary_user": primary_user,
            "created_at": datetime.now().isoformat()
        },
        "chat_controlled_state": {
            "current_spotify_user": spotify_user,
            "lighting_scene": lighting_scene,
            "volume_level": str(volume_level),
            "do_not_disturb": "false"
        },
        "autonomous_state": {
            "notification_queue": {
                primary_user: {
                    "notifications": [],
                    "emails": []
                }
            }
        }
    }


class Notification:
    def __init__(self, intended_recipient, notification_content, relevancy):
        self.intended_recipient = intended_recipient
        self.notification_content = notification_content
        self.relevancy = relevancy
     
    def to_dict(self):
        return {
            "intended_recipient": self.intended_recipient,
            "notification_content": self.notification_content,
            "relevant_when": self.relevancy
        }


class StateManager:
    def __init__(self, filepath=None):
        # Always use the same absolute path to the state file
        self.filepath = str(STATE_FILE) if filepath is None else filepath
        print(f"State manager file path: {os.path.abspath(self.filepath)}")
        self.load()
    
    def get_primary_user(self) -> str:
        """
        Get the primary user from state.
        
        Returns:
            Primary user name, or "User" as fallback.
        """
        self.load()
        return self.state.get("user_state", {}).get("primary_user", "User")

    def set(self, state_type, new_state):
        self.load()
        for state_system, sub_sys_state_dict in self.state.items():
            for state in sub_sys_state_dict.keys():
                if state_type == state:
                    self.state[state_system][state] = new_state
                    self.save()
                    return True
        return False

    def get(self, state_type):
        self.load()
        for state_system, sub_sys_state_dict in self.state.items():
            for state in sub_sys_state_dict.keys():
                if state_type == state:
                    return self.state[state_system][state]
        return None

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.state = json.load(f)
        else:
            # Ensure directory exists
            Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
            # Prompt user for initial setup
            self.state = _prompt_initial_setup()
            self.save()
            print(f"📝 State file created at {self.filepath}")
    
    def add_to_notification_queue(self, notification, intended_recipient):
        self.load()
        # Ensure structure exists
        if "autonomous_state" not in self.state or not isinstance(self.state.get("autonomous_state"), dict):
            self.state["autonomous_state"] = {}
        if "notification_queue" not in self.state["autonomous_state"] or not isinstance(self.state["autonomous_state"].get("notification_queue"), dict):
            self.state["autonomous_state"]["notification_queue"] = {}
        if intended_recipient not in self.state["autonomous_state"]["notification_queue"] or not isinstance(self.state["autonomous_state"]["notification_queue"].get(intended_recipient), dict):
            self.state["autonomous_state"]["notification_queue"][intended_recipient] = {"notifications": []}
        if "notifications" not in self.state["autonomous_state"]["notification_queue"][intended_recipient] or not isinstance(self.state["autonomous_state"]["notification_queue"][intended_recipient].get("notifications"), list):
            self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"] = []

        self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"].append(notification)
        self.save()
    
    def get_notification_queue(self, intended_recipient):
        self.load()
        return self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"]
    
    def add_emails_to_notification_queue(self, notifications, intended_recipient):
        """Append one or multiple email notifications to the 'emails' list for a recipient.
        Creates the structure if missing.
        """
        self.load()
        # Ensure structure exists
        if "autonomous_state" not in self.state or not isinstance(self.state.get("autonomous_state"), dict):
            self.state["autonomous_state"] = {}
        if "notification_queue" not in self.state["autonomous_state"] or not isinstance(self.state["autonomous_state"].get("notification_queue"), dict):
            self.state["autonomous_state"]["notification_queue"] = {}
        if intended_recipient not in self.state["autonomous_state"]["notification_queue"] or not isinstance(self.state["autonomous_state"]["notification_queue"].get(intended_recipient), dict):
            self.state["autonomous_state"]["notification_queue"][intended_recipient] = {}
        if "emails" not in self.state["autonomous_state"]["notification_queue"][intended_recipient] or not isinstance(self.state["autonomous_state"]["notification_queue"][intended_recipient].get("emails"), list):
            self.state["autonomous_state"]["notification_queue"][intended_recipient]["emails"] = []


        target = self.state["autonomous_state"]["notification_queue"][intended_recipient]["emails"]
        if isinstance(notifications, list):
            target.extend(notifications)
        else:
            target.append(notifications)
        self.save()
    

    def refresh_news_summary(self, summary: dict, user: str = None):
        """
        Write a news summary to a user's notification queue.
        
        Args:
            summary: News summary dict to store
            user: Target user (defaults to primary user)
        """
        self.load()
        
        # Use primary user if not specified
        target_user = user or self.get_primary_user()
        
        # Ensure the autonomous_state structure exists
        if "autonomous_state" not in self.state or not isinstance(self.state.get("autonomous_state"), dict):
            self.state["autonomous_state"] = {}
        
        # Ensure the notification_queue structure exists
        if "notification_queue" not in self.state["autonomous_state"] or not isinstance(self.state["autonomous_state"].get("notification_queue"), dict):
            self.state["autonomous_state"]["notification_queue"] = {}
        
        # Ensure target user's entry exists
        if target_user not in self.state["autonomous_state"]["notification_queue"] or not isinstance(self.state["autonomous_state"]["notification_queue"].get(target_user), dict):
            self.state["autonomous_state"]["notification_queue"][target_user] = {}
        
        # Set the news summary (this will overwrite any existing news key)
        self.state["autonomous_state"]["notification_queue"][target_user]["news"] = summary
        
        self.save()

    def _map_topic_notification(self, raw_notification):
        """Map a topic notification into the main notifications schema, preserving content.

        Keeps extra fields like topic/count/email_ids when present.
        """
        try:
            # Convert timestamp to human-readable format
            raw_ts = raw_notification.get("timestamp")
            if raw_ts:
                timestamp = datetime.fromtimestamp(int(raw_ts)).strftime("%B %d, %Y at %I:%M %p")
            else:
                timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            mapped = {
                "id": raw_notification.get("id") or uuid.uuid4().hex[:16],
                "type": raw_notification.get("notification_type") or raw_notification.get("type") or "email",
                "title": raw_notification.get("title") or "Email Update",
                "content": raw_notification.get("content", ""),
                "priority": raw_notification.get("priority") or "normal",
                "timestamp": timestamp,
                "source": raw_notification.get("source") or "email_summarizer",
                "read_status": raw_notification.get("read_status") or "unread",
            }
            # Preserve optional extra fields
            for extra_key in ("topic", "count", "email_ids"):
                if extra_key in raw_notification:
                    mapped[extra_key] = raw_notification[extra_key]
            return mapped
        except Exception:
            # Fallback minimal mapping
            return {
                "id": uuid.uuid4().hex[:16],
                "type": "email",
                "title": str(raw_notification.get("title") or "Email Update"),
                "content": str(raw_notification.get("content", "")),
                "priority": "normal",
                "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                "source": "email_summarizer",
                "read_status": "unread",
            }

    def add_notifications_to_main_queue(self, notifications, intended_recipient):
        """Append one or multiple topic notifications into the main 'notifications' list.

        Ensures the 'content' field matches exactly what is written to email_notifications.json.
        Accepts a single dict or a list of dicts.
        """
        self.load()
        # Ensure structure exists
        if "autonomous_state" not in self.state or not isinstance(self.state.get("autonomous_state"), dict):
            self.state["autonomous_state"] = {}
        if "notification_queue" not in self.state["autonomous_state"] or not isinstance(self.state["autonomous_state"].get("notification_queue"), dict):
            self.state["autonomous_state"]["notification_queue"] = {}
        if intended_recipient not in self.state["autonomous_state"]["notification_queue"] or not isinstance(self.state["autonomous_state"]["notification_queue"].get(intended_recipient), dict):
            self.state["autonomous_state"]["notification_queue"][intended_recipient] = {"notifications": []}
        if "notifications" not in self.state["autonomous_state"]["notification_queue"][intended_recipient] or not isinstance(self.state["autonomous_state"]["notification_queue"][intended_recipient].get("notifications"), list):
            self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"] = []

        target = self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"]

        if isinstance(notifications, list):
            for raw in notifications:
                target.append(self._map_topic_notification(raw))
        else:
            target.append(self._map_topic_notification(notifications))

        self.save()
    
    
    

# Usage
state = StateManager()

