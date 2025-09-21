import json
import os
import time
import uuid
from pathlib import Path

# Get the absolute path to the state management directory
STATE_DIR = Path(__file__).parent.absolute()
STATE_FILE = STATE_DIR / "app_state.json"

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
            print(f"State manager file not found at {self.filepath}")
    
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

    def _map_topic_notification(self, raw_notification):
        """Map a topic notification into the main notifications schema, preserving content.

        Keeps extra fields like topic/count/email_ids when present.
        """
        try:
            mapped = {
                "id": raw_notification.get("id") or uuid.uuid4().hex[:16],
                "type": raw_notification.get("notification_type") or raw_notification.get("type") or "email",
                "title": raw_notification.get("title") or "Email Update",
                "content": raw_notification.get("content", ""),
                "priority": raw_notification.get("priority") or "normal",
                "timestamp": int(raw_notification.get("timestamp") or int(time.time())),
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
                "timestamp": int(time.time()),
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

