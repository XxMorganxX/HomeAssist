import json
import os
import config
import sys
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
        self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"].append(notification)
        self.save()
    
    def get_notification_queue(self, intended_recipient):
        self.load()
        return self.state["autonomous_state"]["notification_queue"][intended_recipient]["notifications"]
    
    
    

# Usage
state = StateManager()

