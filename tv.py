import requests
import os
from dotenv import load_dotenv

load_dotenv()



class TV:
    def __init__(self):
        self.device_id = self.get_tv_id()
        self.url = f"https://api.smartthings.com/v1/devices/{self.device_id}/commands"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('TV_TOKEN')}",
            "Content-Type": "application/json"
        }

    def turn_on(self):
        payload = {
            "commands": [
                {
                    "component": "main",
                    "capability": "switch",
                    "command": "on"  # or "off"
                }
            ]
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        print(response.status_code, response.text)
        
    def turn_off(self):
        payload = {
            "commands": [
                {
                    "component": "main",
                    "capability": "switch",
                    "command": "off"  # or "off"
                }
            ]
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        print(response.status_code, response.text)
    
    def get_tv_id(self, tv_name="Television"):
        headers = {
            "Authorization": f"Bearer {os.getenv('TV_TOKEN')}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.smartthings.com/v1/devices", headers=headers)
        devices = response.json()
        for device in devices['items']:
            if device['label'] == tv_name:
                return device['deviceId']
        return None
    
    @staticmethod
    def display_devices():
        headers = {
            "Authorization": f"Bearer {os.getenv('TV_TOKEN')}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.smartthings.com/v1/devices", headers=headers)
        devices = response.json()

        for device in devices['items']:
            print(device['label'], "-", device['deviceId'])
    
    def control_volume(self, volume):
        payload = {
            "commands": [
                {
                    "component": "main",
                    "capability": "audioVolume",
                    "command": "setVolume",
                    "arguments": [volume]
                }
            ]
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        print(response.status_code, response.text)

    def get_status(self):
        response = requests.get(self.url, headers=self.headers)
        return response.json()


    def display_fireplace(self, channel):
        payload = {
            "commands":[
                {
                    "component":"main",
                    "capability":"tvChannel",
                    "command":"setTvChannel",
                    "arguments":[
                        f"{channel}"
                    ]
                }
            ]
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        print(response.status_code, response.text)

        
tv = TV()
tv.turn_on()
tv.display_fireplace(4587)
tv.turn_off()

