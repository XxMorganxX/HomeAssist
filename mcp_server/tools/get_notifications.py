"""
Get Notifications Tool for MCP Server.
Retrieves pending notifications from app_state.json for the voice assistant.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from mcp_server.base_tool import BaseTool


class GetNotificationsTool(BaseTool):
    """Tool to retrieve pending notifications for users."""
    
    name = "get_notifications"  # Class attribute for tool registry
    description = "Check for pending notifications. Use when user asks about notifications, messages, emails, or news updates. Returns current notifications with counts and details."
    
    def __init__(self):
        """Initialize the notifications tool."""
        
        # Path to app_state.json
        self.state_file = Path(__file__).parent.parent.parent / "core" / "state_management" / "app_state.json"
        
    def get_schema(self) -> Dict:
        """
        Get the JSON schema for this tool.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "description": "Name of the user to get notifications for (Morgan or Spencer). If not specified, checks both users.",
                    "enum": ["Morgan", "Spencer"]
                },
                "mark_as_read": {
                    "type": "boolean", 
                    "description": "Whether to clear notifications after reading them (default: false)",
                    "default": False
                }
            },
            "required": [],
            "description": self.description
        }
    
    def execute(self, user_name: Optional[str] = None, mark_as_read: bool = False) -> Dict:
        """
        Execute the tool to get notifications.
        
        Args:
            user_name: User to get notifications for (defaults to trying both)
            mark_as_read: Whether to clear notifications after reading
            
        Returns:
            Dictionary with notification results
        """
        try:
            # Load app state
            if not self.state_file.exists():
                return {
                    "success": False,
                    "error": "App state file not found"
                }
            
            with open(self.state_file, 'r') as f:
                app_state = json.load(f)
            
            # Get notification queue
            notification_queue = app_state.get("autonomous_state", {}).get("notification_queue", {})
            
            # Determine which users to check
            if user_name:
                users_to_check = [user_name] if user_name in notification_queue else []
            else:
                # Check both users if no specific user provided
                users_to_check = ["Morgan", "Spencer"]
            
            # Collect notifications
            all_notifications = []
            notification_counts = {}
            
            for user in users_to_check:
                if user in notification_queue:
                    user_notifications = notification_queue[user].get("notifications", [])
                    notification_counts[user] = len(user_notifications)
                    
                    for notif in user_notifications:
                        all_notifications.append({
                            "user": user,
                            "content": notif.get("notification_content", ""),
                            "relevance": notif.get("relevant_when", ""),
                            "type": self._determine_notification_type(notif.get("notification_content", ""))
                        })
            
            # Clear notifications if requested
            if mark_as_read and all_notifications:
                for user in users_to_check:
                    if user in notification_queue:
                        notification_queue[user]["notifications"] = []
                
                # Save updated state
                with open(self.state_file, 'w') as f:
                    json.dump(app_state, f, indent=2)
            
            # Format response
            total_count = sum(notification_counts.values())
            
            if total_count == 0:
                summary = "No pending notifications"
            else:
                summary_parts = []
                for user, count in notification_counts.items():
                    if count > 0:
                        summary_parts.append(f"{count} for {user}")
                summary = f"Found {total_count} notification(s): " + ", ".join(summary_parts)
            
            # Group notifications by type
            email_notifications = [n for n in all_notifications if n["type"] == "email"]
            news_notifications = [n for n in all_notifications if n["type"] == "news"]
            other_notifications = [n for n in all_notifications if n["type"] == "other"]
            
            return {
                "success": True,
                "summary": summary,
                "total_count": total_count,
                "counts_by_user": notification_counts,
                "notifications": all_notifications,
                "by_type": {
                    "email": len(email_notifications),
                    "news": len(news_notifications),
                    "other": len(other_notifications)
                },
                "marked_as_read": mark_as_read and total_count > 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting notifications: {str(e)}"
            }
    
    def _determine_notification_type(self, content: str) -> str:
        """Determine the type of notification based on content."""
        content_lower = content.lower()
        if "📧" in content or "email" in content_lower:
            return "email"
        elif "📰" in content or "tech news" in content_lower:
            return "news"
        else:
            return "other"


# Required for MCP tool discovery
tool_instance = GetNotificationsTool()