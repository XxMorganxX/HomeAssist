"""
Get Notifications Tool for MCP Server.
Retrieves pending notifications from app_state.json for the voice assistant.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from mcp_server.base_tool import BaseTool


class GetNotificationsTool(BaseTool):
    """Tool to retrieve pending notifications for users."""
    
    name = "get_notifications"  # Class attribute for tool registry
    description = "Check for pending notifications with filtering options. Use when user asks about notifications, messages, emails, or news updates. Supports filtering by user, type, and limiting results. Always returns detailed notification content."
    
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
                "user": {
                    "type": "string",
                    "description": "Which user's notifications to check. Use 'all' to check both users.",
                    "enum": ["Morgan", "Spencer", "all"],
                    "default": "all"
                },
                "type_filter": {
                    "type": "string",
                    "description": "Filter notifications by type. Use 'all' for all types.",
                    "enum": ["email", "news", "other", "all"],
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of notifications to return (1-50)",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                },
                "mark_as_read": {
                    "type": "boolean", 
                    "description": "Whether to clear notifications after reading them",
                    "default": False
                }
            },
            "required": [],
            "description": self.description
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict:
        """
        Execute the tool to get notifications.
        
        Args:
            params: Dictionary containing user, type_filter, limit, and mark_as_read parameters
            
        Returns:
            Dictionary with notification results
        """
        # Extract parameters with defaults
        user = params.get("user", "all")
        type_filter = params.get("type_filter", "all")
        limit = params.get("limit", 10)
        mark_as_read = params.get("mark_as_read", False)
        
        # Backward compatibility: support old parameter name
        if "user_name" in params:
            user = params["user_name"]
        
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
            if user == "all":
                users_to_check = ["Morgan", "Spencer"]
            elif user in notification_queue:
                users_to_check = [user]
            else:
                users_to_check = []
            
            # Collect notifications
            all_notifications = []
            filtered_counts = {}
            total_counts = {}
            
            for user in users_to_check:
                if user in notification_queue:
                    user_notifications = notification_queue[user].get("notifications", [])
                    total_counts[user] = len(user_notifications)
                    user_filtered = 0
                    
                    for notif in user_notifications:
                        notification_type = self._determine_notification_type(notif)
                        
                        # Apply type filter
                        if type_filter != "all" and notification_type != type_filter:
                            continue
                            
                        user_filtered += 1
                        all_notifications.append({
                            "user": user,
                            "content": notif.get("notification_content", ""),
                            "relevance": notif.get("relevant_when", ""),
                            "type": notification_type,
                            "source": notif.get("source", "unknown")  # Include source info
                        })
                    
                    if user_filtered > 0:
                        filtered_counts[user] = user_filtered
            
            # Apply limit
            if len(all_notifications) > limit:
                all_notifications = all_notifications[:limit]
            
            # Clear notifications if requested
            if mark_as_read and all_notifications:
                for user in users_to_check:
                    if user in notification_queue:
                        notification_queue[user]["notifications"] = []
                
                # Save updated state
                with open(self.state_file, 'w') as f:
                    json.dump(app_state, f, indent=2)
            
            # Format response
            filtered_total = len(all_notifications)
            
            if filtered_total == 0:
                if type_filter != "all":
                    summary = f"No {type_filter} notifications found"
                else:
                    summary = "No pending notifications"
            else:
                summary_parts = []
                for user, count in filtered_counts.items():
                    if count > 0:
                        summary_parts.append(f"{count} for {user}")
                
                type_suffix = f" {type_filter}" if type_filter != "all" else ""
                summary = f"Found {filtered_total}{type_suffix} notification(s): " + ", ".join(summary_parts)
                
                if len(all_notifications) < sum(total_counts.values()):
                    summary += f" (limited to {limit})" if filtered_total == limit else ""
            
            # Group notifications by type
            email_notifications = [n for n in all_notifications if n["type"] == "email"]
            news_notifications = [n for n in all_notifications if n["type"] == "news"]
            other_notifications = [n for n in all_notifications if n["type"] == "other"]
            
            return {
                "success": True,
                "summary": summary,
                "total_count": filtered_total,
                "filtered_counts": filtered_counts,
                "total_counts": total_counts,
                "notifications": all_notifications,
                "by_type": {
                    "email": len(email_notifications),
                    "news": len(news_notifications),
                    "other": len(other_notifications)
                },
                "filters_applied": {
                    "user": user,
                    "type_filter": type_filter,
                    "limit": limit
                },
                "marked_as_read": mark_as_read and filtered_total > 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting notifications: {str(e)}"
            }
    
    def _determine_notification_type(self, notification: Dict) -> str:
        """Determine the type of notification based on source field or content."""
        # First check if there's a source field (new format)
        if "source" in notification:
            source = notification["source"].lower()
            if source in ["email", "news", "other"]:
                return source
        
        # Fallback to content-based detection (backward compatibility)
        content = notification.get("notification_content", "")
        content_lower = content.lower()
        if "📧" in content or "email" in content_lower:
            return "email"
        elif "📰" in content or "tech news" in content_lower:
            return "news"
        else:
            return "other"


# Required for MCP tool discovery
tool_instance = GetNotificationsTool()