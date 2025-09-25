"""
Improved Get Notifications Tool using ImprovedBaseTool.

This tool retrieves pending notifications with enhanced parameter descriptions,
better type safety, and comprehensive filtering options.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from mcp_server.improved_base_tool import ImprovedBaseTool
from config import LOG_TOOLS

class ImprovedGetNotificationsTool(ImprovedBaseTool):
    """Enhanced tool to retrieve pending notifications with comprehensive filtering and metadata."""
    
    name = "improved_get_notifications"
    description = "Check for pending notifications with comprehensive filtering options. Use when user asks about notifications, messages, emails, or news updates. Supports filtering by user, type, and limiting results. Always returns detailed notification content with timestamps and metadata. IMPORTANT: Only query ONE user per request - never multiple users simultaneously."
    version = "1.1.0"
    
    def __init__(self):
        """Initialize the improved notifications tool."""
        super().__init__()
        
        # Path to app_state.json
        self.state_file = Path(__file__).parent.parent.parent / "core" / "state_management" / "app_state.json"
        
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool with detailed parameter descriptions.
        
        Returns:
            Comprehensive JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "user": {
                    "type": "string",
                    "description": "Which user's notifications to check. Must specify exactly ONE user per request - NEVER query multiple users simultaneously. Each user has separate notification queues with different permissions and content. Morgan typically has work emails and system notifications, Spencer has personal and academic notifications.",
                    "enum": ["Morgan", "Spencer"],
                    "default": "Morgan"
                },
                "type_filter": {
                    "type": "string",
                    "description": "Filter notifications by specific type. 'email' includes work and personal messages, calendar invites, and important communications. 'news' includes system updates, app notifications, and announcements. 'other' covers calendar reminders, task notifications, and miscellaneous alerts. Use 'all' to retrieve notifications of all types without filtering.",
                    "enum": ["email", "news", "other", "all"],
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of notifications to return in a single request. Helps manage response size and processing time. Lower values (1-5) for quick checks, higher values (10-50) for comprehensive reviews. Each notification includes full content, so consider bandwidth and readability.",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                },
                "mark_as_read": {
                    "type": "boolean",
                    "description": "Whether to mark notifications as read and permanently remove them from the pending queue after retrieval. Set to true when user wants to 'clear' or 'dismiss' notifications after viewing. Set to false when user just wants to 'check' or 'see' notifications without removing them. Cannot be undone once marked as read.",
                    "default": False
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include additional metadata such as timestamps, priority levels, source application, and read status. Metadata helps with notification management but increases response size. Useful for detailed analysis or when user asks about 'when' or 'from where' notifications originated.",
                    "default": True
                },
                "priority_filter": {
                    "type": "string",
                    "description": "Filter notifications by priority level. 'high' for urgent items requiring immediate attention, 'normal' for standard notifications, 'low' for informational items. Use 'all' to include notifications of any priority level.",
                    "enum": ["high", "normal", "low", "all"],
                    "default": "all"
                }
            },
            "required": []  # All parameters are optional with sensible defaults
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the improved notifications tool.
        
        Args:
            params: Tool parameters matching the schema
            
        Returns:
            Dictionary containing notifications and comprehensive metadata
        """
        try:
            # Extract parameters with defaults
            user = params.get("user", "Morgan")
            # Normalize user input to title-case and handle simple aliases
            try:
                if isinstance(user, str):
                    lowered = user.strip().lower()
                    if lowered in {"morgan", "spencer"}:
                        user = lowered.title()
                    elif lowered in {"me", "my", "default"}:
                        user = "Morgan"
                    else:
                        user = user.title()
            except Exception:
                user = "Morgan"
            type_filter = params.get("type_filter", "all")
            limit = params.get("limit", 10)
            mark_as_read = params.get("mark_as_read", False)
            include_metadata = params.get("include_metadata", True)
            priority_filter = params.get("priority_filter", "all")
            
            # Validate parameters
            if user not in ["Morgan", "Spencer"]:
                return {
                    "success": False,
                    "error": f"Invalid user '{user}'. Must be 'Morgan' or 'Spencer'",
                    "valid_users": ["Morgan", "Spencer"]
                }
            
            if LOG_TOOLS:
                # Log to stderr via logging so it shows in the agent terminal
                self.logger.info("Executing Tool: Notifications -- %s", params)
            
            # Load notifications from state file
            notifications_data = self._load_notifications()
            
            # Get user's notifications
            user_notifications = notifications_data.get(user, [])
            
            # Apply type filter
            if type_filter != "all":
                user_notifications = [
                    notif for notif in user_notifications 
                    if notif.get("type", "other").lower() == type_filter.lower()
                ]
            
            # Apply priority filter
            if priority_filter != "all":
                user_notifications = [
                    notif for notif in user_notifications
                    if notif.get("priority", "normal").lower() == priority_filter.lower()
                ]
            
            # Sort by priority and timestamp (high priority first, then most recent)
            def _sort_key(x: Dict[str, Any]):
                priority_value = {"high": 0, "normal": 1, "low": 2}.get(
                    str(x.get("priority", "normal")).lower(), 1
                )
                ts = x.get("timestamp")
                ts_num = self._coerce_timestamp(ts)
                return (priority_value, -ts_num)

            user_notifications.sort(key=_sort_key)
            
            # Apply limit
            total_available = len(user_notifications)
            limited_notifications = user_notifications[:limit]
            
            # Process notifications for output
            processed_notifications = []
            for notif in limited_notifications:
                processed_notif = notif.copy()
                
                # Optionally remove metadata
                if not include_metadata:
                    # Remove metadata fields but keep essential info
                    metadata_fields = ["timestamp", "priority", "source", "read_status", "id"]
                    for field in metadata_fields:
                        processed_notif.pop(field, None)
                else:
                    # Add human-readable timestamp if available
                    if "timestamp" in processed_notif:
                        try:
                            from datetime import datetime, timezone
                            ts = processed_notif["timestamp"]
                            if isinstance(ts, (int, float)):
                                processed_notif["human_time"] = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception as e:
                            self.logger.error(f"Error: {e}")
                            pass
                
                processed_notifications.append(processed_notif)
            
            # Mark as read if requested
            marked_count = 0
            if mark_as_read and limited_notifications:
                notification_ids = [n.get("id") for n in limited_notifications if n.get("id")]
                marked_count = self._mark_notifications_as_read(user, notification_ids)
            
            # Prepare response with comprehensive information
            response = {
                "success": True,
                "user": user,
                "type_filter": type_filter,
                "priority_filter": priority_filter,
                "notifications": processed_notifications,
                "count": len(processed_notifications),
                "total_available": total_available,
                "has_more": total_available > limit,
                "marked_as_read": mark_as_read,
                "marked_count": marked_count,
                "metadata_included": include_metadata,
                "filters_applied": {
                    "type": type_filter,
                    "priority": priority_filter,
                    "limit": limit
                }
            }
            
            # Add summary information
            if processed_notifications:
                type_counts = {}
                priority_counts = {}
                for notif in user_notifications:  # Use full list for counts
                    notif_type = notif.get("type", "other")
                    notif_priority = notif.get("priority", "normal")
                    type_counts[notif_type] = type_counts.get(notif_type, 0) + 1
                    priority_counts[notif_priority] = priority_counts.get(notif_priority, 0) + 1
                
                response["summary"] = {
                    "total_by_type": type_counts,
                    "total_by_priority": priority_counts,
                    "most_recent": processed_notifications[0].get("human_time") if processed_notifications else None
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing improved notifications tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "user": params.get("user", "unknown"),
                "count": 0,
                "notifications": [],
                "total_available": 0
            }
    
    def _load_notifications(self) -> Dict[str, List[Dict]]:
        """Load notifications from state, including email topic entries.

        Supports structures:
        1) New format: state["notifications"][user] -> list of normalized notifications
        2) Legacy format: state["autonomous_state"]["notification_queue"][user]["notifications"]
        3) Emails list: state["autonomous_state"]["notification_queue"][user]["emails"] -> list of email topic entries
        """
        try:
            if not self.state_file.exists():
                self.logger.warning(f"State file not found: {self.state_file}")
                return {}

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Prefer new format if present
            if isinstance(state.get("notifications"), dict):
                return state["notifications"]

            # Fallback to legacy format and normalize; also merge 'emails' entries
            autonomous_state = state.get("autonomous_state", {})
            notification_queue = autonomous_state.get("notification_queue", {})
            if not isinstance(notification_queue, dict):
                return {}

            normalized: Dict[str, List[Dict]] = {}
            for user, user_data in notification_queue.items():
                raw_list = (user_data or {}).get("notifications", [])
                normalized_list: List[Dict[str, Any]] = []
                for raw in raw_list:
                    # Normalize fields
                    content = raw.get("notification_content") or raw.get("content") or ""
                    raw_type = raw.get("type")
                    notif_type = self._normalize_type(raw_type, content)
                    priority = self._normalize_priority(raw.get("priority"))
                    timestamp = self._coerce_timestamp(raw.get("timestamp"))
                    source = raw.get("source", "unknown")

                    # Generate a stable ID for mark-as-read matching
                    notif_id = raw.get("id") or self._generate_notification_id(content, notif_type, timestamp)

                    normalized_list.append({
                        "id": notif_id,
                        "content": content,
                        "type": notif_type,
                        "priority": priority,
                        "timestamp": timestamp,
                        "source": source,
                        "read_status": raw.get("read_status", "unread")
                    })

                # Merge 'emails' queue items as notifications of type 'email'
                emails_list = (user_data or {}).get("emails", [])
                for email_item in emails_list:
                    try:
                        email_content = email_item.get("content", "")
                        email_title = email_item.get("title") or "Email Update"
                        content = email_content
                        notif_type = "email"
                        priority = self._normalize_priority(email_item.get("priority"))
                        timestamp = self._coerce_timestamp(email_item.get("timestamp"))
                        source = email_item.get("source", "email_summarizer")
                        notif_id = email_item.get("id") or self._generate_notification_id(
                            f"{email_title}|{email_content}", notif_type, timestamp
                        )

                        normalized_entry = {
                            "id": notif_id,
                            "content": content,
                            "type": notif_type,
                            "priority": priority,
                            "timestamp": timestamp,
                            "source": source,
                            "read_status": email_item.get("read_status", "unread"),
                            # Preserve helpful metadata
                            "title": email_title,
                            "topic": email_item.get("topic"),
                            "email_ids": email_item.get("email_ids"),
                        }
                        normalized_list.append(normalized_entry)
                    except Exception:
                        # Skip malformed email entries
                        continue

                # Merge 'news' data as a notification of type 'news'
                news_data = (user_data or {}).get("news")
                if news_data and isinstance(news_data, dict):
                    try:
                        news_summary = news_data.get("summary", "")
                        news_title = "Tech News Summary"
                        content = news_summary
                        notif_type = "news"
                        priority = "normal"  # News summaries are typically normal priority
                        
                        # Use generated_at timestamp if available, otherwise current time
                        generated_at = news_data.get("generated_at")
                        if generated_at:
                            try:
                                from datetime import datetime
                                timestamp = int(datetime.fromisoformat(generated_at).timestamp())
                            except Exception:
                                timestamp = self._coerce_timestamp(None)
                        else:
                            timestamp = self._coerce_timestamp(None)
                        
                        source = "news_summarizer"
                        notif_id = self._generate_notification_id(
                            f"{news_title}|{news_summary}", notif_type, timestamp
                        )

                        normalized_entry = {
                            "id": notif_id,
                            "content": content,
                            "type": notif_type,
                            "priority": priority,
                            "timestamp": timestamp,
                            "source": source,
                            "read_status": "unread",
                            # Preserve helpful metadata
                            "title": news_title,
                            "generated_at": generated_at,
                            "source_articles_count": news_data.get("source_articles_count"),
                            "source_file": news_data.get("source_file"),
                        }
                        normalized_list.append(normalized_entry)
                    except Exception:
                        # Skip malformed news entries
                        continue
                normalized[user] = normalized_list

            return normalized

        except Exception as e:
            self.logger.error(f"Error loading notifications: {e}")
            return {}

    def _generate_notification_id(self, content: str, notif_type: str, timestamp: Any) -> str:
        """Generate a deterministic ID from content/type/timestamp for legacy entries."""
        try:
            import hashlib
            base = f"{notif_type}|{timestamp or ''}|{content}".encode("utf-8", errors="ignore")
            return hashlib.sha1(base).hexdigest()[:16]
        except Exception:
            # Fallback non-cryptographic id
            return f"{notif_type}-{str(timestamp)}-{abs(hash(content)) % (10**8)}"

    def _determine_type_from_content(self, content: str) -> str:
        """Heuristic to infer type from content (legacy compatibility)."""
        txt = (content or "").lower()
        if "📧" in content or "email" in txt:
            return "email"
        if "📰" in content or "news" in txt:
            return "news"
        return "other"

    def _normalize_type(self, raw_type: Optional[str], content: str) -> str:
        """Normalize type; map unknown types to 'other' while preserving email/news."""
        if not raw_type:
            return self._determine_type_from_content(content)
        t = str(raw_type).lower()
        if t in {"email", "news", "other"}:
            return t
        return "other"

    def _normalize_priority(self, raw_priority: Optional[str]) -> str:
        """Normalize priority to high/normal/low; default 'normal'."""
        if not raw_priority:
            return "normal"
        p = str(raw_priority).lower()
        return p if p in {"high", "normal", "low"} else "normal"

    def _coerce_timestamp(self, raw_ts: Any) -> int:
        """Convert timestamp to an integer epoch seconds; default to 0 if missing/invalid."""
        try:
            if raw_ts is None:
                return 0
            if isinstance(raw_ts, (int, float)):
                return int(raw_ts)
            # Try parse from string
            from datetime import datetime
            try:
                # ISO-like or common formats; best-effort
                return int(datetime.fromisoformat(str(raw_ts)).timestamp())
            except Exception:
                return 0
        except Exception:
            return 0
    
    def _mark_notifications_as_read(self, user: str, notification_ids: List[str]) -> int:
        """
        Mark specified notifications as read.
        
        Args:
            user: User whose notifications to mark
            notification_ids: List of notification IDs to mark as read
            
        Returns:
            Number of notifications actually marked as read
        """
        try:
            if not self.state_file.exists():
                return 0
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            marked_count = 0

            if isinstance(state.get("notifications"), dict):
                # New format
                user_notifications = state.get("notifications", {}).get(user, [])
                initial_count = len(user_notifications)
                filtered_notifications = [
                    notif for notif in user_notifications 
                    if notif.get("id") not in notification_ids
                ]
                if "notifications" not in state:
                    state["notifications"] = {}
                state["notifications"][user] = filtered_notifications
                marked_count = initial_count - len(filtered_notifications)
            else:
                # Legacy format
                autonomous_state = state.get("autonomous_state", {})
                queue = autonomous_state.get("notification_queue", {})
                user_block = queue.get(user, {})
                raw_list = user_block.get("notifications", [])

                # Remove by regenerating IDs from legacy items
                remaining = []
                for raw in raw_list:
                    content = raw.get("notification_content") or raw.get("content") or ""
                    notif_type = raw.get("type") or self._determine_type_from_content(content)
                    timestamp = raw.get("timestamp")
                    raw_id = raw.get("id") or self._generate_notification_id(content, notif_type, timestamp)
                    if raw_id not in notification_ids:
                        remaining.append(raw)

                initial_count = len(raw_list)
                user_block["notifications"] = remaining
                
                # Check if any news notifications were marked as read and remove the news key
                news_data = user_block.get("news")
                if news_data and isinstance(news_data, dict):
                    # Generate the news notification ID to check if it should be removed
                    news_summary = news_data.get("summary", "")
                    news_title = "Tech News Summary"
                    generated_at = news_data.get("generated_at")
                    if generated_at:
                        try:
                            from datetime import datetime
                            timestamp = int(datetime.fromisoformat(generated_at).timestamp())
                        except Exception:
                            timestamp = 0
                    else:
                        timestamp = 0
                    
                    news_id = self._generate_notification_id(f"{news_title}|{news_summary}", "news", timestamp)
                    if news_id in notification_ids:
                        # Remove the news key entirely
                        user_block.pop("news", None)
                        marked_count += 1  # Count the news notification as marked
                
                queue[user] = user_block
                autonomous_state["notification_queue"] = queue
                state["autonomous_state"] = autonomous_state
                marked_count += initial_count - len(remaining)

            # Save updated state
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"Marked {marked_count} notifications as read for {user}")
            return marked_count
            
        except Exception as e:
            self.logger.error(f"Error marking notifications as read: {e}")
            return 0