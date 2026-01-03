"""
Get Notifications Tool using BaseTool.

This tool retrieves pending notifications with enhanced parameter descriptions,
better type safety, and comprehensive filtering options.

Data sources (in priority order):
1. Supabase notification_sources table (cloud, persistent)
2. Local app_state.json (fallback for offline/legacy)
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timezone
from mcp_server.base_tool import BaseTool
from mcp_server.config import LOG_TOOLS

# Import user config for dynamic user resolution
try:
    from mcp_server.user_config import get_notification_users, get_default_notification_user
except ImportError:
    # Fallback if user_config not available
    def get_notification_users():
        return ["User"]
    def get_default_notification_user():
        return "User"

# Import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None


class GetNotificationsTool(BaseTool):
    """Enhanced tool to retrieve pending notifications with comprehensive filtering and metadata."""
    
    name = "get_notifications"
    description = """Check for pending notifications with comprehensive filtering options.

IMPORTANT - Use type_filter for specific requests:
- type_filter='email' → ONLY email summaries (inbox updates, email digests)
- type_filter='news' → ONLY news summaries (tech news, daily briefings)  
- type_filter='all' → Both email AND news (default)

Examples:
- "any new emails?" → type_filter='email'
- "what's in the news?" → type_filter='news'
- "any notifications?" → type_filter='all'

Only query ONE user per request - never multiple users simultaneously."""
    version = "1.3.0"
    
    def __init__(self):
        """Initialize the notifications tool."""
        super().__init__()
        
        # Path to app_state.json (fallback)
        self.state_file = Path(__file__).parent.parent.parent / "state_management" / "app_state.json"
        
        # Get configured users dynamically
        self._configured_users = get_notification_users()
        self._default_user = get_default_notification_user()
        
        # Initialize Supabase client
        self._supabase_client: Optional[Client] = None
        self._supabase_available = False
        self._init_supabase()
        
    def _init_supabase(self):
        """Initialize Supabase client for notification retrieval."""
        if not SUPABASE_AVAILABLE:
            self.logger.debug("Supabase package not installed, using local state only")
            return
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            self.logger.debug("SUPABASE_URL or SUPABASE_KEY not set, using local state only")
            return
        
        try:
            self._supabase_client = create_client(url, key)
            self._supabase_available = True
            self.logger.debug("Supabase client initialized for notifications")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Supabase client: {e}")
    
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
                    "description": f"Which user's notifications to check. Must specify exactly ONE user per request - NEVER query multiple users simultaneously. Each user has separate notification queues with different permissions and content.",
                    "enum": self._configured_users,
                    "default": self._default_user
                },
                "type_filter": {
                    "type": "string",
                    "description": "CRITICAL: Filter notifications by type. Use 'email' for email summaries ONLY (inbox updates, email digests from the email summarizer). Use 'news' for news summaries ONLY (tech news briefings from the news scraper). Use 'other' for miscellaneous notifications. Use 'all' to get everything. ALWAYS use the specific filter when the user asks about emails OR news separately - do NOT use 'all' when they ask for one specific type.",
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
        Execute the notifications tool.
        
        Args:
            params: Tool parameters matching the schema
            
        Returns:
            Dictionary containing notifications and comprehensive metadata
        """
        try:
            # Extract parameters with defaults
            user = params.get("user", self._default_user)
            # Normalize user input to title-case and handle simple aliases
            try:
                if isinstance(user, str):
                    lowered = user.strip().lower()
                    # Check if user matches any configured user
                    configured_lower = [u.lower() for u in self._configured_users]
                    if lowered in configured_lower:
                        # Find the properly-cased version
                        idx = configured_lower.index(lowered)
                        user = self._configured_users[idx]
                    elif lowered in {"me", "my", "default"}:
                        user = self._default_user
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
            
            # Load notifications with filtering at source level for efficiency
            # This filters at database level in Supabase and in-memory for local state
            notifications_data = self._load_notifications(type_filter=type_filter, user_filter=user)
            
            # Get user's notifications (already filtered by type at load time)
            user_notifications = notifications_data.get(user, [])
            
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
            self.logger.error(f"Error executing notifications tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "user": params.get("user", "unknown"),
                "count": 0,
                "notifications": [],
                "total_available": 0
            }
    
    def _load_notifications(self, type_filter: str = "all", user_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Load notifications from Supabase (primary) and local state (fallback).

        Data sources in priority order:
        1) Supabase notification_sources table (cloud, persistent)
        2) Local app_state.json (fallback for offline/legacy data)
        
        Merges both sources, deduplicating by ID with Supabase taking precedence.
        
        Args:
            type_filter: Filter by notification type ('email', 'news', 'other', 'all')
            user_filter: Optional user to filter by (filters at query level for efficiency)
        """
        notifications: Dict[str, List[Dict]] = {}
        
        # Try Supabase first (primary source) - filters at database level for efficiency
        if self._supabase_available and self._supabase_client:
            try:
                supabase_notifications = self._load_from_supabase(type_filter=type_filter, user_filter=user_filter)
                if supabase_notifications:
                    notifications = supabase_notifications
                    self.logger.debug(f"Loaded notifications from Supabase: {sum(len(v) for v in notifications.values())} total (type={type_filter})")
            except Exception as e:
                self.logger.warning(f"Failed to load from Supabase, falling back to local: {e}")
        
        # Load from local state (fallback/merge)
        local_notifications = self._load_from_local_state(type_filter=type_filter, user_filter=user_filter)
        
        # Merge local notifications with Supabase (Supabase takes precedence)
        # Use content-based deduplication since IDs may differ between sources
        for user, local_list in local_notifications.items():
            if user not in notifications:
                notifications[user] = []
            
            # Build a set of normalized IDs and content hashes from Supabase for deduplication
            existing_ids = set()
            existing_content_hashes = set()
            for n in notifications.get(user, []):
                nid = n.get("id", "")
                existing_ids.add(nid)
                # Also add the base ID without prefix (email_, news_)
                existing_ids.add(self._strip_id_prefix(nid))
                # Add content hash for content-based deduplication
                content_hash = self._content_hash(n.get("content", ""), n.get("title", ""))
                if content_hash:
                    existing_content_hashes.add(content_hash)
            
            # Add local notifications that aren't duplicates
            for notif in local_list:
                notif_id = notif.get("id", "")
                base_id = self._strip_id_prefix(notif_id)
                content_hash = self._content_hash(notif.get("content", ""), notif.get("title", ""))
                
                # Skip if ID matches (with or without prefix) OR content is duplicate
                if notif_id in existing_ids or base_id in existing_ids:
                    continue
                if content_hash and content_hash in existing_content_hashes:
                    continue
                
                notifications[user].append(notif)
                existing_ids.add(notif_id)
                existing_ids.add(base_id)
                if content_hash:
                    existing_content_hashes.add(content_hash)
        
        return notifications
    
    def _strip_id_prefix(self, notif_id: str) -> str:
        """Strip email_ or news_ prefix from notification ID for comparison."""
        if not notif_id:
            return ""
        for prefix in ("email_", "news_"):
            if notif_id.startswith(prefix):
                return notif_id[len(prefix):]
        return notif_id
    
    def _content_hash(self, content: str, title: str) -> str:
        """Generate a simple hash from content and title for deduplication."""
        if not content and not title:
            return ""
        import hashlib
        combined = f"{title}|{content[:200]}"  # Use first 200 chars of content
        return hashlib.md5(combined.encode("utf-8", errors="ignore")).hexdigest()[:12]
    
    def _load_from_supabase(self, type_filter: str = "all", user_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Load notifications from Supabase notification_sources table.
        
        Args:
            type_filter: Filter by source_type ('email', 'news', 'other', 'all')
            user_filter: Optional user_id to filter by
        """
        if not self._supabase_client:
            return {}
        
        try:
            # Build query with filters at database level for efficiency
            query = (
                self._supabase_client.table("notification_sources")
                .select("*")
                .eq("read_status", "unread")
            )
            
            # Apply type filter at database level (most important for separating email/news)
            if type_filter and type_filter != "all":
                query = query.eq("source_type", type_filter)
                self.logger.debug(f"Supabase query filtered by source_type={type_filter}")
            
            # Apply user filter at database level
            if user_filter:
                query = query.eq("user_id", user_filter)
            
            # Execute with ordering and limit
            response = query.order("created_at", desc=True).limit(100).execute()
            
            if not response.data:
                return {}
            
            # Group by user_id
            notifications: Dict[str, List[Dict]] = {}
            for row in response.data:
                user_id = row.get("user_id", "Unknown")
                
                # Parse timestamp
                created_at = row.get("created_at")
                source_generated_at = row.get("source_generated_at")
                timestamp = self._parse_supabase_timestamp(source_generated_at or created_at)
                
                # Map source_type to type
                source_type = row.get("source_type", "other")
                notif_type = source_type if source_type in ["email", "news"] else "other"
                
                # Extract metadata
                metadata = row.get("metadata") or {}
                
                normalized = {
                    "id": row.get("id"),
                    "content": row.get("content", ""),
                    "type": notif_type,
                    "priority": row.get("priority", "normal"),
                    "timestamp": timestamp,
                    "source": f"supabase_{source_type}",
                    "read_status": row.get("read_status", "unread"),
                    "title": row.get("title", ""),
                    # Include metadata fields
                    "topic": metadata.get("topic"),
                    "email_ids": metadata.get("email_ids"),
                    "source_articles_count": metadata.get("source_articles_count"),
                    "batch_id": row.get("batch_id"),
                }
                
                if user_id not in notifications:
                    notifications[user_id] = []
                notifications[user_id].append(normalized)
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"Error loading from Supabase: {e}")
            return {}
    
    def _parse_supabase_timestamp(self, ts_str: Optional[str]) -> int:
        """Parse a Supabase timestamp string to epoch seconds."""
        if not ts_str:
            return 0
        try:
            # Handle ISO format with timezone
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts_str)
            return int(dt.timestamp())
        except Exception:
            return 0
    
    def _load_from_local_state(self, type_filter: str = "all", user_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Load notifications from local app_state.json (fallback).
        
        Supports structures:
        1) New format: state["notifications"][user] -> list of normalized notifications
        2) Legacy format: state["autonomous_state"]["notification_queue"][user]["notifications"]
        3) Emails list: state["autonomous_state"]["notification_queue"][user]["emails"]
        4) News data: state["autonomous_state"]["notification_queue"][user]["news"]
        
        Args:
            type_filter: Filter by type ('email', 'news', 'other', 'all')
            user_filter: Optional user to filter by
        """
        try:
            if not self.state_file.exists():
                self.logger.debug(f"State file not found: {self.state_file}")
                return {}

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Prefer new format if present
            if isinstance(state.get("notifications"), dict):
                result = state["notifications"]
                # Apply filters
                if user_filter:
                    result = {k: v for k, v in result.items() if k == user_filter}
                if type_filter and type_filter != "all":
                    result = {
                        user: [n for n in notifs if n.get("type") == type_filter]
                        for user, notifs in result.items()
                    }
                return result

            # Fallback to legacy format and normalize
            autonomous_state = state.get("autonomous_state", {})
            notification_queue = autonomous_state.get("notification_queue", {})
            if not isinstance(notification_queue, dict):
                return {}

            # Helper to check if we should include a notification type
            def should_include_type(notif_type: str) -> bool:
                if type_filter == "all":
                    return True
                return notif_type == type_filter

            normalized: Dict[str, List[Dict]] = {}
            for user, user_data in notification_queue.items():
                # Skip users that don't match filter
                if user_filter and user != user_filter:
                    continue
                
                normalized_list: List[Dict[str, Any]] = []
                
                # Process general notifications (only if type_filter allows 'other' or 'all')
                if type_filter in ["all", "other"]:
                    raw_list = (user_data or {}).get("notifications", [])
                    for raw in raw_list:
                        # Normalize fields
                        content = raw.get("notification_content") or raw.get("content") or ""
                        raw_type = raw.get("type")
                        notif_type = self._normalize_type(raw_type, content)
                        
                        # Skip if type doesn't match filter
                        if not should_include_type(notif_type):
                            continue
                        
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

                # Process 'emails' queue items (only if type_filter is 'email' or 'all')
                if type_filter in ["all", "email"]:
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

                # Process 'news' data (only if type_filter is 'news' or 'all')
                if type_filter in ["all", "news"]:
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
            self.logger.error(f"Error loading notifications from local state: {e}")
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
            try:
                # ISO-like or common formats; best-effort
                return int(datetime.fromisoformat(str(raw_ts)).timestamp())
            except Exception:
                return 0
        except Exception:
            return 0
    
    def _mark_notifications_as_read(self, user: str, notification_ids: List[str]) -> int:
        """
        Mark specified notifications as read in both Supabase and local state.
        
        Args:
            user: User whose notifications to mark
            notification_ids: List of notification IDs to mark as read
            
        Returns:
            Number of notifications actually marked as read
        """
        if not notification_ids:
            return 0
        
        marked_count = 0
        
        # Mark as read in Supabase (primary)
        if self._supabase_available and self._supabase_client:
            try:
                supabase_marked = self._mark_as_read_in_supabase(notification_ids)
                marked_count += supabase_marked
                self.logger.debug(f"Marked {supabase_marked} notifications as read in Supabase")
            except Exception as e:
                self.logger.warning(f"Failed to mark as read in Supabase: {e}")
        
        # Also mark as read in local state (for sync)
        try:
            local_marked = self._mark_as_read_in_local_state(user, notification_ids)
            # Don't double-count if we already marked in Supabase
            if not self._supabase_available:
                marked_count += local_marked
            self.logger.debug(f"Marked {local_marked} notifications as read in local state")
        except Exception as e:
            self.logger.warning(f"Failed to mark as read in local state: {e}")
        
        self.logger.info(f"Marked {marked_count} notifications as read for {user}")
        return marked_count
    
    def _mark_as_read_in_supabase(self, notification_ids: List[str]) -> int:
        """Mark notifications as read in Supabase."""
        if not self._supabase_client or not notification_ids:
            return 0
        
        try:
            marked = 0
            for notif_id in notification_ids:
                try:
                    self._supabase_client.table("notification_sources").update(
                        {"read_status": "read"}
                    ).eq("id", notif_id).execute()
                    marked += 1
                except Exception as e:
                    self.logger.debug(f"Failed to mark {notif_id} as read: {e}")
            
            return marked
        except Exception as e:
            self.logger.error(f"Error marking as read in Supabase: {e}")
            return 0
    
    def _mark_as_read_in_local_state(self, user: str, notification_ids: List[str]) -> int:
        """Mark notifications as read in local app_state.json."""
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
                
                # Check if any email notifications were marked as read
                emails_list = user_block.get("emails", [])
                remaining_emails = []
                for email_item in emails_list:
                    email_id = email_item.get("id")
                    if email_id and email_id not in notification_ids:
                        remaining_emails.append(email_item)
                user_block["emails"] = remaining_emails
                
                # Check if any news notifications were marked as read and remove the news key
                news_data = user_block.get("news")
                if news_data and isinstance(news_data, dict):
                    # Generate the news notification ID to check if it should be removed
                    news_summary = news_data.get("summary", "")
                    news_title = "Tech News Summary"
                    generated_at = news_data.get("generated_at")
                    if generated_at:
                        try:
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

            return marked_count
            
        except Exception as e:
            self.logger.error(f"Error marking notifications as read in local state: {e}")
            return 0