"""
Supabase Notification Store

Stores email and news notification summaries in Supabase for persistent storage
and cross-device access.

Required Supabase setup - run this SQL in the Supabase SQL editor:

-- ============================================
-- NOTIFICATION SOURCES TABLE
-- ============================================
CREATE TABLE notification_sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,           -- 'email', 'news', etc.
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    read_status TEXT DEFAULT 'unread',   -- 'unread', 'read', 'dismissed'
    priority TEXT DEFAULT 'normal',      -- 'high', 'normal', 'low'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    source_generated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    batch_id TEXT
);

-- Indexes
CREATE INDEX idx_notifications_user_type ON notification_sources(user_id, source_type);
CREATE INDEX idx_notifications_user_unread ON notification_sources(user_id) WHERE read_status = 'unread';
CREATE INDEX idx_notifications_created ON notification_sources(created_at DESC);
CREATE INDEX idx_notifications_metadata ON notification_sources USING gin (metadata);
CREATE INDEX idx_notifications_batch ON notification_sources(batch_id);
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

from dotenv import load_dotenv

load_dotenv()


class NotificationStore:
    """
    Supabase-backed notification store for email and news summaries.
    
    Usage:
        store = NotificationStore()
        if store.is_available():
            store.store_email_notifications(notifications, user="Morgan")
            store.store_news_summary(summary, user="Morgan")
    """
    
    TABLE_NAME = "notification_sources"
    
    def __init__(self):
        """Initialize the Supabase client."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self._client: Optional[Client] = None
        self._initialized = False
        
        if not SUPABASE_AVAILABLE:
            print("⚠️  NotificationStore: supabase package not installed")
            print("   Install with: pip install supabase")
            return
        
        if not self.url or not self.key:
            print("⚠️  NotificationStore: SUPABASE_URL or SUPABASE_KEY not set")
            return
        
        try:
            self._client = create_client(self.url, self.key)
            self._initialized = True
            print("✅ NotificationStore initialized (Supabase)")
        except Exception as e:
            print(f"❌ NotificationStore: Failed to initialize - {e}")
    
    def is_available(self) -> bool:
        """Check if the notification store is available."""
        return self._initialized and self._client is not None
    
    def _generate_batch_id(self, source_type: str) -> str:
        """Generate a batch ID for tracking notification runs."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{source_type}_run_{timestamp}"
    
    def store_email_notifications(
        self,
        notifications: List[Dict[str, Any]],
        user: str,
        batch_id: Optional[str] = None
    ) -> bool:
        """
        Store email notification summaries to Supabase.
        
        Args:
            notifications: List of notification dicts from build_notifications_from_summaries()
            user: Target user ID (e.g., "Morgan")
            batch_id: Optional batch identifier (auto-generated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            print("⚠️  NotificationStore not available, skipping Supabase write")
            return False
        
        if not notifications:
            print("📭 No email notifications to store")
            return True
        
        batch_id = batch_id or self._generate_batch_id("email")
        
        try:
            records = []
            for n in notifications:
                # Determine priority based on topic
                topic = n.get("topic", "").lower()
                priority = "high" if topic in ["security alert", "urgent", "security"] else "normal"
                
                # Parse timestamp if it's a string
                source_generated_at = None
                timestamp_str = n.get("timestamp")
                if timestamp_str:
                    try:
                        # Try parsing the human-readable format
                        source_generated_at = datetime.strptime(
                            timestamp_str, "%B %d, %Y at %I:%M %p"
                        ).replace(tzinfo=timezone.utc).isoformat()
                    except ValueError:
                        # Fallback to ISO format
                        source_generated_at = timestamp_str
                
                records.append({
                    "id": f"email_{n.get('id', uuid.uuid4().hex[:16])}",
                    "source_type": "email",
                    "user_id": user,
                    "title": n.get("title", "Email Update"),
                    "content": n.get("content", ""),
                    "metadata": {
                        "topic": n.get("topic"),
                        "count": n.get("count"),
                        "email_ids": n.get("email_ids", []),
                        "notification_type": n.get("notification_type", "email")
                    },
                    "priority": priority,
                    "read_status": "unread",
                    "source_generated_at": source_generated_at,
                    "batch_id": batch_id
                })
            
            # Upsert to Supabase
            self._client.table(self.TABLE_NAME).upsert(records).execute()
            print(f"📧 Stored {len(records)} email notifications to Supabase (batch: {batch_id})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store email notifications: {e}")
            return False
    
    def store_news_summary(
        self,
        summary: Dict[str, Any],
        user: str,
        batch_id: Optional[str] = None
    ) -> bool:
        """
        Store a news summary to Supabase.
        
        Args:
            summary: News summary dict from NewsSummaryGenerator
            user: Target user ID (e.g., "Morgan")
            batch_id: Optional batch identifier (auto-generated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            print("⚠️  NotificationStore not available, skipping Supabase write")
            return False
        
        batch_id = batch_id or self._generate_batch_id("news")
        
        try:
            # Get the generated_at timestamp
            generated_at = summary.get("generated_at")
            if generated_at:
                try:
                    # Ensure it's in ISO format with timezone
                    if not generated_at.endswith("Z") and "+" not in generated_at:
                        generated_at = datetime.fromisoformat(generated_at).replace(
                            tzinfo=timezone.utc
                        ).isoformat()
                except (ValueError, AttributeError):
                    generated_at = datetime.now(timezone.utc).isoformat()
            else:
                generated_at = datetime.now(timezone.utc).isoformat()
            
            # Create the record
            record = {
                "id": f"news_{batch_id}",
                "source_type": "news",
                "user_id": user,
                "title": f"Tech News - {datetime.now().strftime('%b %d, %Y')}",
                "content": summary.get("summary", ""),
                "metadata": {
                    "source_articles_count": summary.get("source_articles_count"),
                    "source_file": summary.get("source_file"),
                },
                "priority": "normal",
                "read_status": "unread",
                "source_generated_at": generated_at,
                "batch_id": batch_id
            }
            
            # Upsert to Supabase
            self._client.table(self.TABLE_NAME).upsert(record).execute()
            print(f"📰 Stored news summary to Supabase (batch: {batch_id})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store news summary: {e}")
            return False
    
    def get_unread_notifications(
        self,
        user: str,
        source_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get unread notifications for a user.
        
        Args:
            user: User ID to fetch notifications for
            source_type: Optional filter by type ('email', 'news')
            limit: Maximum number of results
            
        Returns:
            List of notification records
        """
        if not self.is_available():
            return []
        
        try:
            query = (
                self._client.table(self.TABLE_NAME)
                .select("*")
                .eq("user_id", user)
                .eq("read_status", "unread")
                .order("created_at", desc=True)
                .limit(limit)
            )
            
            if source_type:
                query = query.eq("source_type", source_type)
            
            response = query.execute()
            return response.data or []
            
        except Exception as e:
            print(f"❌ Failed to fetch notifications: {e}")
            return []
    
    def mark_as_read(self, notification_ids: List[str]) -> bool:
        """
        Mark notifications as read.
        
        Args:
            notification_ids: List of notification IDs to mark as read
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available() or not notification_ids:
            return False
        
        try:
            for nid in notification_ids:
                self._client.table(self.TABLE_NAME).update(
                    {"read_status": "read"}
                ).eq("id", nid).execute()
            
            print(f"✓ Marked {len(notification_ids)} notifications as read")
            return True
            
        except Exception as e:
            print(f"❌ Failed to mark notifications as read: {e}")
            return False
    
    def cleanup_old_notifications(self, days: int = 30) -> int:
        """
        Delete notifications older than specified days.
        
        Args:
            days: Number of days to retain notifications
            
        Returns:
            Number of deleted records
        """
        if not self.is_available():
            return 0
        
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # First count how many will be deleted
            count_response = (
                self._client.table(self.TABLE_NAME)
                .select("id", count="exact")
                .lt("created_at", cutoff)
                .execute()
            )
            count = count_response.count or 0
            
            if count > 0:
                # Delete old records
                self._client.table(self.TABLE_NAME).delete().lt("created_at", cutoff).execute()
                print(f"🧹 Cleaned up {count} notifications older than {days} days")
            
            return count
            
        except Exception as e:
            print(f"❌ Failed to cleanup old notifications: {e}")
            return 0


# Convenience function for quick access
def get_notification_store() -> NotificationStore:
    """Get a NotificationStore instance."""
    return NotificationStore()

