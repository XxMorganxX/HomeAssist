"""
Supabase-based conversation recorder.

Records all conversation sessions, messages, and tool calls to Supabase
for history, analytics, and debugging purposes.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING

# Lazy import to avoid breaking the module if supabase isn't installed
if TYPE_CHECKING:
    from supabase import Client

# Token counting
try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model("gpt-4o")
except ImportError:
    _ENCODER = None

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if not _ENCODER or not text:
        return 0
    try:
        return len(_ENCODER.encode(text))
    except Exception:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


class ConversationRecorder:
    """
    Supabase-based conversation recorder for HomeAssist.
    
    Records:
    - Conversation sessions (wake word → completion)
    - User and assistant messages
    - Tool calls with arguments and results
    
    Usage:
        recorder = ConversationRecorder()
        await recorder.initialize()
        
        session_id = await recorder.start_session(wake_word_model="alexa")
        await recorder.record_message("user", "Turn on the lights")
        await recorder.record_message("assistant", "Turning on the lights now.")
        await recorder.end_session()
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase connection.
        
        Args:
            supabase_url: Supabase project URL (or set SUPABASE_URL env var)
            supabase_key: Supabase service role key (or set SUPABASE_KEY env var)
        """
        self.url = supabase_url or os.environ.get("SUPABASE_URL")
        self.key = supabase_key or os.environ.get("SUPABASE_KEY")
        
        self._client: Optional["Client"] = None
        self._current_session_id: Optional[str] = None
        self._last_message_id: Optional[int] = None
        self._session_message_count: int = 0  # Track messages per session
        self._session_input_tokens: int = 0   # Track input tokens (user messages)
        self._session_output_tokens: int = 0  # Track output tokens (assistant messages)
        self._is_initialized = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if recorder has valid credentials configured."""
        return bool(self.url and self.key)
    
    @property
    def is_initialized(self) -> bool:
        """Check if recorder is initialized and ready."""
        return self._is_initialized and self._client is not None
    
    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current active session ID."""
        return self._current_session_id
    
    @property
    def session_token_stats(self) -> Dict[str, int]:
        """Get current session token statistics."""
        return {
            "input_tokens": self._session_input_tokens,
            "output_tokens": self._session_output_tokens,
            "total_tokens": self._session_input_tokens + self._session_output_tokens,
            "message_count": self._session_message_count
        }
    
    async def initialize(self) -> bool:
        """
        Initialize Supabase client and verify connection.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not self.is_enabled:
            print("⚠️  Conversation recorder disabled (SUPABASE_URL/SUPABASE_KEY not set)")
            return False
        
        try:
            # Lazy import supabase
            from supabase import create_client
            
            self._client = create_client(self.url, self.key)
            
            # Test connection with a simple query
            self._client.table("conversation_sessions").select("id").limit(1).execute()
            
            self._is_initialized = True
            print("✅ Conversation recorder initialized (Supabase)")
            return True
            
        except ImportError:
            print("⚠️  Conversation recorder disabled (supabase package not installed)")
            print("   Install with: pip install supabase")
            return False
        except Exception as e:
            print(f"❌ Conversation recorder initialization failed: {e}")
            self._client = None
            self._is_initialized = False
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────────────────
    
    async def start_session(
        self,
        wake_word_model: str = None,
        user_id: str = "default",
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Start a new conversation session.
        
        Args:
            wake_word_model: Name of the wake word model that triggered the session
            user_id: Identifier for the user (for multi-user support)
            metadata: Additional session metadata
        
        Returns:
            Session ID if successful, None otherwise.
        """
        if not self.is_initialized:
            return None
        
        try:
            data = {
                "wake_word_model": wake_word_model,
                "user_id": user_id,
                "metadata": metadata or {}
            }
            
            result = self._client.table("conversation_sessions").insert(data).execute()
            
            self._current_session_id = result.data[0]["id"]
            self._last_message_id = None
            self._session_message_count = 0   # Reset message counter
            self._session_input_tokens = 0    # Reset input token counter
            self._session_output_tokens = 0   # Reset output token counter
            
            print(f"📝 Started conversation session: {self._current_session_id[:8]}...")
            return self._current_session_id
            
        except Exception as e:
            print(f"⚠️  Failed to start session: {e}")
            return None
    
    async def end_session(self, metadata: Dict[str, Any] = None) -> bool:
        """
        End the current conversation session.
        
        If no messages were recorded, the session is deleted instead of saved.
        Stores token counts and estimated cost.
        
        Args:
            metadata: Additional metadata to store with the session
        
        Returns:
            True if session was ended successfully.
        """
        if not self.is_initialized or not self._current_session_id:
            return False
        
        try:
            # If no messages were recorded, delete the empty session
            if self._session_message_count == 0:
                self._client.table("conversation_sessions").delete().eq(
                    "id", self._current_session_id
                ).execute()
                print(f"🗑️  Deleted empty session: {self._current_session_id[:8]}... (no messages)")
                self._reset_session_state()
                return True
            
            # Session has messages - update with ended_at timestamp and token stats
            update_data = {
                "ended_at": datetime.utcnow().isoformat(),
                "total_input_tokens": self._session_input_tokens,
                "total_output_tokens": self._session_output_tokens
            }
            
            if metadata:
                # Merge with existing metadata
                existing = self._client.table("conversation_sessions").select("metadata").eq(
                    "id", self._current_session_id
                ).single().execute()
                
                merged_metadata = existing.data.get("metadata", {})
                merged_metadata.update(metadata)
                update_data["metadata"] = merged_metadata
            
            self._client.table("conversation_sessions").update(update_data).eq(
                "id", self._current_session_id
            ).execute()
            
            # Print summary with token stats
            total_tokens = self._session_input_tokens + self._session_output_tokens
            print(f"✅ Ended session: {self._current_session_id[:8]}...")
            print(f"   📊 {self._session_message_count} messages | {total_tokens:,} tokens (in: {self._session_input_tokens:,}, out: {self._session_output_tokens:,})")
            
            self._reset_session_state()
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to end session: {e}")
            return False
    
    def _reset_session_state(self) -> None:
        """Reset all session tracking state."""
        self._current_session_id = None
        self._last_message_id = None
        self._session_message_count = 0
        self._session_input_tokens = 0
        self._session_output_tokens = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Message Recording
    # ─────────────────────────────────────────────────────────────────────────
    
    async def record_message(
        self,
        role: str,
        content: str,
        is_final: bool = True,
        confidence: float = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[int]:
        """
        Record a message in the current session with token counting.
        
        Args:
            role: Message role - 'user', 'assistant', or 'system'
            content: The message text
            is_final: False for partial/streaming transcriptions
            confidence: Transcription confidence score (0-1)
            metadata: Additional message metadata (e.g., latency, model)
        
        Returns:
            Message ID if successful, None otherwise.
        """
        if not self.is_initialized or not self._current_session_id:
            return None
        
        if not content or not content.strip():
            return None
        
        try:
            # Count tokens for this message
            token_count = count_tokens(content)
            
            # Track input vs output tokens
            if role == "user":
                self._session_input_tokens += token_count
            elif role == "assistant":
                self._session_output_tokens += token_count
            
            data = {
                "session_id": self._current_session_id,
                "role": role,
                "content": content,
                "is_final": is_final,
                "confidence": confidence,
                "token_count": token_count,
                "metadata": metadata or {}
            }
            
            result = self._client.table("conversation_messages").insert(data).execute()
            self._last_message_id = result.data[0]["id"]
            self._session_message_count += 1  # Increment message counter
            
            return self._last_message_id
            
        except Exception as e:
            print(f"⚠️  Failed to record message: {e}")
            return None
    
    async def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str = None,
        duration_ms: float = None,
        message_id: int = None
    ) -> Optional[int]:
        """
        Record a tool call, linked to a message.
        
        Args:
            tool_name: Name of the tool that was executed
            arguments: Tool arguments as dictionary
            result: Tool execution result (string or JSON serializable)
            duration_ms: Execution time in milliseconds
            message_id: Message ID to link to (defaults to last message)
        
        Returns:
            Tool call ID if successful, None otherwise.
        """
        if not self.is_initialized:
            return None
        
        msg_id = message_id or self._last_message_id
        if not msg_id:
            print("⚠️  No message to attach tool call to")
            return None
        
        try:
            data = {
                "message_id": msg_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "duration_ms": duration_ms
            }
            
            result_data = self._client.table("tool_calls").insert(data).execute()
            return result_data.data[0]["id"]
            
        except Exception as e:
            print(f"⚠️  Failed to record tool call: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    async def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages from a specific session.
        
        Args:
            session_id: The session UUID
        
        Returns:
            List of message dictionaries ordered by timestamp.
        """
        if not self.is_initialized:
            return []
        
        try:
            result = self._client.table("conversation_messages").select(
                "id, role, content, timestamp, confidence, is_final"
            ).eq("session_id", session_id).order("timestamp").execute()
            
            return result.data
            
        except Exception as e:
            print(f"⚠️  Failed to get session messages: {e}")
            return []
    
    async def get_recent_sessions(
        self,
        limit: int = 20,
        user_id: str = None
    ) -> List[Dict]:
        """
        Get recent conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            user_id: Filter by user ID (optional)
        
        Returns:
            List of session dictionaries ordered by start time (newest first).
        """
        if not self.is_initialized:
            return []
        
        try:
            query = self._client.table("conversation_sessions").select(
                "id, started_at, ended_at, wake_word_model, user_id, metadata"
            ).order("started_at", desc=True).limit(limit)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            return query.execute().data
            
        except Exception as e:
            print(f"⚠️  Failed to get recent sessions: {e}")
            return []
    
    async def get_session_with_details(self, session_id: str) -> Optional[Dict]:
        """
        Get a session with all messages and tool calls.
        
        Args:
            session_id: The session UUID
        
        Returns:
            Dictionary with session info, messages, and tool calls.
        """
        if not self.is_initialized:
            return None
        
        try:
            # Get session
            session = self._client.table("conversation_sessions").select("*").eq(
                "id", session_id
            ).single().execute().data
            
            # Get messages with tool calls using join
            messages = self._client.table("conversation_messages").select(
                "*, tool_calls(*)"
            ).eq("session_id", session_id).order("timestamp").execute().data
            
            return {
                "session": session,
                "messages": messages
            }
            
        except Exception as e:
            print(f"⚠️  Failed to get session details: {e}")
            return None
    
    async def search_conversations(
        self,
        query: str,
        limit: int = 50,
        user_id: str = None
    ) -> List[Dict]:
        """
        Search through conversation content.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            user_id: Filter by user ID (optional)
        
        Returns:
            List of matching sessions with relevant messages.
        """
        if not self.is_initialized:
            return []
        
        try:
            # Simple ILIKE search (for full-text search, use the PostgreSQL function)
            result = self._client.table("conversation_messages").select(
                "session_id, role, content, timestamp"
            ).ilike("content", f"%{query}%").order("timestamp", desc=True).limit(limit).execute()
            
            # Group by session
            sessions = {}
            for msg in result.data:
                sid = msg["session_id"]
                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "matching_messages": []
                    }
                sessions[sid]["matching_messages"].append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
            
            return list(sessions.values())
            
        except Exception as e:
            print(f"⚠️  Failed to search conversations: {e}")
            return []
    
    async def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get conversation statistics for the past N days.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Statistics dictionary with session counts, message distribution, tool usage.
        """
        if not self.is_initialized:
            return {}
        
        try:
            from_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Count sessions
            sessions = self._client.table("conversation_sessions").select(
                "id", count="exact"
            ).gte("started_at", from_date).execute()
            
            # Get messages and count by role
            messages = self._client.table("conversation_messages").select(
                "role"
            ).gte("timestamp", from_date).execute()
            
            role_counts = {}
            for msg in messages.data:
                role = msg["role"]
                role_counts[role] = role_counts.get(role, 0) + 1
            
            # Get tool call counts
            tools = self._client.table("tool_calls").select(
                "tool_name"
            ).gte("executed_at", from_date).execute()
            
            tool_counts = {}
            for tc in tools.data:
                name = tc["tool_name"]
                tool_counts[name] = tool_counts.get(name, 0) + 1
            
            # Sort tools by usage
            sorted_tools = dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Get token statistics
            token_data = self._client.table("conversation_sessions").select(
                "total_input_tokens, total_output_tokens"
            ).gte("started_at", from_date).execute()
            
            total_input_tokens = sum(s.get("total_input_tokens", 0) or 0 for s in token_data.data)
            total_output_tokens = sum(s.get("total_output_tokens", 0) or 0 for s in token_data.data)
            
            return {
                "period_days": days,
                "total_sessions": sessions.count or 0,
                "total_messages": len(messages.data),
                "messages_by_role": role_counts,
                "tool_usage": sorted_tools,
                "avg_messages_per_session": (
                    len(messages.data) / sessions.count if sessions.count else 0
                ),
                "token_usage": {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "avg_tokens_per_session": (
                        (total_input_tokens + total_output_tokens) / sessions.count 
                        if sessions.count else 0
                    )
                }
            }
            
        except Exception as e:
            print(f"⚠️  Failed to get stats: {e}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────
    
    async def cleanup(self) -> None:
        """Clean up resources and end any active session."""
        if self._current_session_id:
            await self.end_session(metadata={"ended_reason": "cleanup"})
        
        self._client = None
        self._is_initialized = False
        print("✅ Conversation recorder cleaned up")

