"""
Unified context management provider.
"""

import json
from typing import List, Dict, Optional, Any
import tiktoken

try:
    # Try relative imports first (when used as package)
    from ...interfaces.context import ContextInterface
    from ...models.data_models import ConversationMessage, MessageRole
    from ...utils.memory.conversation_summarizer import ConversationSummarizer
    from ...utils.memory.persistent_memory import PersistentMemoryManager
    from ...utils.memory.vector_memory import VectorMemoryManager
except ImportError:
    # Fall back to absolute imports (when run as module)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from interfaces.context import ContextInterface
    from models.data_models import ConversationMessage, MessageRole
    from utils.memory.conversation_summarizer import ConversationSummarizer
    from utils.memory.persistent_memory import PersistentMemoryManager
    from utils.memory.vector_memory import VectorMemoryManager


class UnifiedContextProvider(ContextInterface):
    """Unified context manager that handles conversation history and token counting."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize unified context provider.
        
        Args:
            config: Configuration dictionary containing:
                - system_prompt: System prompt to use
                - model: Model name for tokenizer (default: "gpt-4")
                - max_messages: Maximum messages to keep (default: 21)
                - enable_debug: Enable debug logging (default: False)
        """
        self.system_prompt = config.get('system_prompt', '')
        self.model = config.get('model', 'gpt-4')
        self.max_messages = config.get('max_messages', 21)
        # For responder: only send this many most-recent non-system messages
        self.response_recent_messages = int(config.get('response_recent_messages', 8))
        self.enable_debug = config.get('enable_debug', False)
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Initialize summarizer
        summarization_config = config.get('summarization', {})
        self._summarizer = ConversationSummarizer(summarization_config) if summarization_config.get('enabled', False) else None
        
        # Initialize persistent memory manager
        persistent_memory_config = config.get('persistent_memory', {})
        self._persistent_memory = PersistentMemoryManager(persistent_memory_config) if persistent_memory_config.get('enabled', False) else None
        
        # Initialize vector memory manager
        vector_memory_config = config.get('vector_memory', {})
        self._vector_memory = VectorMemoryManager(vector_memory_config) if vector_memory_config.get('enabled', False) else None
        self._vector_memory_initialized = False
        
        # Conversation history
        self.conversation_history: List[ConversationMessage] = []
        
        # Initialize with system prompt if provided
        if self.system_prompt:
            self.initialize(self.system_prompt)
    
    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer for the model."""
        try:
            if "gpt-4" in self.model:
                self.encoder = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model:
                self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to gpt-4 tokenizer
                self.encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # Fallback to general encoding
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def initialize(self, system_prompt: Optional[str] = None) -> None:
        """Initialize the context manager with optional system prompt."""
        if system_prompt:
            self.system_prompt = system_prompt
        
        # Clear existing history
        self.conversation_history = []
        
        # Reset summarizer state for new session
        if self._summarizer:
            self._summarizer.reset()
        
        # Add system prompt if provided
        if self.system_prompt:
            self.add_message(
                MessageRole.SYSTEM.value,
                self.system_prompt,
                metadata={'initialized': True}
            )
        
        if self.enable_debug:
            print(f"[DEBUG] Context initialized with system prompt: {bool(self.system_prompt)}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history."""
        try:
            message_role = MessageRole(role)
        except ValueError:
            # Default to user if role is not recognized
            message_role = MessageRole.USER
        
        message = ConversationMessage(
            role=message_role,
            content=content,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        
        if self.enable_debug:
            print(f"[DEBUG] Added {role} message: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # Check if summarization should trigger
        self._check_summarization()
    
    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Add multiple messages to the conversation history."""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            metadata = msg.get('metadata', {})
            
            self.add_message(role, content, metadata)
    
    def _check_summarization(self) -> None:
        """Check if conversation should be summarized and trigger if needed."""
        if not self._summarizer:
            return
        
        # Count non-system messages
        message_count = sum(
            1 for msg in self.conversation_history 
            if msg.role != MessageRole.SYSTEM
        )
        
        if self._summarizer.should_summarize(message_count):
            # Get messages for summarization (exclude system prompt)
            messages_to_summarize = [
                {"role": msg.role.value, "content": msg.content}
                for msg in self.conversation_history
                if msg.role != MessageRole.SYSTEM
            ]
            self._summarizer.summarize_async(messages_to_summarize, message_count)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return [msg.to_dict() for msg in self.conversation_history]
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Alias: get the full conversation history for final text generation."""
        return self.get_history()

    def get_recent_for_response(self) -> List[Dict[str, Any]]:
        """Return a small recent window to bias responses to latest prompt.
        Keeps the initial system message (if any) and the N most recent messages after it.
        """
        if not self.conversation_history:
            return []

        # Preserve leading system prompt
        start_idx = 1 if (self.conversation_history and self.conversation_history[0].role == MessageRole.SYSTEM) else 0
        head = [self.conversation_history[0].to_dict()] if start_idx == 1 else []

        tail = [msg.to_dict() for msg in self.conversation_history[start_idx:]][-self.response_recent_messages:]
        return head + tail
    
    def get_recent_history(self, n: int) -> List[Dict[str, Any]]:
        """Get the most recent n messages."""
        recent_messages = self.conversation_history[-n:] if n > 0 else []
        return [msg.to_dict() for msg in recent_messages]
    
    def get_tool_context(self, max_user: int = 3, max_assistant: int = 2) -> List[Dict[str, Any]]:
        """
        Build a compact recent context for tool decisioning: last max_user user prompts
        and last max_assistant assistant responses, preserving chronological order.
        Includes the system message if it exists as the very first message.
        """
        if not self.conversation_history:
            return []

        # Optional leading system message
        system_entry = None
        if self.conversation_history and self.conversation_history[0].role == MessageRole.SYSTEM:
            system_entry = (0, self.conversation_history[0])

        # Collect last N users and assistants from the end
        user_msgs: List[tuple[int, ConversationMessage]] = []
        assistant_msgs: List[tuple[int, ConversationMessage]] = []
        for idx in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[idx]
            if msg.role == MessageRole.USER and len(user_msgs) < max_user:
                user_msgs.append((idx, msg))
            elif msg.role == MessageRole.ASSISTANT and len(assistant_msgs) < max_assistant:
                assistant_msgs.append((idx, msg))
            # Stop early if we have all needed
            if len(user_msgs) >= max_user and len(assistant_msgs) >= max_assistant:
                break

        # Reverse to chronological order and merge
        selected = []
        if system_entry:
            selected.append(system_entry)
        selected.extend(sorted(user_msgs, key=lambda x: x[0]))
        selected.extend(sorted(assistant_msgs, key=lambda x: x[0]))

        # Sort again globally by original index to strictly preserve chronology
        selected_sorted = sorted(selected, key=lambda x: x[0])
        return [msg.to_dict() for (_, msg) in selected_sorted]
    
    def trim_history(self, max_messages: int) -> None:
        """Trim conversation history to a maximum number of messages."""
        if len(self.conversation_history) <= max_messages:
            return
        
        # Keep system message if it exists
        system_message = None
        if (self.conversation_history and 
            self.conversation_history[0].role == MessageRole.SYSTEM):
            system_message = self.conversation_history[0]
            remaining_count = max_messages - 1
        else:
            remaining_count = max_messages
        
        # Keep the most recent messages
        recent_messages = self.conversation_history[-remaining_count:]
        
        # Rebuild history
        if system_message:
            self.conversation_history = [system_message] + recent_messages
        else:
            self.conversation_history = recent_messages
        
        if self.enable_debug:
            print(f"[DEBUG] Trimmed conversation history to {len(self.conversation_history)} messages")
    
    def count_tokens(self) -> int:
        """Count the total number of tokens in the conversation history."""
        # Format messages as they would be sent to the API
        formatted_messages = ""
        for message in self.conversation_history:
            role = message.role.value
            content = message.content
            formatted_messages += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Use tiktoken to count exact tokens
        return len(self.encoder.encode(formatted_messages))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context state."""
        token_count = self.count_tokens()
        
        # Calculate role distribution
        role_counts = {}
        for msg in self.conversation_history:
            role = msg.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Get latest message info
        latest_message = self.conversation_history[-1] if self.conversation_history else None
        
        return {
            'message_count': len(self.conversation_history),
            'token_count': token_count,
            'role_distribution': role_counts,
            'has_system_prompt': bool(
                self.conversation_history and 
                self.conversation_history[0].role == MessageRole.SYSTEM
            ),
            'latest_message': {
                'role': latest_message.role.value if latest_message else None,
                'timestamp': latest_message.timestamp if latest_message else None,
                'content_length': len(latest_message.content) if latest_message else 0
            } if latest_message else None,
            'model': self.model,
            'max_messages': self.max_messages
        }
    
    def reset(self) -> None:
        """Reset the conversation history to initial state, including persistent memory."""
        # Build enhanced system prompt with persistent memory
        enhanced_prompt = self.system_prompt
        
        if self._persistent_memory:
            # Reload memory from file to ensure it's fresh
            self._persistent_memory._current_memory = self._persistent_memory.load_memory()
            memory_summary = self._persistent_memory.get_memory_summary()
            if memory_summary:
                enhanced_prompt = f"{self.system_prompt}\n\nPERSISTENT MEMORY (what you know about this user):\n{memory_summary}"
                print(f"🧠 Loaded persistent memory into system prompt")
                print(f"📝 Memory contents:\n{memory_summary}")
            else:
                print("⚠️  No persistent memory to load (summary empty)")
        else:
            print("⚠️  Persistent memory manager not initialized")
        
        self.initialize(enhanced_prompt)
        
        if self.enable_debug:
            print("[DEBUG] Context reset to initial state")
    
    def export_history(self) -> str:
        """Export conversation history as JSON string."""
        return json.dumps(self.get_history(), indent=2)
    
    def import_history(self, history_json: str) -> None:
        """Import conversation history from JSON string."""
        try:
            history_data = json.loads(history_json)
            self.conversation_history = []
            
            for msg_data in history_data:
                message = ConversationMessage.from_dict(msg_data)
                self.conversation_history.append(message)
            
            if self.enable_debug:
                print(f"[DEBUG] Imported {len(self.conversation_history)} messages")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise Exception(f"Failed to import history: {e}")
    
    def get_conversation_str(self) -> str:
        """Get conversation history as a formatted string."""
        lines = []
        for msg in self.conversation_history:
            lines.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(lines)
    
    def auto_trim_if_needed(self) -> bool:
        """Automatically trim history if it exceeds max_messages."""
        if len(self.conversation_history) > self.max_messages:
            self.trim_history(self.max_messages)
            return True
        return False
    
    @property
    def capabilities(self) -> dict:
        """Get provider capabilities."""
        return {
            'token_counting': True,
            'auto_trimming': True,
            'metadata_support': True,
            'persistence': False,
            'export_import': True,
            'role_validation': True,
            'features': ['tiktoken_counting', 'message_trimming', 'role_distribution']
        }
    
    def on_conversation_end(self) -> None:
        """
        Called when a conversation session ends.
        Triggers persistent memory update and vector memory storage.
        """
        # Get the current conversation summary
        conversation_summary = None
        if self._summarizer and self._summarizer.current_summary:
            conversation_summary = self._summarizer.current_summary
        else:
            # Fallback: create a quick summary from recent messages
            recent_messages = [
                msg for msg in self.conversation_history 
                if msg.role != MessageRole.SYSTEM
            ][-10:]  # Last 10 messages
            if recent_messages:
                conversation_summary = "\n".join([
                    f"{msg.role.value}: {msg.content[:200]}" 
                    for msg in recent_messages
                ])
        
        if conversation_summary:
            # Update persistent memory (extracts facts)
            if self._persistent_memory:
                self._persistent_memory.update_after_conversation(conversation_summary)
            
            # Store in vector memory (semantic search)
            if self._vector_memory and self._vector_memory_initialized:
                import asyncio
                try:
                    # Run async storage in background
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._store_in_vector_memory(conversation_summary))
                    else:
                        loop.run_until_complete(self._store_in_vector_memory(conversation_summary))
                except Exception as e:
                    print(f"⚠️  Failed to store in vector memory: {e}")
    
    async def _store_in_vector_memory(self, summary: str) -> None:
        """Store conversation summary in vector memory."""
        if self._vector_memory:
            await self._vector_memory.store_conversation(summary)
    
    async def initialize_vector_memory(self) -> bool:
        """Initialize the vector memory manager (call once at startup)."""
        if not self._vector_memory:
            return True
        
        if self._vector_memory_initialized:
            return True
        
        success = await self._vector_memory.initialize()
        self._vector_memory_initialized = success
        return success
    
    async def get_vector_memory_context(self, user_message: str) -> str:
        """
        Retrieve relevant past conversations for context enrichment.
        
        Args:
            user_message: The user's current message
            
        Returns:
            Formatted context string with relevant past conversations
        """
        if not self._vector_memory or not self._vector_memory_initialized:
            return ""
        
        return await self._vector_memory.get_context_enrichment(user_message)
    
    @property
    def persistent_memory(self) -> Optional[PersistentMemoryManager]:
        """Get the persistent memory manager."""
        return self._persistent_memory
    
    @property
    def vector_memory(self) -> Optional[VectorMemoryManager]:
        """Get the vector memory manager."""
        return self._vector_memory
    
    async def preload_vector_cache(self) -> bool:
        """
        Preload vector memory cache for faster first query.
        
        Call this speculatively during idle/wake word detection.
        
        Returns:
            True if cache is ready, False otherwise
        """
        if not self._vector_memory or not self._vector_memory_initialized:
            return False
        
        return await self._vector_memory.preload_recent_vectors()
    
    def is_vector_cache_warm(self) -> bool:
        """Check if vector memory cache is populated."""
        if not self._vector_memory:
            return False
        return self._vector_memory.is_cache_warm()