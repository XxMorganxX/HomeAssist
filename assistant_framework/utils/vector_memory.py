"""
Vector Memory Manager.

Orchestrates embedding generation and vector storage for semantic memory.
Stores conversation summaries and retrieves relevant past context.

Features:
- Local numpy cache for fast similarity search (~1ms vs ~50ms)
- Background sync with Supabase (source of truth)
- K-fold partitioning for long conversations
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from ..interfaces.vector_store import VectorRecord, VectorSearchResult
    from ..providers.embedding import OpenAIEmbeddingProvider
    from ..providers.vector_store import SupabasePgVectorStore
    from .local_vector_cache import LocalVectorCache, CachedVector, SearchResult
except ImportError:
    from assistant_framework.interfaces.vector_store import VectorRecord, VectorSearchResult
    from assistant_framework.providers.embedding import OpenAIEmbeddingProvider
    from assistant_framework.providers.vector_store import SupabasePgVectorStore
    from assistant_framework.utils.local_vector_cache import LocalVectorCache, CachedVector, SearchResult


@dataclass
class RetrievedMemory:
    """A retrieved memory from the vector store."""
    summary: str
    similarity: float
    date: datetime
    conversation_id: str
    topics: List[str]


class VectorMemoryManager:
    """
    Manages long-term semantic memory using embeddings and vector search.
    
    Responsibilities:
    - Store conversation summaries as embeddings at conversation end
    - Retrieve semantically similar past conversations
    - Provide context enrichment for new conversations
    
    Configuration:
        enabled: bool - Enable/disable vector memory
        embedding_provider: str - "openai" (default)
        embedding_model: str - OpenAI model name
        vector_store_provider: str - "supabase" (default)
        retrieve_top_k: int - Max memories to retrieve
        relevance_threshold: float - Min similarity (0-1)
        max_age_days: int - Ignore memories older than this
        min_summary_length: int - Skip trivial summaries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector memory manager.
        
        Args:
            config: Configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        
        if not self.enabled:
            print("ℹ️  Vector memory disabled")
            return
        
        # Retrieval settings
        self.retrieve_top_k = config.get("retrieve_top_k", 3)
        # We do NOT gate results by minimum similarity by default.
        # Always return top-K ranked results; callers can still pass a threshold explicitly.
        self.relevance_threshold = float(config.get("relevance_threshold", 0.0))
        self.max_age_days = config.get("max_age_days", 90)
        self.min_summary_length = config.get("min_summary_length", 50)
        
        # User isolation
        self.console_token = config.get("console_token")
        self.user_id = config.get("user_id", "default")
        
        # Initialize providers
        self._embedding_provider = None
        self._vector_store = None
        self._local_cache = None
        self._initialized = False
        
        # Cache settings
        self._cache_enabled = config.get("local_cache_enabled", True)
        self._preload_on_startup = config.get("preload_on_startup", True)
        
        # Build provider configs
        embedding_provider_name = config.get("embedding_provider", "openai")
        vector_store_provider_name = config.get("vector_store_provider", "supabase")
        
        # Embedding config
        self._embedding_config = {
            "api_key": config.get("openai_api_key"),
            "model": config.get("embedding_model", "text-embedding-3-small"),
        }
        
        # Vector store config
        embedding_dimensions = config.get("embedding_dimensions", 3072)
        self._vector_store_config = {
            "url": config.get("supabase_url"),
            "key": config.get("supabase_key"),
            "table_name": config.get("table_name", "conversation_memories"),
            "embedding_dimensions": embedding_dimensions,
        }
        
        # Local cache config
        self._cache_config = {
            "embedding_dimensions": embedding_dimensions,
            "max_cached_vectors": config.get("max_cached_vectors", 10000),
            "sync_interval_seconds": config.get("sync_interval_seconds", 300),
        }
        
        # Create providers
        if embedding_provider_name == "openai":
            self._embedding_provider = OpenAIEmbeddingProvider(self._embedding_config)
        
        if vector_store_provider_name == "supabase":
            self._vector_store = SupabasePgVectorStore(self._vector_store_config)
        
        # Create local cache
        if self._cache_enabled:
            self._local_cache = LocalVectorCache(self._cache_config)
    
    async def initialize(self) -> bool:
        """Initialize embedding and vector store providers, and load cache."""
        if not self.enabled:
            return True
        
        if self._initialized:
            return True
        
        try:
            # Initialize embedding provider
            if self._embedding_provider:
                if not await self._embedding_provider.initialize():
                    print("⚠️  Failed to initialize embedding provider")
                    return False
            
            # Initialize vector store
            if self._vector_store:
                if not await self._vector_store.initialize():
                    print("⚠️  Failed to initialize vector store")
                    return False
            
            # Load vectors into local cache
            if self._local_cache and self._preload_on_startup:
                await self._load_cache_from_database()
            
            self._initialized = True
            print("✅ Vector memory manager initialized")
            return True
            
        except Exception as e:
            print(f"❌ Vector memory initialization error: {e}")
            return False
    
    async def _load_cache_from_database(self) -> int:
        """
        Load all vectors from Supabase into local cache.
        
        Returns:
            Number of vectors loaded
        """
        if not self._vector_store or not self._local_cache:
            return 0
        
        try:
            print("📥 Loading vectors from database into local cache...")
            
            # Build metadata filter for user isolation
            filter_metadata = {"user_id": self.user_id}
            if self.console_token:
                filter_metadata["console_token"] = self.console_token
            
            # Fetch all vectors from Supabase
            # Note: This queries the table directly, not via similarity search
            table_name = self._vector_store_config.get("table_name", "conversation_memories")
            
            query = self._vector_store._client.table(table_name).select("*")
            
            # Apply metadata filters
            for key, value in filter_metadata.items():
                query = query.contains("metadata", {key: value})
            
            # Limit to max cache size
            query = query.limit(self._local_cache.max_vectors)
            
            response = query.execute()
            
            if response.data:
                # Initialize cache with the vectors
                loaded = await self._local_cache.initialize(response.data)
                return loaded
            else:
                print("📦 No existing vectors found in database")
                return 0
                
        except Exception as e:
            print(f"⚠️  Failed to load cache from database: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    async def sync_cache(self) -> int:
        """
        Sync local cache with database (pull latest).
        
        Returns:
            Number of vectors synced
        """
        return await self._load_cache_from_database()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get local cache statistics."""
        if self._local_cache:
            return self._local_cache.get_stats()
        return {"enabled": False}
    
    def _parse_turns(self, conversation: str) -> List[Dict[str, str]]:
        """
        Parse conversation text into individual turns (user/assistant pairs).
        
        Args:
            conversation: Raw conversation text
            
        Returns:
            List of dicts with 'user' and 'assistant' keys
        """
        turns = []
        lines = conversation.strip().split('\n')
        
        current_turn = {}
        current_role = None
        current_text = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('user:'):
                # Save previous turn if complete
                if current_role == 'assistant' and current_text:
                    current_turn['assistant'] = ' '.join(current_text).strip()
                    if 'user' in current_turn and 'assistant' in current_turn:
                        turns.append(current_turn)
                    current_turn = {}
                
                current_role = 'user'
                # Extract text after "user:"
                text = line[line.lower().find('user:') + 5:].strip()
                current_text = [text] if text else []
                
            elif line_lower.startswith('assistant:'):
                # Save user text
                if current_role == 'user' and current_text:
                    current_turn['user'] = ' '.join(current_text).strip()
                
                current_role = 'assistant'
                text = line[line.lower().find('assistant:') + 10:].strip()
                current_text = [text] if text else []
                
            else:
                # Continuation of current role's text
                if current_role and line.strip():
                    current_text.append(line.strip())
        
        # Don't forget the last turn
        if current_role == 'assistant' and current_text:
            current_turn['assistant'] = ' '.join(current_text).strip()
        if 'user' in current_turn and 'assistant' in current_turn:
            turns.append(current_turn)
        
        return turns
    
    def _compute_k(self, num_turns: int) -> int:
        """
        Compute number of overlapping partitions based on turn count.
        
        k = ceil(turns^(1/1.5)) = ceil(turns^0.667)
        
        Examples:
            2 turns → k=2
            4 turns → k=3
            8 turns → k=4
            16 turns → k=6
            27 turns → k=9
        """
        if num_turns <= 1:
            return 1
        
        # k = turns^(2/3) ≈ turns^0.667
        k = int(round(num_turns ** (1 / 1.5)))
        return max(1, min(k, num_turns))  # Clamp between 1 and num_turns
    
    def _create_overlapping_partitions(
        self, 
        turns: List[Dict[str, str]], 
        k: int
    ) -> List[tuple]:
        """
        Create k overlapping sliding window partitions.
        
        Windows are distributed evenly with overlap. Each window covers
        roughly n/k turns, but positioned so windows overlap smoothly.
        
        Args:
            turns: List of turn dicts
            k: Number of partitions to create
            
        Returns:
            List of (start_idx, end_idx, segment_text) tuples
        """
        n = len(turns)
        
        if k <= 1 or n <= 1:
            return [(0, n, self._turns_to_text(turns))]
        
        # Window size: each partition covers roughly half the conversation
        # This ensures good overlap between adjacent windows
        window_size = max(1, int(n * 0.6))  # 60% of conversation per window
        
        # Ensure window isn't larger than total
        window_size = min(window_size, n)
        
        # Calculate stride to evenly distribute k windows
        if k > 1:
            stride = (n - window_size) / (k - 1)
        else:
            stride = 0
        
        partitions = []
        for i in range(k):
            start = int(round(i * stride))
            end = min(start + window_size, n)
            
            # Ensure we don't create duplicate windows
            segment_turns = turns[start:end]
            if segment_turns:
                segment_text = self._turns_to_text(segment_turns)
                partitions.append((start, end, segment_text))
        
        return partitions
    
    def _turns_to_text(self, turns: List[Dict[str, str]]) -> str:
        """Convert turn dicts back to text format."""
        lines = []
        for turn in turns:
            if 'user' in turn:
                lines.append(f"user: {turn['user']}")
            if 'assistant' in turn:
                lines.append(f"assistant: {turn['assistant']}")
        return '\n'.join(lines)

    async def store_conversation(
        self,
        summary: str,
        conversation_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a conversation in the vector database with k overlapping partitions.
        
        Stores:
        - 1 full conversation vector (always)
        - k overlapping partition vectors (k = turns^(1/1.5))
        
        The k partitions are sliding windows with ~60% overlap, ensuring
        different perspectives of the conversation are captured.
        
        Args:
            summary: The conversation text (user:/assistant: format)
            conversation_id: Optional base ID (generates UUID if not provided)
            topics: Optional list of topics discussed
            metadata: Optional additional metadata
            
        Returns:
            bool: True if at least one record stored successfully
        """
        if not self.enabled or not self._initialized:
            return False
        
        # Skip trivial summaries
        if not summary or len(summary.strip()) < self.min_summary_length:
            print("ℹ️  Skipping trivial conversation (too short for vector memory)")
            return False
        
        try:
            # Parse conversation into turns
            turns = self._parse_turns(summary)
            num_turns = len(turns)
            
            if num_turns == 0:
                # Fallback: store as single vector if parsing fails
                print("ℹ️  Could not parse turns, storing as single vector")
                turns = [{"user": summary, "assistant": ""}]
                num_turns = 1
            
            # Compute k (number of partitions) based on turn count
            k = self._compute_k(num_turns)
            
            # Base ID for this conversation
            base_id = conversation_id or str(uuid.uuid4())
            stored_at = datetime.utcnow().isoformat()
            
            # Collect all segments to embed
            all_segments = []
            segment_metadata = []
            
            # 1. Full conversation (always)
            full_text = self._turns_to_text(turns)
            all_segments.append(full_text)
            segment_metadata.append({
                "partition_type": "full",
                "partition_index": 0,
                "partition_of": 1,
                "turn_range": f"0-{num_turns}",
            })
            
            # 2. k overlapping partitions (only if > 4 turns)
            if num_turns > 4 and k > 1:
                partitions = self._create_overlapping_partitions(turns, k)
                for i, (start, end, segment_text) in enumerate(partitions):
                    # Skip if identical to full conversation
                    if start == 0 and end == num_turns:
                        continue
                    
                    all_segments.append(segment_text)
                    segment_metadata.append({
                        "partition_type": "window",
                        "partition_index": i,
                        "partition_of": k,
                        "turn_range": f"{start}-{end}",
                    })
            
            if not all_segments:
                print("ℹ️  No valid segments to store")
                return False
            
            # Batch embed all segments
            embeddings = await self._embedding_provider.embed_batch(all_segments)
            
            # Build records
            records = []
            for i, (segment_text, embedding, seg_meta) in enumerate(
                zip(all_segments, embeddings, segment_metadata)
            ):
                part_type = seg_meta['partition_type']
                part_idx = seg_meta['partition_index']
                record_id = f"{base_id}_{part_type}_{part_idx}"
                
                record_metadata = {
                    "user_id": self.user_id,
                    "topics": topics or [],
                    "summary_length": len(segment_text),
                    "stored_at": stored_at,
                    "conversation_id": base_id,
                    "partition_type": seg_meta["partition_type"],
                    "partition_index": seg_meta["partition_index"],
                    "partition_of": seg_meta["partition_of"],
                    "turn_range": seg_meta["turn_range"],
                    "total_turns": num_turns,
                    "k_partitions": k,
                }
                
                if self.console_token:
                    record_metadata["console_token"] = self.console_token
                
                if metadata:
                    record_metadata.update(metadata)
                
                records.append(VectorRecord(
                    id=record_id,
                    embedding=embedding,
                    text=segment_text,
                    metadata=record_metadata
                ))
            
            # Store all records in Supabase
            stored_count = await self._vector_store.store_batch(records)
            
            # Also add to local cache for immediate availability
            if self._local_cache and stored_count > 0:
                cached_vectors = []
                for record in records:
                    cached_vectors.append(CachedVector(
                        id=record.id,
                        embedding=record.embedding,
                        text=record.text,
                        metadata=record.metadata,
                        created_at=datetime.utcnow()
                    ))
                await self._local_cache.add_batch(cached_vectors)
            
            if stored_count > 0:
                num_windows = len(records) - 1  # Subtract the full conversation
                cache_info = f" (cache: {self._local_cache.size})" if self._local_cache else ""
                print(f"🧠 Stored {len(records)} vectors (1 full + {num_windows} overlapping windows, k={k}) for {num_turns} turns{cache_info}")
            
            return stored_count > 0
            
        except Exception as e:
            print(f"⚠️  Failed to store conversation in vector memory: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant past conversations based on semantic similarity.
        
        Args:
            query: The query text (e.g., user's message)
            top_k: Max results (defaults to config value)
            min_similarity: Min similarity threshold (defaults to config value)
            
        Returns:
            List of retrieved memories, ordered by relevance
        """
        if not self.enabled or not self._initialized:
            return []
        
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.retrieve_top_k
        # If caller doesn't provide a threshold, don't gate results at all.
        # Always return top-K ranked results.
        if min_similarity is None:
            min_similarity = 0.0
        
        try:
            # Log the exact query being used
            query_preview = query[:100] + "..." if len(query) > 100 else query
            print(f"🔎 Vector query: \"{query_preview}\"")
            
            # Generate query embedding
            query_embedding = await self._embedding_provider.embed(query)
            
            # Build metadata filter
            filter_metadata = {"user_id": self.user_id}
            if self.console_token:
                filter_metadata["console_token"] = self.console_token
            
            # Use local cache if available and populated
            if self._local_cache and self._local_cache.size > 0:
                # Fast local search
                import time
                start = time.perf_counter()
                
                cache_results = await self._local_cache.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Get more, filter by age
                    min_similarity=min_similarity,
                    filter_metadata=filter_metadata
                )
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"⚡ Cache search: {len(cache_results)} results in {elapsed_ms:.1f}ms")
                
                # Convert cache results to standard format
                results = []
                for cr in cache_results:
                    results.append(VectorSearchResult(
                        id=cr.id,
                        similarity=cr.similarity,
                        text=cr.text,
                        metadata=cr.metadata,
                        created_at=cr.created_at
                    ))
            else:
                # Fall back to Supabase
                print("📡 Using Supabase (cache empty)")
                results = await self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Get more, filter by age
                    min_similarity=min_similarity,
                    filter_metadata=filter_metadata
                )
            
            # Filter by age
            cutoff_date = datetime.utcnow() - timedelta(days=self.max_age_days)
            filtered_results = []
            
            for result in results:
                if result.created_at and result.created_at.replace(tzinfo=None) < cutoff_date:
                    continue
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
            
            # Convert to RetrievedMemory objects
            memories = []
            for result in filtered_results:
                memories.append(RetrievedMemory(
                    summary=result.text,
                    similarity=result.similarity,
                    date=result.created_at or datetime.utcnow(),
                    conversation_id=result.id,
                    topics=result.metadata.get("topics", [])
                ))
            
            if memories:
                print(f"🔍 Retrieved {len(memories)} relevant past conversation(s)")
            
            return memories
            
        except Exception as e:
            print(f"⚠️  Failed to retrieve from vector memory: {e}")
            return []
    
    def format_for_context(self, memories: List[RetrievedMemory]) -> str:
        """
        Format retrieved memories for injection into system prompt.
        
        Args:
            memories: List of retrieved memories
            
        Returns:
            Formatted string for context injection
        """
        if not memories:
            return ""
        
        lines = ["RELEVANT PAST CONVERSATIONS:"]
        
        for mem in memories:
            date_str = mem.date.strftime("%b %d") if mem.date else "Unknown"
            # Send full content to AI (no truncation)
            summary = mem.summary
            
            lines.append(f"- [{date_str}]: {summary}")
        
        return "\n".join(lines)
    
    async def get_context_enrichment(self, user_message: str) -> str:
        """
        Convenience method: retrieve relevant memories and format for context.
        
        Args:
            user_message: The user's current message
            
        Returns:
            Formatted context string (empty if no relevant memories)
        """
        memories = await self.retrieve_relevant(user_message)

        # Smart filtering: only take top 2 if they have similar scores
        # Otherwise just take the most relevant result
        if len(memories) >= 2:
            top_score = memories[0].similarity
            second_score = memories[1].similarity
            similarity_gap = top_score - second_score
            
            # If scores are within 0.06 (6%), keep both; otherwise just top 1
            if similarity_gap > 0.06:
                print(f"🎯 Score gap {similarity_gap:.2f} > 0.06 → keeping only top result")
                memories = [memories[0]]
            else:
                print(f"🎯 Score gap {similarity_gap:.2f} ≤ 0.06 → keeping top 2 results")
                memories = memories[:2]
        elif len(memories) == 1:
            print("🎯 Single result found")

        # Format the context
        formatted_context = self.format_for_context(memories)
        
        # Print the EXACT content being sent to AI
        if formatted_context:
            print("📤 Vector context sent to AI:")
            print("─" * 60)
            print(formatted_context)
            print("─" * 60)
        else:
            print("🔎 Vector recall: none found")

        return formatted_context
    
    async def get_memory_count(self) -> int:
        """Get the total number of stored memories for this user."""
        if not self.enabled or not self._initialized:
            return 0
        
        try:
            filter_metadata = {"user_id": self.user_id}
            if self.console_token:
                filter_metadata["console_token"] = self.console_token
            
            return await self._vector_store.count(filter_metadata)
        except Exception:
            return 0
    
    async def delete_memory(self, conversation_id: str) -> bool:
        """Delete a specific memory by conversation ID."""
        if not self.enabled or not self._initialized:
            return False
        
        try:
            return await self._vector_store.delete(conversation_id)
        except Exception as e:
            print(f"⚠️  Failed to delete memory: {e}")
            return False
    
    async def clear_all_memories(self) -> int:
        """
        Clear all memories for this user.
        Use with caution!
        
        Returns:
            Number of memories deleted
        """
        if not self.enabled or not self._initialized:
            return 0
        
        try:
            filter_metadata = {"user_id": self.user_id}
            if self.console_token:
                filter_metadata["console_token"] = self.console_token
            
            count = await self._vector_store.delete_by_metadata(filter_metadata)
            print(f"🗑️  Cleared {count} memories from vector store")
            return count
        except Exception as e:
            print(f"⚠️  Failed to clear memories: {e}")
            return 0
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._embedding_provider:
            await self._embedding_provider.cleanup()
        
        if self._vector_store:
            await self._vector_store.cleanup()
        
        self._initialized = False
    
    def get_setup_sql(self) -> str:
        """
        Get the SQL needed to set up the vector store table.
        Run this in Supabase SQL editor.
        """
        if self._vector_store and hasattr(self._vector_store, 'create_table_sql'):
            return self._vector_store.create_table_sql
        return "-- Vector store not configured"
