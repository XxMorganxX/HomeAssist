"""
Local Vector Cache for fast in-memory similarity search.

Caches vectors locally and uses numpy for fast cosine similarity,
with background sync to Supabase (source of truth).
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CachedVector:
    """A single cached vector with metadata."""
    id: str
    embedding: np.ndarray  # (dims,) float32
    text: str
    metadata: Dict[str, Any]
    created_at: datetime
    
    def __post_init__(self):
        # Ensure embedding is numpy array
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)


@dataclass
class SearchResult:
    """Result from a cache search."""
    id: str
    similarity: float
    text: str
    metadata: Dict[str, Any]
    created_at: datetime


class LocalVectorCache:
    """
    In-memory vector cache with numpy-based similarity search.
    
    Features:
    - Fast local search via matrix multiplication
    - Background sync with Supabase
    - Thread-safe operations via asyncio.Lock
    - Lazy matrix rebuilding for efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local vector cache.
        
        Args:
            config: Configuration dict with:
                - embedding_dimensions: Vector dimensions (default: 3072)
                - max_cached_vectors: Maximum vectors to cache (default: 10000)
                - sync_interval_seconds: Background sync interval (default: 300)
        """
        self.dimensions = config.get("embedding_dimensions", 3072)
        self.max_vectors = config.get("max_cached_vectors", 10000)
        self.sync_interval = config.get("sync_interval_seconds", 300)
        
        # Cache storage
        self._vectors: Dict[str, CachedVector] = {}
        
        # Precomputed normalized matrix for fast search
        self._matrix: Optional[np.ndarray] = None  # (N, dims)
        self._ids: List[str] = []  # Row index -> vector id
        self._matrix_dirty: bool = True
        
        # Sync state
        self._last_sync: Optional[datetime] = None
        self._pending_writes: List[CachedVector] = []
        self._sync_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Stats
        self._search_count = 0
        self._cache_hits = 0
        
    @property
    def size(self) -> int:
        """Number of vectors in cache."""
        return len(self._vectors)
    
    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in MB."""
        if self._matrix is not None:
            matrix_bytes = self._matrix.nbytes
        else:
            matrix_bytes = 0
        # Rough estimate: matrix + metadata overhead
        return (matrix_bytes + len(self._vectors) * 1000) / (1024 * 1024)
    
    async def initialize(self, vectors: List[Dict[str, Any]]) -> int:
        """
        Initialize cache with vectors from database.
        
        Args:
            vectors: List of vector dicts with id, embedding, text, metadata, created_at
            
        Returns:
            Number of vectors loaded
        """
        async with self._lock:
            start_time = datetime.now()
            
            self._vectors.clear()
            self._matrix = None
            self._ids = []
            self._matrix_dirty = True
            
            loaded = 0
            for v in vectors:
                try:
                    # Parse embedding
                    embedding = v.get("embedding", [])
                    if isinstance(embedding, str):
                        # Handle string format "[0.1, 0.2, ...]"
                        embedding = [float(x) for x in embedding.strip("[]").split(",")]
                    
                    # Parse created_at
                    created_at = v.get("created_at")
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    elif created_at is None:
                        created_at = datetime.utcnow()
                    
                    cached = CachedVector(
                        id=v["id"],
                        embedding=np.array(embedding, dtype=np.float32),
                        text=v.get("text", ""),
                        metadata=v.get("metadata", {}),
                        created_at=created_at if isinstance(created_at, datetime) else datetime.utcnow()
                    )
                    
                    # Validate dimensions
                    if len(cached.embedding) != self.dimensions:
                        logger.warning(f"Skipping vector {v['id']}: wrong dimensions ({len(cached.embedding)} vs {self.dimensions})")
                        continue
                    
                    self._vectors[cached.id] = cached
                    loaded += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load vector {v.get('id', 'unknown')}: {e}")
            
            self._last_sync = datetime.utcnow()
            
            # Pre-build matrix if we have vectors
            if loaded > 0:
                self._rebuild_matrix()
            
            load_time = (datetime.now() - start_time).total_seconds()
            print(f"📦 Vector cache loaded: {loaded} vectors ({self.memory_mb:.1f}MB) in {load_time:.2f}s")
            
            return loaded
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the normalized embedding matrix for fast search."""
        if not self._vectors:
            self._matrix = None
            self._ids = []
            self._matrix_dirty = False
            return
        
        # Build matrix and id list
        self._ids = list(self._vectors.keys())
        embeddings = [self._vectors[vid].embedding for vid in self._ids]
        
        # Stack into matrix (N, dims)
        self._matrix = np.vstack(embeddings)
        
        # Normalize rows for cosine similarity (so we just need dot product)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self._matrix = self._matrix / norms
        
        self._matrix_dirty = False
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filter_metadata: Optional metadata filter (must contain all key-values)
            
        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        async with self._lock:
            self._search_count += 1
            
            if not self._vectors:
                return []
            
            # Rebuild matrix if needed
            if self._matrix_dirty or self._matrix is None:
                self._rebuild_matrix()
            
            if self._matrix is None or len(self._matrix) == 0:
                return []
            
            # Convert and normalize query
            query = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return []
            query = query / query_norm
            
            # Compute all similarities at once (matrix @ vector)
            similarities = self._matrix @ query  # (N,)
            
            # Apply metadata filter if provided
            if filter_metadata:
                valid_indices = []
                for i, vid in enumerate(self._ids):
                    vec = self._vectors[vid]
                    match = all(
                        vec.metadata.get(k) == v 
                        for k, v in filter_metadata.items()
                    )
                    if match:
                        valid_indices.append(i)
                
                if not valid_indices:
                    return []
                
                # Filter similarities
                valid_indices = np.array(valid_indices)
                filtered_sims = similarities[valid_indices]
                filtered_ids = [self._ids[i] for i in valid_indices]
            else:
                filtered_sims = similarities
                filtered_ids = self._ids
            
            # Apply minimum similarity
            mask = filtered_sims >= min_similarity
            filtered_sims = filtered_sims[mask]
            filtered_ids = [filtered_ids[i] for i, m in enumerate(mask) if m]
            
            if len(filtered_sims) == 0:
                return []
            
            # Get top-k
            k = min(top_k, len(filtered_sims))
            if k < len(filtered_sims):
                top_indices = np.argpartition(filtered_sims, -k)[-k:]
                top_indices = top_indices[np.argsort(filtered_sims[top_indices])[::-1]]
            else:
                top_indices = np.argsort(filtered_sims)[::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                vid = filtered_ids[idx]
                vec = self._vectors[vid]
                results.append(SearchResult(
                    id=vid,
                    similarity=float(filtered_sims[idx]),
                    text=vec.text,
                    metadata=vec.metadata,
                    created_at=vec.created_at
                ))
            
            self._cache_hits += 1
            return results
    
    async def add(self, vector: CachedVector) -> bool:
        """
        Add a vector to the cache.
        
        Args:
            vector: Vector to add
            
        Returns:
            True if added successfully
        """
        async with self._lock:
            # Validate dimensions
            if len(vector.embedding) != self.dimensions:
                logger.warning(f"Cannot add vector: wrong dimensions ({len(vector.embedding)} vs {self.dimensions})")
                return False
            
            # Check capacity
            if len(self._vectors) >= self.max_vectors:
                # LRU eviction: remove oldest
                oldest_id = min(
                    self._vectors.keys(),
                    key=lambda vid: self._vectors[vid].created_at
                )
                del self._vectors[oldest_id]
                logger.info(f"Evicted oldest vector {oldest_id} due to capacity")
            
            # Add to cache
            self._vectors[vector.id] = vector
            self._matrix_dirty = True
            
            return True
    
    async def add_batch(self, vectors: List[CachedVector]) -> int:
        """
        Add multiple vectors to the cache.
        
        Args:
            vectors: List of vectors to add
            
        Returns:
            Number of vectors added
        """
        added = 0
        for vec in vectors:
            if await self.add(vec):
                added += 1
        return added
    
    async def remove(self, vector_id: str) -> bool:
        """Remove a vector from the cache."""
        async with self._lock:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                self._matrix_dirty = True
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all vectors from the cache."""
        async with self._lock:
            self._vectors.clear()
            self._matrix = None
            self._ids = []
            self._matrix_dirty = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "memory_mb": round(self.memory_mb, 2),
            "search_count": self._search_count,
            "cache_hits": self._cache_hits,
            "hit_rate": round(self._cache_hits / max(1, self._search_count) * 100, 1),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "matrix_dirty": self._matrix_dirty,
        }
    
    async def needs_sync(self) -> bool:
        """Check if cache needs to sync with database."""
        if self._last_sync is None:
            return True
        elapsed = (datetime.utcnow() - self._last_sync).total_seconds()
        return elapsed >= self.sync_interval
