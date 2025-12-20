"""
Supabase pgvector store provider.
Uses Supabase with the pgvector extension for vector similarity search.

Required Supabase setup:
1. Enable pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
2. Create table (see create_table_sql property)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

try:
    from ...interfaces.vector_store import (
        VectorStoreInterface,
        VectorSearchResult,
        VectorRecord
    )
except ImportError:
    from assistant_framework.interfaces.vector_store import (
        VectorStoreInterface,
        VectorSearchResult,
        VectorRecord
    )


class SupabasePgVectorStore(VectorStoreInterface):
    """
    Supabase pgvector implementation for vector storage and search.
    
    Configuration:
        - url: Supabase project URL
        - key: Supabase service role key
        - table_name: Name of the vector table (default: conversation_memories)
        - embedding_dimensions: Dimension of embeddings (default: 1536)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Supabase vector store.
        
        Args:
            config: Configuration dict
        """
        if create_client is None:
            raise ImportError("supabase library required. Install with: pip install supabase")
        
        self.url = config.get("url") or os.getenv("SUPABASE_URL")
        self.key = config.get("key") or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key are required")
        
        self.table_name = config.get("table_name", "conversation_memories")
        self.embedding_dimensions = config.get("embedding_dimensions", 1536)
        
        self._client: Optional[Client] = None
        self._initialized = False
    
    @property
    def create_table_sql(self) -> str:
        """SQL to create the vector table. Run this in Supabase SQL editor."""
        return f"""
-- Enable pgvector extension (run once)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the memories table
CREATE TABLE IF NOT EXISTS {self.table_name} (
    id TEXT PRIMARY KEY,
    embedding vector({self.embedding_dimensions}),
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
ON {self.table_name} 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for metadata filtering
CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
ON {self.table_name} 
USING gin (metadata);

-- Create index for timestamp queries
CREATE INDEX IF NOT EXISTS {self.table_name}_created_at_idx 
ON {self.table_name} (created_at DESC);

-- Function for similarity search
CREATE OR REPLACE FUNCTION match_{self.table_name}(
    query_embedding vector({self.embedding_dimensions}),
    match_count int DEFAULT 5,
    min_similarity float DEFAULT 0.0,
    filter_metadata jsonb DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cm.id,
        cm.text,
        cm.metadata,
        1 - (cm.embedding <=> query_embedding) as similarity,
        cm.created_at
    FROM {self.table_name} cm
    WHERE 
        (filter_metadata IS NULL OR cm.metadata @> filter_metadata)
        AND (1 - (cm.embedding <=> query_embedding)) >= min_similarity
    ORDER BY cm.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""
    
    async def initialize(self) -> bool:
        """Initialize the Supabase client."""
        try:
            self._client = create_client(self.url, self.key)
            self._initialized = True
            print(f"✅ Supabase vector store initialized (table: {self.table_name})")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Supabase vector store: {e}")
            return False
    
    def _ensure_initialized(self):
        """Ensure client is initialized (sync check for sync operations)."""
        if not self._initialized or not self._client:
            self._client = create_client(self.url, self.key)
            self._initialized = True
    
    async def store(self, record: VectorRecord) -> bool:
        """Store a single vector record."""
        self._ensure_initialized()
        
        try:
            data = {
                "id": record.id,
                "embedding": record.embedding,
                "text": record.text,
                "metadata": record.metadata,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upsert (insert or update)
            self._client.table(self.table_name).upsert(data).execute()
            return True
            
        except Exception as e:
            print(f"⚠️  Vector store error: {e}")
            return False
    
    async def store_batch(self, records: List[VectorRecord]) -> int:
        """Store multiple vector records."""
        self._ensure_initialized()
        
        if not records:
            return 0
        
        try:
            data = [
                {
                    "id": r.id,
                    "embedding": r.embedding,
                    "text": r.text,
                    "metadata": r.metadata,
                    "updated_at": datetime.utcnow().isoformat()
                }
                for r in records
            ]
            
            self._client.table(self.table_name).upsert(data).execute()
            return len(records)
            
        except Exception as e:
            print(f"⚠️  Batch vector store error: {e}")
            return 0
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        self._ensure_initialized()
        
        # Use direct in-memory similarity calculation
        # This works without any RPC setup and is fine for < 10k records
        return await self._direct_search(
            query_embedding, top_k, min_similarity, filter_metadata
        )
    
    async def _direct_search(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """Direct search - fetches records and calculates similarity in Python."""
        try:
            # Fetch records from table
            query = self._client.table(self.table_name).select("id, text, metadata, embedding, created_at")
            
            # Apply metadata filter if provided
            if filter_metadata:
                query = query.contains("metadata", filter_metadata)
            
            # Get records (limit to reasonable size)
            response = query.limit(500).execute()
            
            if not response.data:
                print("   📭 No records in table")
                return []
            
            print(f"   📊 Checking {len(response.data)} records...")
            
            # Calculate similarity for each record
            results = []
            for row in response.data:
                stored_embedding = row.get("embedding")
                
                if stored_embedding is None:
                    continue
                
                # Parse embedding (pgvector returns as string "[0.1,0.2,...]")
                if isinstance(stored_embedding, str):
                    try:
                        stored_embedding = [float(x) for x in stored_embedding.strip("[]").split(",")]
                    except (ValueError, AttributeError):
                        continue
                elif not isinstance(stored_embedding, list):
                    continue
                
                # Skip if dimensions don't match
                if len(stored_embedding) != len(query_embedding):
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= min_similarity:
                    created_at = None
                    if row.get("created_at"):
                        try:
                            ts = row["created_at"]
                            if isinstance(ts, str):
                                created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except Exception:
                            pass
                    
                    results.append(VectorSearchResult(
                        id=row["id"],
                        similarity=similarity,
                        text=row["text"],
                        metadata=row.get("metadata") or {},
                        created_at=created_at
                    ))
            
            # Sort by similarity (highest first) and limit
            results.sort(key=lambda x: x.similarity, reverse=True)
            
            if results:
                print(f"   ✅ Found {len(results)} matches (top: {results[0].similarity:.1%})")
            
            return results[:top_k]
            
        except Exception as e:
            print(f"⚠️  Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _fallback_search(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """Fallback search using direct query (less efficient but doesn't need RPC)."""
        try:
            # Fetch all records (filtered by metadata if possible)
            # Note: This loads embeddings into memory - not ideal for large datasets
            query = self._client.table(self.table_name).select("id, text, metadata, embedding, created_at")
            
            # JSONB containment filter using Supabase's contains() method
            if filter_metadata:
                query = query.contains("metadata", filter_metadata)
            
            # Limit to reasonable batch size
            response = query.limit(100).execute()
            
            if not response.data:
                print("   (No records found in table)")
                return []
            
            # Calculate similarity manually
            results = []
            for row in response.data or []:
                stored_embedding = row.get("embedding")
                
                # Handle different embedding formats from Supabase
                if stored_embedding is None:
                    continue
                
                # pgvector returns as string like "[0.1,0.2,...]" or as list
                if isinstance(stored_embedding, str):
                    try:
                        # Parse vector string format
                        stored_embedding = [float(x) for x in stored_embedding.strip("[]").split(",")]
                    except:
                        continue
                
                if not stored_embedding or len(stored_embedding) != len(query_embedding):
                    continue
                
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= min_similarity:
                    created_at = None
                    if row.get("created_at"):
                        try:
                            created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                        except:
                            pass
                    
                    results.append(VectorSearchResult(
                        id=row["id"],
                        similarity=similarity,
                        text=row["text"],
                        metadata=row.get("metadata") or {},
                        created_at=created_at
                    ))
            
            # Sort by similarity (highest first) and limit
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"⚠️  Fallback search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    async def delete(self, id: str) -> bool:
        """Delete a vector record by ID."""
        self._ensure_initialized()
        
        try:
            self._client.table(self.table_name).delete().eq("id", id).execute()
            return True
        except Exception as e:
            print(f"⚠️  Delete error: {e}")
            return False
    
    async def delete_by_metadata(self, filter_metadata: Dict[str, Any]) -> int:
        """Delete records matching metadata filters."""
        self._ensure_initialized()
        
        try:
            # First, find matching records
            query = self._client.table(self.table_name).select("id")
            
            # Use JSONB containment for filtering
            if filter_metadata:
                query = query.contains("metadata", filter_metadata)
            
            response = query.execute()
            
            if not response.data:
                return 0
            
            # Delete each matching record
            deleted = 0
            for row in response.data:
                try:
                    self._client.table(self.table_name).delete().eq("id", row["id"]).execute()
                    deleted += 1
                except Exception:
                    pass
            
            return deleted
            
        except Exception as e:
            print(f"⚠️  Delete by metadata error: {e}")
            return 0
    
    async def count(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count records in the store."""
        self._ensure_initialized()
        
        try:
            query = self._client.table(self.table_name).select("id", count="exact")
            
            # Use JSONB containment for filtering
            if filter_metadata:
                query = query.contains("metadata", filter_metadata)
            
            response = query.execute()
            return response.count or 0
            
        except Exception as e:
            print(f"⚠️  Count error: {e}")
            return 0
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self._client = None
        self._initialized = False
