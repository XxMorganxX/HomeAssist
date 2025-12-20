"""
Abstract interface for vector store providers.
Handles storage and retrieval of embedding vectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""
    id: str
    similarity: float  # 0.0 to 1.0, higher = more similar
    text: str  # The original text that was embedded
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None


@dataclass
class VectorRecord:
    """A record to store in the vector database."""
    id: str
    embedding: List[float]
    text: str  # Original text for reference
    metadata: Dict[str, Any]


class VectorStoreInterface(ABC):
    """Abstract base class for all vector store providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the vector store connection.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def store(self, record: VectorRecord) -> bool:
        """
        Store a single vector record.
        
        Args:
            record: The vector record to store
            
        Returns:
            bool: True if storage successful
        """
        pass
    
    @abstractmethod
    async def store_batch(self, records: List[VectorRecord]) -> int:
        """
        Store multiple vector records.
        
        Args:
            records: List of vector records to store
            
        Returns:
            int: Number of records successfully stored
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: The embedding to search for
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results ordered by similarity (highest first)
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a vector record by ID.
        
        Args:
            id: The ID of the record to delete
            
        Returns:
            bool: True if deletion successful
        """
        pass
    
    @abstractmethod
    async def delete_by_metadata(self, filter_metadata: Dict[str, Any]) -> int:
        """
        Delete records matching metadata filters.
        
        Args:
            filter_metadata: Metadata key-value pairs to match
            
        Returns:
            int: Number of records deleted
        """
        pass
    
    @abstractmethod
    async def count(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records in the store.
        
        Args:
            filter_metadata: Optional filters to apply
            
        Returns:
            int: Number of matching records
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        pass
