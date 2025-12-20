"""
Abstract interface for embedding providers.
Converts text to vector representations for semantic search.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingInterface(ABC):
    """Abstract base class for all embedding providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the embedding provider.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in one call.
        More efficient than calling embed() multiple times.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            int: Number of dimensions (e.g., 1536 for text-embedding-3-small)
        """
        pass
    
    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        return "unknown"
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the embedding provider."""
        pass
