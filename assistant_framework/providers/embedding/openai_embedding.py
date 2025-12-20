"""
OpenAI embedding provider.
Uses text-embedding-3-small or text-embedding-3-large for vector generation.
"""

import os
from typing import List, Dict, Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from ...interfaces.embedding import EmbeddingInterface
except ImportError:
    from assistant_framework.interfaces.embedding import EmbeddingInterface


class OpenAIEmbeddingProvider(EmbeddingInterface):
    """
    OpenAI embedding provider using the embeddings API.
    
    Supports:
    - text-embedding-3-small (1536 dims, $0.02/1M tokens) - recommended
    - text-embedding-3-large (3072 dims, $0.13/1M tokens) - higher quality
    - text-embedding-ada-002 (1536 dims, $0.10/1M tokens) - legacy
    """
    
    # Dimension lookup for known models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            config: Configuration dict with:
                - api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
                - model: Model name (default: text-embedding-3-small)
        """
        if AsyncOpenAI is None:
            raise ImportError("openai library required. Install with: pip install openai")
        
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self._model = config.get("model", "text-embedding-3-small")
        self._client: Optional[AsyncOpenAI] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            print(f"✅ OpenAI Embedding provider initialized (model: {self._model})")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI Embedding provider: {e}")
            return False
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        if not self._initialized or not self._client:
            await self.initialize()
        
        # Clean and validate text
        text = text.strip()
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.dimensions
        
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in one API call.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized or not self._client:
            await self.initialize()
        
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [t.strip() if t else "" for t in texts]
        
        # Handle empty texts
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
        non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]
        
        if not non_empty_texts:
            # All empty, return zero vectors
            return [[0.0] * self.dimensions for _ in texts]
        
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=non_empty_texts,
                encoding_format="float"
            )
            
            # Build result with zero vectors for empty texts
            result = [[0.0] * self.dimensions for _ in texts]
            for idx, embedding_data in zip(non_empty_indices, response.data):
                result[idx] = embedding_data.embedding
            
            return result
            
        except Exception as e:
            print(f"⚠️  Batch embedding error: {e}")
            raise
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for the current model."""
        return self.MODEL_DIMENSIONS.get(self._model, 1536)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self._client = None
        self._initialized = False
