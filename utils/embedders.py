"""
Embedder utilities for AgentForge - Working embedder implementations.

Provides consistent embedder interfaces that work across all agents,
replacing problematic Agno embedder imports with sentence_transformers.
"""

from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
import numpy as np


class WorkingEmbedder:
    """
    Consistent working embedder using sentence_transformers.
    
    Replaces problematic Agno embedders with a reliable implementation.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with specified model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class QdrantCompatibleEmbedder:
    """
    Embedder compatible with QDrant operations.
    
    Provides the interface expected by QDrant while using sentence_transformers internally.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedder."""
        self.embedder = WorkingEmbedder(model_name)
        
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents for QDrant storage."""
        return self.embedder.embed_batch(documents)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query for QDrant search.""" 
        return self.embedder.embed(query)


# Factory functions for easy usage
def create_working_embedder(model_name: str = 'all-MiniLM-L6-v2') -> WorkingEmbedder:
    """Create a working embedder instance."""
    return WorkingEmbedder(model_name)


def create_qdrant_embedder(model_name: str = 'all-MiniLM-L6-v2') -> QdrantCompatibleEmbedder:
    """Create a QDrant-compatible embedder instance."""
    return QdrantCompatibleEmbedder(model_name)


# Default instances for import convenience
default_embedder = create_working_embedder()
qdrant_embedder = create_qdrant_embedder()