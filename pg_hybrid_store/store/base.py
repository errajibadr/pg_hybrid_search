from abc import ABC, abstractmethod


import pandas as pd


from pg_hybrid_store.search_types import EmbeddingVector


class BaseHybridStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def setup_store(self, recreate: bool = False, recreate_indexes: bool = False) -> None:
        """Setup the vector store"""
        pass

    @abstractmethod
    async def create_tables(self) -> None:
        """Create necessary database tables."""
        pass

    @abstractmethod
    async def drop_tables(self) -> None:
        """Drop database tables."""
        pass

    @abstractmethod
    async def create_embedding_index(self) -> None:
        """Create embedding search index."""
        pass

    @abstractmethod
    async def create_bm25_search_index(self) -> None:
        """Create BM25 search index."""
        pass

    @abstractmethod
    async def drop_indexes(self) -> None:
        """Drop embedding search index."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def upsert(self, df: pd.DataFrame) -> None:
        """Insert or update records."""
        pass

    @abstractmethod
    async def as_retriever(self) -> "BaseHybridStore":
        """Return a retriever for the vector store."""
        pass
