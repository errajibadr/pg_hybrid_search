from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from pg_hybrid_store.search_types import SearchOptions, SearchResult, EmbeddingVector


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
    async def semantic_search(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """Perform semantic search."""
        pass

    @abstractmethod
    async def keyword_search(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """Perform keyword search."""
        pass

    @abstractmethod
    async def hybrid_search(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """Perform hybrid search."""
        pass
