from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional, Tuple

import pandas as pd

from pg_hybrid_store.search_types import SearchOptions, SearchResult


class BaseRetriever(ABC):
    """Base class for all retriever implementations."""

    @abstractmethod
    async def retrieve(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """Retrieve documents based on the query."""
        pass


class BaseVectorRetriever(BaseRetriever):
    """Base class for retrievers that use vector similarity."""

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        pass


class BaseKeywordRetriever(BaseRetriever):
    """Base class for retrievers that use keyword matching."""


class BaseHybridRetriever(BaseRetriever):
    """Base class for retrievers that combine multiple retrieval methods."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        semantic_search_only: bool = False,
        semantic_limit: int = 5,
        fulltext_limit: int = 5,
        metadata_filter: Optional[dict] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        predicates: Optional[List[Tuple[str, str, Any]]] = None,
    ) -> List[SearchResult]:
        """Retrieve documents using both methods."""
        pass

    @abstractmethod
    async def rerank_results(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank the combined results."""
        pass
