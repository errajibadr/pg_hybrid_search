from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional, Tuple

import pandas as pd

from pg_hybrid_store.search_types import HybridSearchResult, SearchOptions, SearchResult


class BaseRetriever(ABC):
    """Base class for all retriever implementations."""

    @abstractmethod
    async def retrieve(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> List[SearchResult]:
        """Retrieve documents based on the query.

        Returns:
            A list of SearchResult objects.
            SearchResult is a dictionary with the following keys:
            - id
            - content
            - metadata
            - distance
        """
        pass

    def _create_dataframe_from_results(
        self,
        results: List[SearchResult],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of SearchResult objects containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
            DataFrame columns: id, metadata, content, distance, search_type
        """
        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=["id", "metadata", "content", "distance", "search_type"])

        # Expand metadata column
        # df = pd.concat([df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1)

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df


class BaseVectorRetriever(BaseRetriever):
    """Base class for retrievers that use vector similarity."""

    @abstractmethod
    async def retrieve(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> List[SearchResult] | pd.DataFrame:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.
            DataFrame columns: id, metadata, content, embedding, distance

        Basic Examples:
            Basic search:
                vector_retriever.retrieve("What are your shipping options?")
            Search with metadata filter:
                vector_retriever.retrieve("Shipping options", metadata_filter={"category": "Shipping"})

        Predicates Examples:
            Search with predicates:
                vector_retriever.retrieve("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_retriever.retrieve("High-quality products", predicates=complex_pred)

        Time-based filtering:
            Search with time range:
                vector_retriever.retrieve("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        pass

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
    ) -> List[HybridSearchResult]:
        """Retrieve documents using both methods.

        Returns:
            A list of SearchResult objects.
            SearchResult is a dictionary with the following keys:
            - id
            - content
            - metadata
            - semantic_distance
            - fulltext_distance
        """
        pass

    @abstractmethod
    async def rerank_results(
        self, query: str, results: List[HybridSearchResult], top_k: int
    ) -> List[HybridSearchResult]:
        """Rerank the combined results."""
        pass
