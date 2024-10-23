from datetime import datetime
from typing import Any, List, Optional, Tuple
import pandas as pd
from openai import AsyncOpenAI

import time

from pg_hybrid_store.store.async_pg_hybrid_store import AsyncPGHybridStore

from pg_hybrid_store.retrievers.base import BaseVectorRetriever, BaseKeywordRetriever, BaseHybridRetriever
from pg_hybrid_store.search_types import SearchOptions, SearchResult

# from pg_hybrid_store.logger import setup_logger
from pg_hybrid_store.config import get_settings

from timescale_vector import client

import logging

logger = logging.getLogger(__name__)


DEFAULT_SEARCH_OPTIONS = SearchOptions(
    limit=5, return_dataframe=True, time_range=None, metadata_filter=None, predicates=None
)


class OpenAIVectorRetriever(BaseVectorRetriever):
    """Vector retriever using OpenAI embeddings."""

    def __init__(self, store_client: AsyncPGHybridStore, embedding_model: str = None, embedding_dimensions: int = None):
        """Initialize the vector retriever."""

        self.settings = get_settings()
        self.store_client = store_client
        self.embedding_model = embedding_model or self.settings.vector_store.embedding_model
        self.embedding_dimensions = embedding_dimensions or self.settings.vector_store.embedding_dimensions
        self.client = AsyncOpenAI(api_key=self.settings.llm.api_key)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        return await self.store_client.get_embedding(text)

    async def retrieve(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult] | pd.DataFrame:
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
                vector_store.semantic_search("What are your shipping options?")
            Search with metadata filter:
                vector_store.semantic_search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                vector_store.semantic_search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.semantic_search("High-quality products", predicates=complex_pred)
        
        Time-based filtering:
            Search with time range:
                vector_store.semantic_search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        query_embedding = await self.get_embedding(query)

        start_time = time.time()
        if options is None:
            options = DEFAULT_SEARCH_OPTIONS

        search_args = {
            "limit": options.get("limit", 5),
        }

        if options.get("metadata_filter"):
            search_args["filter"] = options.get("metadata_filter")

        if options.get("predicates"):
            search_args["predicates"] = options.get("predicates")

        if options.get("time_range"):
            start_date, end_date = options.get("time_range")
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = await self.store_client.client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logger.info(f"Search completed in {elapsed_time:.3f} seconds")

        if options.get("return_dataframe"):
            return self._create_dataframe_from_results(results)
        else:
            search_results = [
                SearchResult(
                    id=str(r["id"]),
                    content=r["contents"],
                    metadata=r["metadata"],
                    distance=r["distance"],
                    search_type="semantic",
                )
                for r in results
            ]
            return search_results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
            DataFrame columns: id, metadata, content, distance, search_type
        """
        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=["id", "metadata", "content", "embedding", "distance"])

        # Expand metadata column
        # df = pd.concat([df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1)

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)
        df["search_type"] = "semantic"
        df = df.drop(columns=["embedding"])

        return df


class BM25KeywordRetriever(BaseKeywordRetriever):
    """Keyword retriever using BM25 ranking."""

    def __init__(self, vector_store: AsyncPGHybridStore):
        self.vector_store = vector_store

    async def retrieve(self, query: str, options: Optional[SearchOptions] = None) -> List[SearchResult] | pd.DataFrame:
        """Retrieve documents using BM25 ranking."""

        options = options or DEFAULT_SEARCH_OPTIONS
        search_sql = f"""
        SELECT id, metadata, contents, paradedb.score(id) as distance
        FROM {self.vector_store.vector_store_table}
        WHERE contents @@@ $1
        ORDER BY distance DESC
        LIMIT $2
        """

        async with self.vector_store.pool.acquire() as conn:
            results = await conn.fetch(search_sql, query, options.get("limit", 5))

        if options.get("return_dataframe"):
            return self._create_dataframe_from_results(results)
        else:
            return [
                SearchResult(
                    id=str(r["id"]),
                    content=r["contents"],
                    metadata=r["metadata"],
                    distance=r["distance"],
                    search_type="keyword",
                )
                for r in results
            ]

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
            DataFrame columns: id, metadata, content, distance, search_type
        """
        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=["id", "metadata", "content", "distance"])

        # Expand metadata column
        # df = pd.concat([df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1)

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)
        df["search_type"] = "keyword"

        return df


class HybridRetriever(BaseHybridRetriever):
    """Combines vector and keyword retrieval with optional reranking."""

    def __init__(
        self,
        vector_retriever: BaseVectorRetriever,
        keyword_retriever: BaseKeywordRetriever,
        use_reranking: bool = False,
    ):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.use_reranking = use_reranking

    async def retrieve(
        self,
        query: str,
        semantic_search_only: bool = False,
        fulltext_search_only: bool = False,
        semantic_limit: int = 5,
        fulltext_limit: int = 5,
        metadata_filter: Optional[dict] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        predicates: Optional[List[Tuple[str, str, Any]]] = None,
    ) -> List[SearchResult]:
        """Retrieve documents using both methods."""
        if semantic_search_only and fulltext_search_only:
            raise ValueError("Cannot perform both semantic and fulltext search simultaneously.")

        # Split the limit between both retrievers
        vector_options = SearchOptions(
            limit=semantic_limit, metadata_filter=metadata_filter, time_range=time_range, predicates=predicates
        )
        keyword_options = SearchOptions(limit=fulltext_limit, metadata_filter=metadata_filter)

        # Get results from both retrievers
        if not fulltext_search_only:
            vector_results = await self.vector_retriever.retrieve(query, vector_options)
        else:
            vector_results = []

        if not semantic_search_only:
            keyword_results = await self.keyword_retriever.retrieve(query, keyword_options)
        else:
            keyword_results = []

        # Combine results
        combined_results = vector_results + keyword_results

        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for result in combined_results:
            if result.get("id") not in seen_ids:
                seen_ids.add(result.get("id"))
                unique_results.append(result)

        if self.use_reranking:
            return await self.rerank_results(query, unique_results, semantic_limit)

        return unique_results[: semantic_limit + fulltext_limit]

    async def rerank_results(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank results using a reranking model."""
        # Note: This is a placeholder for reranking logic
        # You could implement different reranking strategies here
        # For example, using Cohere's reranking API or other methods
        return results[:top_k]
