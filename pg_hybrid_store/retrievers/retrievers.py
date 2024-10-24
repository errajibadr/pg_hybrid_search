import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Optional, Tuple

import asyncpg
import pandas as pd
from timescale_vector import client

from pg_hybrid_store.config import DatabaseSettings
from pg_hybrid_store.retrievers.base import (
    BaseHybridRetriever,
    BaseKeywordRetriever,
    BaseVectorRetriever,
)
from pg_hybrid_store.search_types import HybridSearchResult, SearchOptions, SearchResult

logger = logging.getLogger(__name__)


DEFAULT_SEARCH_OPTIONS = SearchOptions(
    limit=5,
    return_dataframe=True,
    time_range=None,
    metadata_filter=None,
    predicates=None,
)


class OpenAIVectorRetriever(BaseVectorRetriever):
    """Vector retriever using OpenAI embeddings."""

    def __init__(self, vector_store_client: client.Async, embed_fn: callable):
        """Initialize the vector retriever."""
        self.vector_store_client = vector_store_client
        self.embed_fn = embed_fn

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        return await self.embed_fn(text)

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

        results = await self.vector_store_client.search(query_embedding, **search_args)
        print(results[0])
        elapsed_time = time.time() - start_time

        logger.info(f"Search completed in {elapsed_time:.3f} seconds")

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

        if options.get("return_dataframe"):
            return self._create_dataframe_from_results(search_results)
        return search_results


class BM25KeywordRetriever(BaseKeywordRetriever):
    """Keyword retriever using BM25 ranking."""

    def __init__(self, vector_store_table: str, settings: DatabaseSettings):
        self.vector_store_table = vector_store_table
        self.pool = None
        self.db_settings = settings

    async def initialize(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.db_settings.service_url)

    @asynccontextmanager
    async def get_connection(self):
        async with self.pool.acquire() as conn:
            yield conn

    async def retrieve(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> List[SearchResult] | pd.DataFrame:
        """Retrieve documents using BM25 ranking."""
        if self.pool is None:
            await self.initialize()

        options = options or DEFAULT_SEARCH_OPTIONS
        search_sql = f"""
        SELECT id, metadata, contents, paradedb.score(id) as distance
        FROM {self.vector_store_table}
        WHERE contents @@@ $1
        ORDER BY distance DESC
        LIMIT $2
        """

        async with self.get_connection() as conn:
            results = await conn.fetch(search_sql, query, options["limit"])

        search_results = [
            SearchResult(
                id=str(r["id"]),
                content=r["contents"],
                metadata=r["metadata"],
                distance=r["distance"],
                search_type="fulltext",
            )
            for r in results
        ]

        if options.get("return_dataframe"):
            return self._create_dataframe_from_results(search_results)

        return search_results


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
        semantic_limit: int = 5,
        fulltext_limit: int = 5,
        metadata_filter: Optional[dict] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        predicates: Optional[List[Tuple[str, str, Any]]] = None,
        semantic_search_only: bool = False,
        fulltext_search_only: bool = False,
    ) -> List[HybridSearchResult]:
        """Retrieve documents using both semantic and fulltext methods."""
        if semantic_search_only and fulltext_search_only:
            raise ValueError("Cannot perform both semantic and fulltext search simultaneously.")

        vector_results = await self._get_vector_results(
            query,
            semantic_limit,
            metadata_filter,
            time_range,
            predicates,
            fulltext_search_only,
        )
        keyword_results = await self._get_keyword_results(
            query, fulltext_limit, metadata_filter, semantic_search_only
        )

        combined_results = self._combine_and_deduplicate_results(vector_results, keyword_results)

        if self.use_reranking:
            return await self.rerank_results(
                query, combined_results, semantic_limit + fulltext_limit
            )

        return combined_results[: semantic_limit + fulltext_limit]

    async def _get_vector_results(
        self,
        query,
        limit,
        metadata_filter,
        time_range,
        predicates,
        fulltext_search_only,
    ):
        if fulltext_search_only:
            return []
        vector_options = SearchOptions(
            limit=limit,
            metadata_filter=metadata_filter,
            time_range=time_range,
            predicates=predicates,
        )
        results = await self.vector_retriever.retrieve(query, vector_options)
        return results

    async def _get_keyword_results(self, query, limit, metadata_filter, semantic_search_only):
        if semantic_search_only:
            return []
        keyword_options = SearchOptions(limit=limit, metadata_filter=metadata_filter)
        results = await self.keyword_retriever.retrieve(query, keyword_options)
        return results

    def _combine_and_deduplicate_results(
        self, vector_results: List[SearchResult], keyword_results: List[SearchResult]
    ) -> List[HybridSearchResult]:
        combined_results = vector_results + keyword_results
        seen_ids = set()
        unique_results = []
        for result in combined_results:
            if result["id"] not in seen_ids:
                hybrid_result: HybridSearchResult = {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_distance": (
                        result["distance"] if result["search_type"] == "semantic" else None
                    ),
                    "fulltext_distance": (
                        result["distance"] if result["search_type"] == "fulltext" else None
                    ),
                }
                seen_ids.add(result["id"])
                unique_results.append(hybrid_result)
            else:
                # If the result exists in both, update the existing result
                existing_result = next(r for r in unique_results if r["id"] == result["id"])
                if result["search_type"] == "semantic":
                    existing_result["semantic_distance"] = result["distance"]
                elif result["search_type"] == "fulltext":
                    existing_result["fulltext_distance"] = result["distance"]
        return unique_results

    async def rerank_results(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Rerank results using a reranking model."""
        # Note: This is a placeholder for reranking logic
        # You could implement different reranking strategies here
        # For example, using Cohere's reranking API or other methods
        return results[:top_k]

    def _rerank_results(
        self, query: str, combined_results: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """
        Rerank the combined search results using Cohere.

        Args:
            query: The original search query.
            combined_results: DataFrame containing the combined keyword and semantic search results.
            top_n: The number of top results to return after reranking.

        Returns:
            A pandas DataFrame containing the reranked results.
        """
        rerank_results = self.cohere_client.v2.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=combined_results["content"].tolist(),
            top_n=top_n,
            return_documents=True,
        )

        reranked_df = pd.DataFrame(
            [
                {
                    "id": combined_results.iloc[result.index]["id"],
                    "content": result.document,
                    "search_type": combined_results.iloc[result.index]["search_type"],
                    "relevance_score": result.relevance_score,
                }
                for result in rerank_results.results
            ]
        )

        return reranked_df.sort_values("relevance_score", ascending=False).drop(
            columns=["relevance_score"]
        )
