from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from pg_hybrid_store.config import DatabaseSettings
from pg_hybrid_store.retrievers.retrievers import (
    BM25KeywordRetriever,
    HybridRetriever,
    HybridSearchResult,
    OpenAIVectorRetriever,
    SearchResult,
)


@pytest.fixture(scope="function")
def mock_vector_store_client():
    return AsyncMock()


@pytest.fixture(scope="function")
def mock_embed_fn():
    return AsyncMock(return_value=[0.1, 0.2, 0.3])


@pytest.fixture(scope="function")
def mock_db_pool():
    return AsyncMock()


@pytest.fixture(scope="function")
def sample_search_results():
    return [
        {
            "id": "1",
            "contents": "Content 1",
            "metadata": {"key": "value1"},
            "distance": 0.1,
        },
        {
            "id": "2",
            "contents": "Content 2",
            "metadata": {"key": "value2"},
            "distance": 0.2,
        },
    ]


@pytest.mark.asyncio
async def test_openai_vector_retriever(
    mock_vector_store_client, mock_embed_fn, sample_search_results
):
    retriever = OpenAIVectorRetriever(mock_vector_store_client, mock_embed_fn)
    mock_vector_store_client.search.return_value = sample_search_results

    results = await retriever.retrieve("test query")

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert list(results.columns) == [
        "id",
        "metadata",
        "content",
        "distance",
        "search_type",
    ]
    assert results["search_type"].unique() == ["semantic"]

    mock_embed_fn.assert_called_once_with("test query")
    mock_vector_store_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_bm25_keyword_retriever(mock_db_pool):
    settings = DatabaseSettings(service_url="postgresql://user:pass@localhost/testdb")
    retriever = BM25KeywordRetriever("test_table", settings)
    retriever.pool = mock_db_pool

    # Create a mock connection
    mock_connection = AsyncMock()
    mock_connection.fetch.return_value = [
        {
            "id": "1",
            "contents": "Content 1",
            "metadata": {"key": "value1"},
            "distance": 0.1,
        },
        {
            "id": "2",
            "contents": "Content 2",
            "metadata": {"key": "value2"},
            "distance": 0.2,
        },
    ]

    # Create an async context manager that returns mock_connection
    class MockAcquireContextManager:
        async def __aenter__(self):
            return mock_connection

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_db_pool.acquire = MagicMock(return_value=MockAcquireContextManager())
    results = await retriever.retrieve("test query")

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert list(results.columns) == [
        "id",
        "metadata",
        "content",
        "distance",
        "search_type",
    ]
    assert results["search_type"].unique() == ["fulltext"]

    mock_db_pool.acquire.assert_called_once()
    mock_connection.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_retriever():
    mock_vector_retriever = AsyncMock()
    mock_keyword_retriever = AsyncMock()

    vector_results = [
        SearchResult(
            id="1",
            content="Content 1",
            metadata={"key": "value1"},
            distance=0.1,
            search_type="semantic",
        ),
        SearchResult(
            id="2",
            content="Content 2",
            metadata={"key": "value2"},
            distance=0.2,
            search_type="semantic",
        ),
    ]
    keyword_results = [
        SearchResult(
            id="3",
            content="Content 3",
            metadata={"key": "value3"},
            distance=0.3,
            search_type="fulltext",
        ),
        SearchResult(
            id="4",
            content="Content 4",
            metadata={"key": "value4"},
            distance=0.4,
            search_type="fulltext",
        ),
    ]

    mock_vector_retriever.retrieve.return_value = vector_results
    mock_keyword_retriever.retrieve.return_value = keyword_results

    retriever = HybridRetriever(mock_vector_retriever, mock_keyword_retriever)

    results = await retriever.retrieve("test query", semantic_limit=2, fulltext_limit=2)

    assert len(results) == 4
    assert all(
        isinstance(r, dict) and set(r.keys()) == set(HybridSearchResult.__annotations__.keys())
        for r in results
    )
    assert [r["id"] for r in results] == ["1", "2", "3", "4"]

    mock_vector_retriever.retrieve.assert_called_once()
    mock_keyword_retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_retriever_semantic_only():
    mock_vector_retriever = AsyncMock()
    mock_keyword_retriever = AsyncMock()

    vector_results = [
        SearchResult(
            id="1",
            content="Content 1",
            metadata={"key": "value1"},
            distance=0.1,
            search_type="semantic",
        ),
        SearchResult(
            id="2",
            content="Content 2",
            metadata={"key": "value2"},
            distance=0.2,
            search_type="semantic",
        ),
    ]

    mock_vector_retriever.retrieve.return_value = vector_results

    retriever = HybridRetriever(mock_vector_retriever, mock_keyword_retriever)

    results = await retriever.retrieve(
        "test query", semantic_limit=2, fulltext_limit=2, semantic_search_only=True
    )

    assert len(results) == 2
    assert all(
        isinstance(r, dict) and set(r.keys()) == set(HybridSearchResult.__annotations__.keys())
        for r in results
    )
    assert [r["id"] for r in results] == ["1", "2"]

    mock_vector_retriever.retrieve.assert_called_once()
    mock_keyword_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_retriever_fulltext_only():
    mock_vector_retriever = AsyncMock()
    mock_keyword_retriever = AsyncMock()

    keyword_results = [
        SearchResult(
            id="3",
            content="Content 3",
            metadata={"key": "value3"},
            distance=0.3,
            search_type="fulltext",
        ),
        SearchResult(
            id="4",
            content="Content 4",
            metadata={"key": "value4"},
            distance=0.4,
            search_type="fulltext",
        ),
    ]

    mock_keyword_retriever.retrieve.return_value = keyword_results

    retriever = HybridRetriever(mock_vector_retriever, mock_keyword_retriever)

    results = await retriever.retrieve(
        query="test query", semantic_limit=2, fulltext_limit=2, fulltext_search_only=True
    )

    assert len(results) == 2
    assert all(
        isinstance(r, dict) and set(r.keys()) == set(HybridSearchResult.__annotations__.keys())
        for r in results
    )
    assert [r["id"] for r in results] == ["3", "4"]

    mock_vector_retriever.retrieve.assert_not_called()
    mock_keyword_retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_retriever_with_reranking():
    mock_vector_retriever = AsyncMock()
    mock_keyword_retriever = AsyncMock()

    vector_results = [
        SearchResult(
            id="1",
            content="Content 1",
            metadata={"key": "value1"},
            distance=0.1,
            search_type="semantic",
        ),
        SearchResult(
            id="2",
            content="Content 2",
            metadata={"key": "value2"},
            distance=0.2,
            search_type="semantic",
        ),
    ]
    keyword_results = [
        SearchResult(
            id="1",
            content="Content 1",
            metadata={"key": "value1"},
            distance=0.3,
            search_type="fulltext",
        ),
        SearchResult(
            id="4",
            content="Content 4",
            metadata={"key": "value4"},
            distance=0.4,
            search_type="fulltext",
        ),
    ]

    mock_vector_retriever.retrieve.return_value = vector_results
    mock_keyword_retriever.retrieve.return_value = keyword_results

    retriever = HybridRetriever(mock_vector_retriever, mock_keyword_retriever, use_reranking=True)

    with patch.object(retriever, "rerank_results", new_callable=AsyncMock) as mock_rerank:
        mock_rerank.return_value = [
            {
                "id": "2",
                "content": "Content 2",
                "metadata": {"key": "value2"},
                "semantic_distance": 0.2,
                "fulltext_distance": None,
            },
            {
                "id": "3",
                "content": "Content 3",
                "metadata": {"key": "value3"},
                "semantic_distance": None,
                "fulltext_distance": 0.3,
            },
        ]

        results = await retriever.retrieve("test query", semantic_limit=2, fulltext_limit=2)

        assert len(results) == 2
        assert all(
            isinstance(r, dict) and set(r.keys()) == set(HybridSearchResult.__annotations__.keys())
            for r in results
        )
        assert [r["id"] for r in results] == ["2", "3"]

        mock_rerank.assert_called_once()
