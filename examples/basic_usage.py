import asyncio
import logging
import os
import uuid

# from pg_hybrid_store.rag_ingestor import RAGIngestor
import pandas as pd
from openai import AsyncOpenAI

from pg_hybrid_store.retrievers.retrievers import (
    BM25KeywordRetriever,
    HybridRetriever,
    OpenAIVectorRetriever,
)
from pg_hybrid_store.store.async_pg_hybrid_store import AsyncPGHybridStore

logging.basicConfig(level=logging.INFO)


async def _create_df(embedding_client):
    return pd.DataFrame(
        {
            "id": [uuid.uuid1() for _ in range(3)],
            "metadata": [{"source": "test"}, {"source": "test"}, {"source": "test"}],
            "contents": ["return policy", "warranty policy", "shipping policy"],
            "embedding": [
                (
                    await embedding_client.embeddings.create(
                        model="text-embedding-3-small", input=content
                    )
                )
                .data[0]
                .embedding
                for content in ["return policy", "warranty policy", "shipping policy"]
            ],
        }
    )


async def main():
    # Initialize vector store
    embedding_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = AsyncPGHybridStore(
        vector_store_table="example_store", embedding_client=embedding_client
    )

    ## Setup store and upsert data
    await vector_store.setup_store(recreate=True, recreate_indexes=True)
    df = await _create_df(embedding_client)
    await vector_store.upsert(df)

    ## Retrieve using vector store
    retriever = OpenAIVectorRetriever(
        vector_store_client=vector_store.client, embed_fn=vector_store.get_embedding
    )
    results = await retriever.retrieve("What is the return policy?")
    print(results)

    ## Retrieve using fulltext search
    fulltext_retriever = BM25KeywordRetriever(
        vector_store_table="example_store", settings=vector_store.settings.database
    )
    results = await fulltext_retriever.retrieve("What is the return policy?")
    print(results)

    ## Retrieve using hybrid search
    hybrid_retriever = HybridRetriever(retriever, fulltext_retriever)
    results = await hybrid_retriever.retrieve("What is the return policy?")
    print(results)

    ## Retrieve using vector store as retriever
    hybrid_retriever = vector_store.as_retriever()
    results = await hybrid_retriever.retrieve("What is the return policy?")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
