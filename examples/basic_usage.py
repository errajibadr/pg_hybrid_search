import asyncio
import os
import uuid

from openai import AsyncOpenAI
from pg_hybrid_store.retrievers.retrievers import BM25KeywordRetriever, HybridRetriever, OpenAIVectorRetriever
from pg_hybrid_store.store.async_pg_hybrid_store import AsyncPGHybridStore
# from pg_hybrid_store.rag_ingestor import RAGIngestor

import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


async def _create_df(embedding_client):
    return pd.DataFrame(
        {
            "id": [uuid.uuid1() for _ in range(3)],
            "metadata": [{"source": "test"}, {"source": "test"}, {"source": "test"}],
            "contents": ["return policy", "warranty policy", "shipping policy"],
            "embedding": [
                (await embedding_client.embeddings.create(model="text-embedding-3-small", input=content))
                .data[0]
                .embedding
                for content in ["return policy", "warranty policy", "shipping policy"]
            ],
        }
    )


async def main():
    # Initialize vector store
    embedding_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = AsyncPGHybridStore(vector_store_table="example_store", embedding_client=embedding_client)
    await vector_store.setup_store(recreate=False, recreate_indexes=True)
    # df = await _create_df(embedding_client)
    # await vector_store.upsert(df)

    retriever = OpenAIVectorRetriever(vector_store_client=vector_store.client, embed_fn=vector_store.get_embedding)
    results = await retriever.retrieve("What is the return policy?")
    print(results)

    fulltext_retriever = BM25KeywordRetriever(
        vector_store_table="example_store", settings=vector_store.settings.database
    )
    results = await fulltext_retriever.retrieve("What is the return policy?")
    print(results)

    hybrid_retriever = HybridRetriever(retriever, fulltext_retriever)
    results = await hybrid_retriever.retrieve("What is the return policy?")
    print(results)

    # # Initialize RAG ingestor
    # ingestor = RAGIngestor(vector_store)

    # # Ingest PDFs
    # await ingestor.ingest_pdfs("path/to/pdfs", mode="overwrite")

    # # Perform a search
    # query = "What is the return policy?"
    # results = await vector_store.hybrid_search(query)

    # print(f"Results for query: '{query}'")
    # for result in results:
    #     print(f"- {result.page_content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
