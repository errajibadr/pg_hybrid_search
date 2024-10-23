import asyncio
import os

from openai import AsyncOpenAI
from pg_hybrid_store.store.async_pg_hybrid_store import AsyncPGHybridStore
# from pg_hybrid_store.rag_ingestor import RAGIngestor

import logging

logging.basicConfig(level=logging.INFO)


async def main():
    # Initialize vector store
    embedding_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = AsyncPGHybridStore(vector_store_table="example_store", embedding_client=embedding_client)
    await vector_store.setup_store(recreate=False, recreate_indexes=True)

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
