import logging
import time
import uuid
from typing import Any, List, Tuple

import asyncpg
import pandas as pd
from langchain_core.documents import Document
from openai import AsyncOpenAI
from timescale_vector import client as timescale_vector_client

from pg_hybrid_store.config import PGHybridStoreSettings, get_settings
from pg_hybrid_store.retrievers.retrievers import (
    BM25KeywordRetriever,
    HybridRetriever,
    OpenAIVectorRetriever,
)
from pg_hybrid_store.store.base import BaseHybridStore

logger = logging.getLogger(__name__)


class AsyncPGHybridStore(BaseHybridStore):
    def __init__(
        self,
        vector_store_table: str,
        embedding_client: AsyncOpenAI,
        settings: PGHybridStoreSettings | None = None,
        **kwargs,
    ):
        assert vector_store_table.isidentifier() and not vector_store_table.startswith(
            "_"
        ), "Invalid vector_store_table. It must be a valid SQL identifier."

        if settings is None:
            settings = get_settings()
        self.settings = settings
        self.vector_store_table = vector_store_table
        self.embedding_client = embedding_client

        self.client = timescale_vector_client.Async(
            service_url=self.settings.database.service_url,
            table_name=self.vector_store_table,
            num_dimensions=self.settings.vector_store.embedding_dimensions,
            # time_partition_interval=self.settings.database.time_partition_interval,
        )

        self.pool = None

    async def initialize_pool(self):
        """Initialize the connection pool"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.settings.database.service_url)

    async def setup_store(self, recreate: bool = False, recreate_indexes: bool = False) -> None:
        """Setup the vector store"""
        await self.initialize_pool()
        if recreate:
            await self.drop_tables()
        await self.create_tables()
        if not recreate and recreate_indexes:
            await self.drop_indexes()
            logger.info(f"Indexes dropped for {self.vector_store_table}")
        await self.create_embedding_index()
        await self.create_bm25_search_index()
        logger.info("Vector store setup complete")

    async def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        await self.client.create_tables()
        logger.info(f"Tables created  {self.vector_store_table}")

    async def drop_tables(self) -> None:
        """Drop the tables in the database"""
        await self.client.drop_table()
        logger.info(f"Tables dropped  {self.vector_store_table}")

    async def create_embedding_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        try:
            await self.client.create_embedding_index(timescale_vector_client.DiskAnnIndex())
            logger.info(f"DiskAnnIndex created for {self.vector_store_table}")
        except asyncpg.exceptions.DuplicateTableError as e:
            logger.info(f"Error while creating DiskAnnIndex: {str(e)}")

    async def create_bm25_search_index(self):
        """Create a BM25 index for full-text search if it doesn't exist."""
        index_name = f"{self.vector_store_table}"
        create_bm25_index_sql = f"""
        CREATE EXTENSION IF NOT EXISTS pg_search;
        CALL paradedb.create_bm25(
            index_name => '{index_name}',
            table_name => '{self.vector_store_table}',
            text_fields => paradedb.field('contents'),
            key_field => 'id'
        );
        """

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(create_bm25_index_sql)
                logger.info(f"BM25 index '{index_name}_bm25_index' created or already exists.")
        except asyncpg.exceptions.DuplicateTableError as e:
            logger.info(f"Error while creating BM25 index: {str(e)}")
        except Exception as e:
            logger.error(f"Error while creating BM25 index: {str(e)}")
            raise e

    async def drop_indexes(self, embedding_index: bool = True, bm25_index: bool = True) -> None:
        """Drop the StreamingDiskANN index in the database"""
        if embedding_index:
            await self.client.drop_embedding_index()
        if bm25_index:
            await self.drop_bm25_index()

    async def drop_bm25_index(self):
        """Drop the BM25 index in the database"""
        index_name = f"{self.vector_store_table}"
        drop_index_sql = f"CALL paradedb.drop_bm25('{index_name}');"
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(drop_index_sql)
            logger.info(f"BM25 index '{index_name}_bm25_index' dropped successfully.")
        except Exception as e:
            logger.error(f"Error while dropping BM25 index: {str(e)}")
            raise e

    async def _refresh_index(self):
        """Refresh the BM25 index in the database"""
        refresh_index_sql = f"VACUUM '{self.vector_store_table}';"
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(refresh_index_sql)
        except Exception as e:
            logger.error(f"Error while refreshing BM25 index: {str(e)}")
            raise e

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            (
                await self.embedding_client.embeddings.create(
                    input=[text],
                    model=self.settings.vector_store.embedding_model,
                )
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logger.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    async def get_embeddings(self, texts: List[str]) -> List[Tuple[str, List[float]]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: The input texts to generate embeddings for.

        Returns:
            A list of tuples containing the index and the embedding.
        """
        start_time = time.time()
        embeddings = (
            await self.embedding_client.embeddings.create(
                input=texts,
                model=self.settings.vector_store.embedding_model,
            )
        ).data
        elapsed_time = time.time() - start_time
        logging.info(f"Embeddings generated in {elapsed_time:.3f} seconds")
        return [(embedding.index, embedding.embedding) for embedding in embeddings]

    async def embed_documents(self, documents: List[Document]) -> List[dict[str, Any]]:
        """Embed a list of documents"""
        embeddings = await self.get_embeddings([doc.page_content for doc in documents])

        documents_formatted = [
            {
                "id": str(uuid.uuid1()),
                "metadata": documents[ix].metadata,
                "contents": documents[ix].page_content,
                "embedding": embedding,
            }
            for ix, embedding in embeddings
        ]

        return documents_formatted

    def as_retriever(self) -> HybridRetriever:
        vector_retriever = OpenAIVectorRetriever(
            vector_store_client=self.client, embed_fn=self.get_embedding
        )
        keyword_retriever = BM25KeywordRetriever(
            vector_store_table=self.vector_store_table, settings=self.settings.database
        )
        return HybridRetriever(vector_retriever, keyword_retriever)

    async def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        await self.client.upsert(list(records))
        logger.info(f"Inserted {len(df)} records into {self.vector_store_table}")
