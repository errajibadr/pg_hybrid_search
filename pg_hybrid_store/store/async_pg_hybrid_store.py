from typing import List
import asyncpg
from openai import AsyncOpenAI
from pg_hybrid_store.config import PGHybridStoreSettings, get_settings
from pg_hybrid_store.search_types import SearchOptions, SearchResult
from pg_hybrid_store.store.base import BaseHybridStore

from timescale_vector import client as timescale_vector_client
from langchain_core.documents import Document

import logging

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

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()

    async def get_embedding(self, text: str) -> List[float]:
        pass

    async def hybrid_search(self, query: str, k: int = 10, search_options: SearchOptions = None) -> List[SearchResult]:
        pass

    async def semantic_search(
        self, query: str, k: int = 10, search_options: SearchOptions = None
    ) -> List[SearchResult]:
        pass

    async def keyword_search(self, query: str, k: int = 10, search_options: SearchOptions = None) -> List[SearchResult]:
        pass

    async def upsert(self, documents: List[Document]) -> None:
        pass
