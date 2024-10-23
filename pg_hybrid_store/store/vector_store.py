import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
import uuid
import pandas as pd
from typing import Any, List, Optional, Tuple, Union
from langchain.schema import Document

import asyncpg
from openai import AsyncOpenAI

from timescale_vector import client

from src.config.config import get_settings

from logging import getLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


class AsyncVectorStore:
    def __init__(self, vector_store_name: str = "vector_store"):
        assert vector_store_name.isidentifier() and not vector_store_name.startswith(
            "_"
        ), "Invalid vector_store_name. It must be a valid SQL identifier."
        self.settings = get_settings()
        self.openai_client = AsyncOpenAI(
            api_key=self.settings.llm_model.api_key,
        )
        self.embedding_model = self.settings.llm_model.embedding_model
        self.vector_store_name = vector_store_name or self.settings.database.table_name
        self.client = client.Async(
            service_url=self.settings.database.service_url,
            table_name=self.vector_store_name,
            num_dimensions=self.settings.database.num_dimensions,
            # time_partition_interval=self.settings.database.time_partition_interval,
        )

    async def init(self, recreate: bool = False, recreate_indexes: bool = False) -> None:
        await self._setup_vector_store(recreate, recreate_indexes)

    async def _setup_vector_store(self, recreate: bool = False, recreate_indexes: bool = False) -> None:
        if recreate:
            await self.drop_tables()
        await self.create_tables()
        if not recreate and recreate_indexes:
            await self.drop_indexes()
            logger.info(f"Indexes dropped for {self.vector_store_name}")
        await self.create_embedding_index()
        await self.create_bm25_search_index()
        logger.info("Vector store setup complete")

    async def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        await self.client.create_tables()
        logger.info(f"Tables created  {self.vector_store_name}")

    async def drop_tables(self) -> None:
        """Drop the tables in the database"""
        await self.client.drop_table()
        logger.info(f"Tables dropped  {self.vector_store_name}")

    async def create_embedding_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        try:
            await self.client.create_embedding_index(client.DiskAnnIndex())
            logger.info(f"DiskAnnIndex created for {self.vector_store_name}")
        except asyncpg.exceptions.DuplicateTableError as e:
            logger.info(f"Error while creating DiskAnnIndex: {str(e)}")

    async def drop_indexes(self, embedding_index: bool = True, bm25_index: bool = True) -> None:
        """Drop the StreamingDiskANN index in the database"""
        if embedding_index:
            await self.client.drop_embedding_index()
        if bm25_index:
            await self.drop_bm25_index()

    async def create_keyword_search_index(self):
        """Create a GIN index for keyword search if it doesn't exist."""
        index_name = f"idx_{self.vector_store_name}_contents_gin"
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {self.vector_store_name} USING gin(to_tsvector('french', contents));
        """
        try:
            conn = await asyncpg.connect(self.settings.database.service_url)
            async with conn.transaction():
                await conn.execute(create_index_sql)
                logger.info(f"GIN index '{index_name}' created or already exists.")
        except Exception as e:
            logger.error(f"Error while creating GIN index: {str(e)}")
            raise e
        finally:
            if conn:
                await conn.close()

    async def create_bm25_search_index(self):
        """Create a BM25 index for full-text search if it doesn't exist."""
        index_name = f"{self.vector_store_name}"
        create_bm25_index_sql = f"""
        CREATE EXTENSION IF NOT EXISTS pg_search;
        CALL paradedb.create_bm25(
            index_name => '{index_name}',
            table_name => '{self.vector_store_name}',
            text_fields => paradedb.field('contents'),
            key_field => 'id'
        );
        """
        try:
            conn = await asyncpg.connect(self.settings.database.service_url)
            async with conn.transaction():
                await conn.execute(create_bm25_index_sql)
                logger.info(f"BM25 index '{index_name}_bm25_index' created or already exists.")
        except asyncpg.exceptions.DuplicateTableError as e:
            logger.info(f"Error while creating BM25 index: {str(e)}")
        except Exception as e:
            logger.error(f"Error while creating BM25 index: {str(e)}")
            raise e

    async def drop_bm25_index(self):
        """Drop the BM25 index in the database"""
        index_name = f"{self.vector_store_name}"
        drop_index_sql = f"CALL paradedb.drop_bm25('{index_name}');"
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(drop_index_sql)
                logger.info(f"BM25 index '{index_name}_bm25_index' dropped successfully.")
        except Exception as e:
            logger.error(f"Error while dropping BM25 index: {str(e)}")
            raise e
        finally:
            if conn:
                await conn.close()

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
                await self.openai_client.embeddings.create(
                    input=[text],
                    model=self.embedding_model,
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
            A list of floats representing the embedding.
        """
        start_time = time.time()
        embeddings = (
            await self.openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model,
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

    async def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        await self.client.upsert(list(records))
        logging.info(f"Inserted {len(df)} records into {self.vector_store_name}")

    async def semantic_search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
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

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = await self.client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logger.info(f"Search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

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
        """
        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=["id", "metadata", "content", "embedding", "distance"])

        # Expand metadata column
        df = pd.concat([df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1)

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    async def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError("Provide exactly one of: ids, metadata_filter, or delete_all")

        if delete_all:
            await self.client.delete_all()
            logging.info(f"Deleted all records from {self.vector_store_name}")
        elif ids:
            await self.client.delete_by_ids(ids)
            logging.info(f"Deleted {len(ids)} records from {self.vector_store_name}")
        elif metadata_filter:
            resp = await self.client.delete_by_metadata(metadata_filter)
            logging.info(f"Deleted records matching metadata filter from {self.vector_store_name}, {resp}")

    async def keyword_search(
        self, query: str, limit: int = 5, return_dataframe: bool = True
    ) -> Union[List[Tuple[str, str, float]], pd.DataFrame]:
        """
        Perform a keyword search on the contents of the vector store.

        Args:
            query: The search query string.
            limit: The maximum number of results to return. Defaults to 5.
            return_dataframe: Whether to return results as a DataFrame. Defaults to True.

        Returns:
            Either a list of tuples (id, contents, rank) or a pandas DataFrame containing the search results.

        Example:
            results = vector_store.keyword_search("shipping options")
        """
        search_sql = f"""
        SELECT id, contents, ts_rank_cd(to_tsvector('french', contents), query) as rank
        FROM {self.vector_store_name}, websearch_to_tsquery('french', $1) query
        WHERE to_tsvector('french', contents) @@ query
        ORDER BY rank DESC
        LIMIT $2
        """

        start_time = time.time()

        conn = await asyncpg.connect(self.settings.database.service_url)
        async with conn.transaction():
            results = await conn.fetch(search_sql, query, limit)

        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            df = pd.DataFrame(results, columns=["id", "content", "rank"])
            df["id"] = df["id"].astype(str)
            return df
        else:
            return results

    async def _refresh_index(self):
        """Refresh the BM25 index in the database"""
        refresh_index_sql = f"VACUUM '{self.vector_store_name}';"
        try:
            conn = await asyncpg.connect(self.settings.database.service_url)
            async with conn.transaction():
                await conn.execute(refresh_index_sql)
        except Exception as e:
            logger.error(f"Error while refreshing BM25 index: {str(e)}")
            raise e
        finally:
            if conn:
                await conn.close()

    async def keyword_search_bm25(
        self, query: str, limit: int = 5, return_dataframe: bool = True
    ) -> Union[List[Tuple[str, str, float]], pd.DataFrame]:
        """Perform a BM25 keyword search on the contents of the vector store.

        Args:
            query (str): The search query string.
            limit (int, optional): The maximum number of results to return. Defaults to 5.
            return_dataframe (bool, optional): Whether to return results as a DataFrame. Defaults to True.

        Returns:
            Either a list of tuples (id, contents, rank) or a pandas DataFrame containing the search results.
        """
        search_sql = f"""
        SELECT id, contents, paradedb.score(id) as score
        FROM {self.vector_store_name}
        WHERE contents @@@ $1
        ORDER BY score DESC
        LIMIT $2
        """

        start_time = time.time()
        conn = await asyncpg.connect(self.settings.database.service_url)
        async with conn.transaction():
            results = await conn.fetch(search_sql, query, limit)

        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return pd.DataFrame(results, columns=["id", "content", "rank"])
        else:
            return results

    def _rerank_results(self, query: str, combined_results: pd.DataFrame, top_n: int) -> pd.DataFrame:
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

        return reranked_df.sort_values("relevance_score", ascending=False).drop(columns=["relevance_score"])

    async def hybrid_search(
        self,
        query: str,
        keyword_k: int = 5,
        semantic_k: int = 5,
        rerank: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Perform a hybrid search combining keyword and semantic search results,
        with optional reranking using Cohere.

        Args:
            query: The search query string.
            keyword_k: The number of results to return from keyword search. Defaults to 5.
            semantic_k: The number of results to return from semantic search. Defaults to 5.
            rerank: Whether to apply Cohere reranking. Defaults to True.
            top_n: The number of top results to return after reranking. Defaults to 5.

        Returns:
            A pandas DataFrame containing the combined search results with a 'search_type' column.
            dataframe with columns: id, content, search_type

        Example:
            results = vector_store.hybrid_search("shipping options", keyword_k=3, semantic_k=3, rerank=True, top_n=5)
        """
        # Perform keyword search
        keyword_results = await self.keyword_search_bm25(query, limit=keyword_k, return_dataframe=True)
        keyword_results["search_type"] = "keyword"
        keyword_results = keyword_results[["id", "content", "search_type"]]

        # Perform semantic search
        semantic_results = await self.semantic_search(query, limit=semantic_k, return_dataframe=True)
        semantic_results["search_type"] = "semantic"
        semantic_results = semantic_results[["id", "content", "search_type"]]

        # Combine results
        combined_results = pd.concat([keyword_results, semantic_results], ignore_index=True)
        combined_results["id"] = combined_results["id"].astype(str)
        logger.info(f"Combined results: {len(combined_results)}")
        combined_results = combined_results.drop_duplicates(subset=["id"])
        logger.info(f"Combined results after dropping duplicates: {len(combined_results)}")

        if rerank:
            return await self._rerank_results(query, combined_results, top_n)

        return combined_results

    @staticmethod
    async def list_vector_stores(service_url: str) -> List[str]:
        """
        List all vector stores in the database.

        Args:
            service_url: The URL of the DB service.

        Returns:
            A list of vector store names.
        """
        list_vector_stores_query = """
        SELECT 
            i.indrelid::regclass AS table_name
        FROM 
            pg_index i
        JOIN 
            pg_class c ON c.oid = i.indexrelid
        JOIN 
            pg_am am ON am.oid = c.relam
        WHERE 
            am.amname = 'diskann';
        """
        try:
            conn = await asyncpg.connect(service_url)
            async with conn.transaction():
                rows = await conn.fetch(list_vector_stores_query)
                vector_stores = [row["table_name"] for row in rows]
                return vector_stores
        except Exception as e:
            logger.error(f"Error while listing vector stores: {str(e)}")
            raise e
        finally:
            if conn:
                await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def test():
        vector_store = AsyncVectorStore("prestige_webb")

        # await vector_store.drop_tables()
        # await vector_store.create_tables()
        # # await vector_store.drop_indexes()
        # await vector_store.create_embedding_index()
        # await vector_store.create_keyword_search_index()
        # await vector_store.create_bm25_search_index()

        # # await vector_store.delete(delete_all=True)
        # await vector_store.upsert(
        #     pd.DataFrame(
        #         {
        #             "id": [str(uuid.uuid1()) for _ in range(4)],
        #             "metadata": [{"source": "test"}, {"source": "test"}, {"source": "test"}, {"source": "test"}],
        #             "contents": [
        #                 "Complicated sentence",
        #                 "Nothing similar, eveything dissimilar",
        #                 "Hello world",
        #                 "Hello world_2",
        #             ],
        #             "embedding": [
        #                 await vector_store.get_embedding("Complicated sentence"),
        #                 await vector_store.get_embedding("Nothing similar, eveything dissimilar"),
        #                 await vector_store.get_embedding("Hello world"),
        #                 await vector_store.get_embedding("Hello world_2"),
        #             ],
        #         }
        #     )
        # )
        pd.set_option("display.max_columns", None)
        query = "Comment vous contacter ?"
        # print((await vector_store.semantic_search(query)).head())
        # print((await vector_store.keyword_search_bm25(query)).head())
        # print((await vector_store.keyword_search(query)).head())
        print("--------------")
        # print(await vector_store.hybrid_search(query))
        print(await vector_store.list_vector_stores(os.getenv("TIMESCALE_SERVICE_URL")))

    asyncio.run(test())
