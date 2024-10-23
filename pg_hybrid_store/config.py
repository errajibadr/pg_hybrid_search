import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from dotenv import load_dotenv

from pg_hybrid_store.constants import EMBEDDING_MODEL_DIMENSIONS

import logging

logger = logging.getLogger(__name__)


load_dotenv()


class VectorStoreSettings(BaseSettings):
    """Settings for the vector store
    Default values:
    - embedding_model: text-embedding-3-small
    - embedding_dimensions: 1536
    """

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    @field_validator("embedding_dimensions")
    def set_embedding_dimensions(cls, v, values):
        embedding_model = values.data.get("embedding_model")
        if embedding_model in EMBEDDING_MODEL_DIMENSIONS:
            if v != EMBEDDING_MODEL_DIMENSIONS[embedding_model]:
                logger.warning("Embedding dimensions do not match the expected dimensions for the embedding model")
                return EMBEDDING_MODEL_DIMENSIONS[embedding_model]
        return v


class DatabaseSettings(BaseSettings):
    """Settings for the database
    Default values:
    - service_url: the URL of the PostgreSQL service
    """

    service_url: str = Field(default_factory=lambda: os.getenv("PG_SERVICE_URL"))


class LLMSettings(BaseSettings):
    """Settings for the LLM
    Default values:
    - api_key: the API key for the OpenAI API
    """

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


class PGHybridStoreSettings(BaseSettings):
    """Settings for the application
    Default values:
    - vector_store: the settings for the vector store
    - llm: the settings for the LLM
    - database: the settings for the database
    """

    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


def get_settings(vector_store_settings: Optional[VectorStoreSettings] = None) -> PGHybridStoreSettings:
    if vector_store_settings is None:
        vector_store_settings = VectorStoreSettings()
    return PGHybridStoreSettings(vector_store=vector_store_settings)
