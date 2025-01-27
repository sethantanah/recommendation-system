from functools import lru_cache
from typing_extensions import Annotated
from fastapi import Depends
from src.config.settings import Settings
from src.database.mongodb_connector import MongoDBConnector
from src.database.vector_store import MongoDBVectorStore


@lru_cache(maxsize=None, typed=False)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=None, typed=False)
def get_mongo_db(
    settings: Annotated[Settings, Depends(get_settings)]
) -> MongoDBConnector:
    # Initialize the pipeline and vector store
    return MongoDBConnector(settings.PASSWORD, settings.DATABASE_NAME)


@lru_cache(maxsize=None, typed=False)
def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)]
) -> MongoDBVectorStore:
    # Initialize the pipeline and vector store
    return MongoDBVectorStore(
        settings.PASSWORD,
        settings.DATABASE_NAME,
        settings.SOURCE_COLLECTION,
        settings.VECTOR_COLLECTION,
    )
