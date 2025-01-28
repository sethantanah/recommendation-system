from typing_extensions import Annotated
from fastapi import Depends, HTTPException, Query, APIRouter
from src.config.dependency import get_mongo_db, get_settings, get_vector_store

from src.config.settings import Settings
from src.database.mongodb_connector import MongoDBConnector
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.database.vector_store import MongoDBVectorStore
from src.config.logging import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/search")
def search_similar_vectors(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[MongoDBConnector, Depends(get_mongo_db)],
    vector_store: Annotated[MongoDBVectorStore, Depends(get_vector_store)],
    query: str,
    top_k: int = Query(default=5, ge=1, le=20),
):
    """
    Searches for similar vectors in the vector store.

    :param query: The query text.
    :param top_k: Number of similar vectors to return.
    """
    try:
        # Generate embedding for the query
        pipeline: IngestionPipeline = IngestionPipeline(
            source_db=database, vector_store=vector_store
        )
        embedding = pipeline.sbert_model.encode([query])[0]

        # Search for similar vectors
        results = vector_store.find_similar_vectors(
            embedding, top_k, collection_name=settings.SOURCE_COLLECTION
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching for similar vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))
