from typing import Dict, List
from typing_extensions import Annotated
from fastapi import Depends, HTTPException, APIRouter
from src.config.dependency import get_mongo_db, get_settings, get_vector_store

from src.config.settings import Settings
from src.database.mongodb_connector import MongoDBConnector
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.database.vector_store import MongoDBVectorStore
from src.config.logging import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.put("/vectors/embeddings")
def get_vector_embedding(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[MongoDBConnector, Depends(get_mongo_db)],
    vector_store: Annotated[MongoDBVectorStore, Depends(get_vector_store)],
    models_metadata: List[Dict],
):
    """
    Updates a vector document with new metadata and/or embedding.
    """
    try:
        pipeline: IngestionPipeline = IngestionPipeline(
            source_db=database, vector_store=vector_store
        )
        embeddings = []

        for model_metadata in models_metadata:
            embedding = pipeline.get_embedding(model_metadata)
            embeddings.append(embedding)

        return embeddings

    except Exception as e:
        logger.error(f"Error updating vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vectors/update")
def update_vector(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[MongoDBConnector, Depends(get_mongo_db)],
    vector_store: Annotated[MongoDBVectorStore, Depends(get_vector_store)],
    vector_ids: List[str],
):
    """
    Updates a vector document with new metadata and/or embedding.

    :param vector_ids: The IDs of the vector document to update.
    """
    try:
        pipeline: IngestionPipeline = IngestionPipeline(
            source_db=database, vector_store=vector_store
        )

        for vector_id in vector_ids:
            model_metadata = database.get_model(
                settings.SOURCE_COLLECTION, {"_id": vector_id}
            )[0]

            embedding = pipeline.get_embedding(model_metadata)
            model_metadata["embedding"] = embedding

            database.update_vector(
                settings.SOURCE_COLLECTION, vector_id, model_metadata
            )

            model_metadata = database.get_model(
                settings.SOURCE_COLLECTION, {"_id": vector_id}
            )[0]

            embedding = pipeline.get_embedding(model_metadata)
            model_metadata["embedding"] = embedding

            database.update_vector(
                settings.SOURCE_COLLECTION, vector_id, model_metadata
            )

        return {"message": "documents updated successfully."}
    except Exception as e:
        logger.error(f"Error updating vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))