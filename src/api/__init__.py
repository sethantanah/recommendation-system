from fastapi import FastAPI, HTTPException, Query
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.database.vector_store import MongoDBVectorStore
from src.config.logging import setup_logger
from typing import List, Dict, Optional

logger = setup_logger(__name__)

app = FastAPI()

# Configuration
PASSWORD = "kanddle32"
DATABASE_NAME = "model_inventory_system"
SOURCE_COLLECTION = "source_data"
VECTOR_COLLECTION = "vector_data"

# Initialize the pipeline and vector store
pipeline = IngestionPipeline(PASSWORD, DATABASE_NAME, SOURCE_COLLECTION, VECTOR_COLLECTION)
vector_store = MongoDBVectorStore(PASSWORD, DATABASE_NAME, VECTOR_COLLECTION)

@app.post("/ingest")
def ingest_data():
    """
    Triggers the ingestion pipeline.
    """
    try:
        pipeline.run()
        return {"message": "Ingestion pipeline completed successfully."}
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors")
def add_vector(text: str, embedding: List[float], metadata: Dict):
    """
    Adds a new vector with metadata to the vector store.

    :param text: The text associated with the vector.
    :param embedding: The vector embedding.
    :param metadata: Metadata to store with the vector.
    """
    try:
        vector_doc = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
        vector_store.insert_vectors([vector_doc])
        return {"message": "Vector added successfully."}
    except Exception as e:
        logger.error(f"Error adding vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/vectors/{vector_id}")
def update_vector(vector_id: str, metadata: Dict, embedding: Optional[List[float]] = None):
    """
    Updates a vector document with new metadata and/or embedding.

    :param vector_id: The ID of the vector document to update.
    :param metadata: New metadata to update.
    :param embedding: New embedding to update (optional).
    """
    try:
        vector_store.update_vector(vector_id, metadata, embedding)
        return {"message": "Vector updated successfully."}
    except Exception as e:
        logger.error(f"Error updating vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vectors/{vector_id}")
def delete_vector(vector_id: str):
    """
    Deletes a vector document by its ID.

    :param vector_id: The ID of the vector document to delete.
    """
    try:
        vector_store.delete_vector(vector_id)
        return {"message": "Vector deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_similar_vectors(query: str, top_k: int = Query(default=5, ge=1, le=20)):
    """
    Searches for similar vectors in the vector store.

    :param query: The query text.
    :param top_k: Number of similar vectors to return.
    """
    try:
        # Generate embedding for the query
        embedding = pipeline.sbert_model.encode([query])[0]

        # Search for similar vectors
        results = vector_store.find_similar_vectors(embedding.tolist(), top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching for similar vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))