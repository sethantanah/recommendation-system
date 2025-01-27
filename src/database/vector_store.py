from pymongo import MongoClient
from typing import List, Dict, Optional
from src.config.logging import setup_logger

logger = setup_logger(__name__)

class MongoDBVectorStore:
    def __init__(self, password: str, database_name: str, vector_collection_name: str):
        """
        Initializes the MongoDBVectorStore for storing and querying vector embeddings.

        :param password: MongoDB password.
        :param database_name: Name of the database.
        :param vector_collection_name: Name of the collection for storing vectors.
        """
        self.uri = f"mongodb+srv://kanddle:{password}@recommendationcluster.4559f.mongodb.net/?retryWrites=true&w=majority&appName=RecommendationCluster"
        self.database_name = database_name
        self.vector_collection_name = vector_collection_name
        self.client = MongoClient(self.uri)
        self.database = self.client[self.database_name]
        self.collection = self.database[self.vector_collection_name]

    def insert_vectors(self, vectors: List[Dict]):
        """
        Inserts a list of vector documents into the vector collection.

        :param vectors: List of vector documents to insert.
        """
        try:
            result = self.collection.insert_many(vectors)
            logger.info(f"Inserted {len(result.inserted_ids)} vectors into collection: {self.vector_collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    def update_vector(self, vector_id: str, metadata: Dict, embedding: Optional[List[float]] = None):
        """
        Updates a vector document with new metadata and/or embedding.

        :param vector_id: The ID of the vector document to update.
        :param metadata: New metadata to update.
        :param embedding: New embedding to update (optional).
        """
        try:
            update_data = {"metadata": metadata}
            if embedding:
                update_data["embedding"] = embedding

            result = self.collection.update_one(
                {"_id": vector_id},
                {"$set": update_data}
            )
            if result.modified_count > 0:
                logger.info(f"Updated vector with ID: {vector_id}")
            else:
                logger.warning(f"No vector found with ID: {vector_id}")
        except Exception as e:
            logger.error(f"Failed to update vector: {e}")
            raise

    def delete_vector(self, vector_id: str):
        """
        Deletes a vector document by its ID.

        :param vector_id: The ID of the vector document to delete.
        """
        try:
            result = self.collection.delete_one({"_id": vector_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted vector with ID: {vector_id}")
            else:
                logger.warning(f"No vector found with ID: {vector_id}")
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            raise

    def find_similar_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Finds similar vectors using cosine similarity.

        :param query_vector: The query vector to compare against.
        :param top_k: Number of similar vectors to return.
        :return: List of similar vectors.
        """
        try:
            # Use MongoDB's $vectorSearch or custom similarity search logic
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_vector,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": top_k,
                        "index": "vector_index"  # Ensure you have a vector index created
                    }
                }
            ]
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar vectors.")
            return results
        except Exception as e:
            logger.error(f"Failed to find similar vectors: {e}")
            raise