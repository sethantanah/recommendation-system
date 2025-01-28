from src.database.mongodb_connector import MongoDBConnector
from src.database.vector_store import MongoDBVectorStore
from src.models.sbert_model import SBERTModel
from src.utils.data_processor import models_to_text
from src.config.logging import setup_logger
from typing import Optional
import math

logger = setup_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        source_db: MongoDBConnector,
        vector_store: MongoDBVectorStore,
    ):
        """
        Initializes the batch ingestion pipeline.

        Args:
            password: MongoDB password
            database_name: Name of the database
            source_collection: Name of the source collection
            vector_collection: Name of the vector collection
        """
        self.source_db = source_db
        self.vector_store = vector_store
        self.sbert_model = SBERTModel()

    def _single_process(self, data: dict):
        """
        Process a single data.

        Args:
            data: List of documents to process
        """
        # Process the batch
        processed_data = models_to_text([data])

        # Generate embeddings
        texts = [item["content"] for item in processed_data]
        embeddings = self.sbert_model.encode(texts)

        return embeddings

    def _process_batch(self, batch_data):
        """
        Process a single batch of data.

        Args:
            batch_data: List of documents to process
        """
        # Process the batch
        processed_data = models_to_text(batch_data)

        # # Generate embeddings
        model_ids = [item["_id"] for item in processed_data]
        texts = [item["content"] for item in processed_data]
        embeddings = self.sbert_model.encode(texts)

        # Prepare documents for vector store
        vector_docs = [
            {"_id": model_id, "text": text, "embedding": embedding}
            for model_id, text, embedding in zip(model_ids, texts, embeddings)
        ]

        # Store in vector database
        self.vector_store.insert_vectors(vector_docs)

        return len(vector_docs)

    def run(
        self,
        collection_name: str,
        page_size: Optional[int] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ):
        """
        Runs the batch ingestion pipeline with pagination support.

        Args:
            page_size: Number of documents to process in each batch
            start_page: Optional starting page number (1-based indexing)
            end_page: Optional ending page number (inclusive)
        """
        try:
            self.source_collection = collection_name
            # Get total document count
            total_docs = self.source_db.count_documents(self.source_collection)
            total_pages = math.ceil(total_docs / page_size)

            # Validate and adjust page ranges
            current_page = max(1, start_page if start_page else 1)
            end_page = min(total_pages, end_page if end_page else total_pages)

            logger.info(
                f"Starting batch ingestion: {total_docs} documents, {total_pages} total pages"
            )
            logger.info(
                f"Processing pages {current_page} to {end_page} with batch size {page_size}"
            )

            total_processed = 0

            # Process each batch
            while current_page <= end_page:
                try:
                    logger.info(f"Processing page {current_page}/{end_page}")

                    # Load batch data from MongoDB
                    batch_data = self.source_db.load_collection_with_pagination(
                        self.source_collection,
                        page=current_page,
                        page_size=page_size,
                    )

                    if not batch_data:
                        logger.info("No more documents to process")
                        break

                    # Process the batch
                    processed_count = self._process_batch(batch_data)
                    total_processed += processed_count

                    logger.info(
                        f"Successfully processed {processed_count} documents in page {current_page}"
                    )
                    current_page += 1

                except Exception as batch_error:
                    logger.error(
                        f"Error processing batch on page {current_page}: {batch_error}"
                    )
                    raise

            logger.info(
                f"Batch ingestion completed. Total documents processed: {total_processed}"
            )

            return {
                "status": "success",
                "total_processed": total_processed,
                "pages_processed": current_page - start_page,
                "total_pages": total_pages,
            }

        except Exception as e:
            logger.error(f"Error in batch ingestion pipeline: {e}")
            raise

    def get_embedding(self, data: dict):
        return self._single_process(data)

    def get_progress(self):
        """
        Get the current progress of document processing.

        Returns:
            dict: Progress information including total documents and processed count
        """
        try:
            total_docs = self.source_db.count_documents(self.source_collection)
            processed_docs = self.vector_store.count_documents()

            return {
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "remaining_documents": total_docs - processed_docs,
                "progress_percentage": (
                    (processed_docs / total_docs * 100) if total_docs > 0 else 0
                ),
            }
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            raise
