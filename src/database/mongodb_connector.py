from pymongo import MongoClient
from pymongo.server_api import ServerApi
import logging
from typing import List, Dict, Optional

# Set up logging
from src.config.logging import setup_logger

logger = setup_logger(__name__, debug_level=logging.DEBUG)


class MongoDBConnector:
    def __init__(self, password: str, database_name: str):
        """
        Initializes the MongoDBConnector with the connection URI and database name.

        :param password: MongoDB password (sensitive, not logged).
        :param database_name: Name of the database to connect to.
        """
        self.uri = f"mongodb+srv://kanddle:{password}@recommendationcluster.4559f.mongodb.net/?retryWrites=true&w=majority&appName=RecommendationCluster"
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.database = None
        self.connect()

    def connect(self):
        """
        Establishes a connection to the MongoDB database.
        """
        try:
            self.client = MongoClient(self.uri, server_api=ServerApi("1"))
            self.database = self.client[self.database_name]
            logger.info(f"Connected to database: {self.database_name}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    def test_db(self):
        """
        Tests the MongoDB connection by sending a ping.
        """
        try:
            self.client.admin.command("ping")
            logger.info(
                "Pinged your deployment. You successfully connected to MongoDB!"
            )
        except Exception as e:
            logger.error(f"Error pinging MongoDB: {e}")
            raise

    def load_collection(self, collection_name: str) -> List[Dict]:
        """
        Loads a collection from the connected database.

        :param collection_name: Name of the collection to load.
        :return: A list of documents in the collection.
        """
        if self.database is None:
            logger.error(
                "Database connection is not established. Call connect() first."
            )
            raise Exception("Database connection is not established.")

        try:
            collection = self.database[collection_name]
            data = list(collection.find())
            logger.debug(
                f"Loaded {len(data)} documents from collection: {collection_name}"
            )
            return data
        except Exception as e:
            logger.error(f"Error loading collection '{collection_name}': {e}")
            raise

    def get_model(self, collection_name: str, filter: dict) -> Dict:
        """
        Loads a collection from the connected database.

        :param collection_name: Name of the collection to load.
        :param model_name: Name of the model
        :return: A list of documents in the collection.
        """
        if self.database is None:
            logger.error(
                "Database connection is not established. Call connect() first."
            )
            raise Exception("Database connection is not established.")

        try:
            collection = self.database[collection_name]
            data = list(collection.find(filter))
            logger.debug(
                f"Loaded {len(data)} documents from collection: {collection_name}"
            )

            if data:
                return data[0]
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading collection '{collection_name}': {e}")
            raise

    def get_all_with_pagination(
        self, collection_name: str, page: int, page_size: int
    ) -> List[Dict]:
        """
        Fetches all documents from a collection with pagination.

        :param collection_name: Name of the collection.
        :param page: The page number (1-based index).
        :param page_size: The number of documents per page.
        :return: A list of documents for the specified page.
        """
        if self.database is None:
            logger.error(
                "Database connection is not established. Call connect() first."
            )
            raise Exception("Database connection is not established.")

        try:
            collection = self.database[collection_name]
            skip = (page - 1) * page_size
            documents = list(collection.find().skip(skip).limit(page_size))
            logger.debug(
                f"Fetched {len(documents)} documents from page {page} (size {page_size})"
            )
            return documents
        except Exception as e:
            logger.error(f"Error fetching documents with pagination: {e}")
            raise

    def insert_to_collection(self, collection_name: str, data_list: List[Dict]) -> List:
        """
        Inserts a list of documents into a MongoDB collection.

        :param collection_name: Name of the collection to insert data into.
        :param data_list: List of documents to insert.
        :return: List of inserted document IDs.
        """
        if self.database is None:
            logger.error(
                "Database connection is not established. Call connect() first."
            )
            raise Exception("Database connection is not established.")

        if not isinstance(data_list, list):
            logger.error("data_list must be a list of documents")
            raise TypeError("data_list must be a list of documents")

        if not data_list:
            logger.error("data_list cannot be empty")
            raise ValueError("data_list cannot be empty")

        try:
            result = self.database[collection_name].insert_many(data_list)
            logger.info(
                f"Inserted {len(result.inserted_ids)} documents into collection: {collection_name}"
            )
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
