from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


class SBERTModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list) -> list:
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        return self.project_embeddings(embeddings)
    
    def project_embeddings(self, embedding):
        # Original embedding size (e.g., 384 for all-MiniLM-L6-v2)
        original_embedding_size = 384
        target_embedding_size = 1024

        # Projection layer
        projection_layer = nn.Linear(original_embedding_size, target_embedding_size)

        # Example embedding (from SBERT)
        embedding = torch.randn(1, original_embedding_size)  # Simulated embedding

        # Project to 1024 dimensions
        expanded_embedding = projection_layer(embedding)  # Shape: (1, 1024)

        # Convert to Python list
        expanded_embedding_list = expanded_embedding.detach().numpy().tolist()
        return expanded_embedding_list
    
