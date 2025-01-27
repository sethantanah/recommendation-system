from sentence_transformers import SentenceTransformer


class SBERTModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=True).tolist()
