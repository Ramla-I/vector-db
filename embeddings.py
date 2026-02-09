"""Embedding provider abstraction."""

from abc import ABC, abstractmethod
from typing import List

import config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    # Batch size to stay under OpenAI's 300k token limit
    BATCH_SIZE = 100

    def __init__(self):
        from openai import OpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL

    def embed(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching."""
        all_embeddings = []
        total_batches = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            batch_num = i // self.BATCH_SIZE + 1

            if progress_callback:
                progress_callback(batch_num, total_batches)

            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        return self.embed([text])[0]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get the configured embedding provider."""
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        return OpenAIEmbedding()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
