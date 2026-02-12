"""ChromaDB operations for vector storage and retrieval."""

import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

import config
from embeddings import get_embedding_provider


class VectorStore:
    """ChromaDB wrapper for vector storage and retrieval."""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.db_path = config.get_db_path(db_name)
        self._client = None
        self._collection = None
        self._embedding_provider = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the documents collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def embedding_provider(self):
        """Lazy-load embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    def add_documents(self, chunks: List[Dict[str, Any]], progress_callback=None) -> int:
        """Add document chunks to the collection."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_provider.embed(texts, progress_callback)

        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        if n_results is None:
            n_results = config.TOP_K_RESULTS

        query_embedding = self.embedding_provider.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                distance = results["distances"][0][i]
                score = 1 - distance

                formatted_results.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i],
                        "score": score,
                    }
                )

        return formatted_results

    def _apply_keyword_boost(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Boost results that contain exact query keywords.

        Extracts significant terms from query (register names, technical terms)
        and boosts results containing exact matches. Uses word boundaries to
        avoid partial matches (e.g. AFIO_MAPR vs AFIO_MAPR2).

        Boost tiers: +0.20 for REGISTER DEFINITION match, +0.10 for KEY term
        match, +0.05 for body text match. Scores are NOT capped at 1.0 to
        preserve differentiation between boosted results.
        """

        # Extract potential register/technical terms from query
        # Matches: AFIO_MAPR2, GPIO_CRL, TIM1_CH1, etc.
        term_pattern = r'\b([A-Z]{2,}[0-9]*_[A-Z0-9_]+)\b'
        query_terms = re.findall(term_pattern, query.upper())

        if not query_terms:
            return results

        boosted = []
        for result in results:
            text_upper = result["text"].upper()
            boost = 0.0

            for term in query_terms:
                # Use word boundary regex to avoid partial matches
                # This prevents AFIO_MAPR from matching when searching for AFIO_MAPR2
                exact_pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(exact_pattern, text_upper):
                    # Stronger boost for exact match in REGISTER DEFINITION line
                    if re.search(r'REGISTER DEFINITION:\s*' + re.escape(term) + r'\b', text_upper):
                        boost += 0.20
                    # Medium boost for exact match in KEY terms
                    elif "[KEY:" in result["text"] and re.search(exact_pattern, text_upper):
                        boost += 0.10
                    # Small boost for any exact match
                    else:
                        boost += 0.05

            boosted_result = result.copy()
            # Don't cap at 1.0 - allow boost to differentiate similar scores
            boosted_result["score"] = result["score"] + boost
            boosted_result["keyword_boost"] = boost
            boosted.append(boosted_result)

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x["score"], reverse=True)
        return boosted

    def list_documents(self) -> List[str]:
        """List all unique document sources in the collection."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in results["metadatas"]:
            if "source" in metadata:
                sources.add(metadata["source"])
        return sorted(sources)

    def delete_document(self, source_name: str) -> int:
        """Delete all chunks from a specific document."""
        results = self.collection.get(
            where={"source": source_name}, include=["metadatas"]
        )
        ids_to_delete = results["ids"]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "documents": self.list_documents(),
        }


def create_database(db_name: str) -> bool:
    """Create a new database."""
    config.ensure_databases_dir()
    db_path = config.get_db_path(db_name)

    if db_path.exists():
        return False

    # Initialize the database by creating a VectorStore
    store = VectorStore(db_name)
    _ = store.collection  # This creates the collection
    return True


def list_databases() -> List[str]:
    """List all available databases."""
    config.ensure_databases_dir()
    databases = []

    for path in config.DATABASES_DIR.iterdir():
        if path.is_dir():
            databases.append(path.name)

    return sorted(databases)


def delete_database(db_name: str) -> bool:
    """Delete a database."""
    db_path = config.get_db_path(db_name)

    if not db_path.exists():
        return False

    shutil.rmtree(db_path)
    return True


def database_exists(db_name: str) -> bool:
    """Check if a database exists."""
    return config.get_db_path(db_name).exists()
