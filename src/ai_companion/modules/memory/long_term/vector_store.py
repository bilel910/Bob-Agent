import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from ai_companion.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    # Multilingual model: German memories are embedded in the same space as German
    # queries. The English-only all-MiniLM-L6-v2 used previously embedded German
    # poorly. Both models output 384 dimensions, so the collection schema is unchanged.
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    # Bumped to _v2 because vectors written by the old English model live in a
    # different semantic space and carry no session_id, so they cannot be reused.
    # The old "long_term_memory" collection is left untouched and simply goes unused.
    COLLECTION_NAME = "long_term_memory_v2"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._validate_env_vars()
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            self._initialized = True

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists."""
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )
        # Indexed so that per-session filtering stays fast as the collection grows.
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def _session_filter(session_id: str) -> Filter:
        """Build a filter restricting results to a single user's memories."""
        return Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])

    def find_similar_memory(self, text: str, session_id: str) -> Optional[Memory]:
        """Find if a similar memory already exists for this user.

        Args:
            text: The text to search for
            session_id: The user/session whose memories to search

        Returns:
            Optional Memory if a similar one is found
        """
        results = self.search_memories(text, k=1, session_id=session_id)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory. Must contain a
                "session_id" so the memory stays scoped to the user it came from.
        """
        if not self._collection_exists():
            self._create_collection()

        session_id = metadata.get("session_id")
        if not session_id:
            raise ValueError("Cannot store a memory without a session_id")

        # Check if a similar memory exists for this same user
        similar_memory = self.find_similar_memory(text, session_id)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, k: int = 5, *, session_id: str) -> List[Memory]:
        """Search for similar memories belonging to a single user.

        Args:
            query: Text to search for
            k: Number of results to return
            session_id: The user/session whose memories to search. Required, so that
                one user's memories can never surface in another user's conversation.

        Returns:
            List of Memory objects
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=self._session_filter(session_id),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    return VectorStore()
