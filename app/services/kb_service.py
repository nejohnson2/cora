"""
Knowledge Base Service with Performance Monitoring

Handles all knowledge base operations including:
- Loading and indexing documents
- Semantic search
- Embedding generation for queries

With integrated performance timing for identifying bottlenecks.
"""

import json
import logging
import time
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config
from app.utils.performance import timer, timer_context

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """
    Service for managing the knowledge base and performing semantic search.

    The KB uses BAAI/bge-base-en-v1.5 for embeddings, which produces
    768-dimensional vectors optimized for retrieval tasks.
    """

    def __init__(self, kb_index_file: str = None):
        """
        Initialize the Knowledge Base Service.

        Args:
            kb_index_file: Path to the knowledge base index JSON file.
                          If None, uses Config.KB_INDEX_FILE.

        Raises:
            FileNotFoundError: If the KB index file doesn't exist
            ValueError: If the KB file is malformed
        """
        self.kb_index_file = kb_index_file or Config.KB_INDEX_FILE
        self.items: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.embedding_model: SentenceTransformer = None
        self._loaded = False

        # Load the knowledge base
        self._load_kb()

        # Initialize the query embedding model
        self._init_embedding_model()

    @timer("KB: Load index file", log_level="INFO")
    def _load_kb(self) -> None:
        """
        Load the pre-built knowledge base index from disk.

        The index contains text chunks and their embeddings computed by build_kb.py.

        Raises:
            FileNotFoundError: If KB file doesn't exist
            ValueError: If KB file is malformed
        """
        logger.info(f"Loading knowledge base from {self.kb_index_file}")

        kb_path = Path(self.kb_index_file)
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base file '{self.kb_index_file}' not found. "
                f"Run build_kb.py first to create it."
            )

        try:
            with timer_context("KB: Read JSON file"):
                with open(self.kb_index_file, "r", encoding="utf-8") as f:
                    kb_data = json.load(f)

            with timer_context("KB: Extract items and embeddings"):
                self.items = kb_data.get("items", [])
                embeddings_list = kb_data.get("embeddings", [])

                if not self.items or not embeddings_list:
                    raise ValueError("KB file is empty or malformed")

            # Convert embeddings to numpy array
            with timer_context("KB: Convert embeddings to numpy array"):
                self.embeddings = np.array(embeddings_list, dtype=np.float32)

            logger.info(
                f"Successfully loaded KB: {len(self.items)} chunks, "
                f"embedding dimension: {self.embeddings.shape[1]}"
            )
            self._loaded = True

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse KB file: {e}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise

    @timer("KB: Initialize embedding model", log_level="INFO")
    def _init_embedding_model(self) -> None:
        """
        Initialize the sentence transformer model for query embeddings.

        Uses the same BAAI/bge-base-en-v1.5 model that was used to build
        the KB, ensuring embedding compatibility.
        """
        logger.info("Loading BAAI/bge-base-en-v1.5 embedding model for queries...")
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        logger.info("Query embedding model loaded successfully")

    @timer("KB: Generate query embedding", log_level="DEBUG")
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for a user query.

        According to BAAI documentation, queries should be prefixed with
        an instruction for better retrieval performance.

        Args:
            text: The user's query text

        Returns:
            numpy array of the embedding vector (768 dimensions)

        Raises:
            RuntimeError: If embedding generation fails
        """
        logger.debug(f"Generating embedding for query: {text[:100]}...")

        try:
            # Add instruction prefix for better retrieval
            instruction = "Represent this sentence for searching relevant passages:"
            query_with_instruction = instruction + text

            # Generate embedding with normalization for cosine similarity
            embedding = self.embedding_model.encode(
                query_with_instruction,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")

    @timer("KB: Semantic search", log_level="DEBUG")
    def semantic_search(
        self,
        query_vec: np.ndarray,
        k: int = 6
    ) -> List[Tuple[int, float]]:
        """
        Find the top-k most similar KB chunks using cosine similarity.

        Args:
            query_vec: The embedding vector for the user's query
            k: Number of top results to return (default: 6)

        Returns:
            List of tuples (chunk_index, similarity_score) sorted by
            descending similarity

        Raises:
            ValueError: If KB is not loaded or k is invalid
        """
        if not self._loaded:
            raise ValueError("Knowledge base not loaded")

        if k <= 0 or k > len(self.items):
            raise ValueError(f"k must be between 1 and {len(self.items)}")

        # Normalize query vector
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        # Normalize all KB embeddings
        M = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)

        # Compute cosine similarities via dot product
        sims = M @ q

        # Get indices of top-k highest similarities
        idx = np.argsort(-sims)[:k]

        results = [(int(i), float(sims[i])) for i in idx]
        logger.debug(
            f"Top {k} results: scores range from {results[0][1]:.3f} to {results[-1][1]:.3f}"
        )

        return results

    @timer("KB: Build context (total)", log_level="INFO")
    def build_context(self, user_message: str, k: int = None) -> str:
        """
        Build context from the KB for the AI tutor.

        Retrieves the most relevant chunks based on semantic similarity
        to the user's message.

        Args:
            user_message: The student's question or message
            k: Number of relevant chunks to retrieve (default: from config)

        Returns:
            Formatted string containing the top-k relevant KB chunks
            with metadata (source file, chunk ID, similarity score)
        """
        if k is None:
            k = Config.CONTEXT_CHUNKS

        logger.debug(f"Building context for message: {user_message[:100]}...")

        # Generate embedding for the query
        qvec = self.embed_query(user_message)

        # Find most similar chunks
        top_results = self.semantic_search(qvec, k=k)

        # Format results with metadata
        with timer_context("KB: Format context results", log_level="DEBUG"):
            blocks = []
            for i, score in top_results:
                item = self.items[i]
                blocks.append(
                    f"[Source: {item['source']} | chunk {item['chunk_id']} | score {score:.3f}]\n"
                    f"{item['text']}"
                )

            context = "\n\n---\n\n".join(blocks)
            logger.debug(f"Built context with {len(blocks)} chunks, total length: {len(context)} chars")

        return context

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary containing KB statistics including:
            - total_chunks: Number of text chunks
            - total_sources: Number of unique source documents
            - embedding_dimension: Dimension of embedding vectors
            - sources: List of source document names
            - avg_chunk_length: Average chunk length in characters
        """
        if not self._loaded:
            return {
                "total_chunks": 0,
                "total_sources": 0,
                "embedding_dimension": 0,
                "sources": [],
                "avg_chunk_length": 0
            }

        # Extract unique sources
        sources = list(set(item['source'] for item in self.items))

        # Calculate average chunk length
        chunk_lengths = [len(item['text']) for item in self.items]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

        return {
            "total_chunks": len(self.items),
            "total_sources": len(sources),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": sorted(sources),
            "avg_chunk_length": round(avg_length, 2)
        }

    def is_loaded(self) -> bool:
        """
        Check if the knowledge base is successfully loaded.

        Returns:
            True if KB is loaded and ready, False otherwise
        """
        return self._loaded and len(self.items) > 0
