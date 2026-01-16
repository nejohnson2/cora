"""
OpenAI Knowledge Base Service

Handles knowledge base operations using OpenAI's Vector Stores API.
This service provides a thin wrapper that returns the vector store ID
for use with OpenAI's file_search tool in chat completions.
"""

import logging
from typing import Dict, Any
from openai import OpenAI

from config import Config
from app.utils.performance import timer

logger = logging.getLogger(__name__)


class OpenAIKnowledgeBaseService:
    """
    Service for managing knowledge base using OpenAI Vector Stores API.

    Unlike the local KB service, this doesn't perform search itself.
    Instead, it verifies the vector store exists and provides the
    vector_store_id for use with OpenAI's file_search tool.
    """

    def __init__(self, vector_store_id: str = None):
        """
        Initialize the OpenAI Knowledge Base Service.

        Args:
            vector_store_id: OpenAI vector store ID. If None, uses Config.OPENAI_VECTOR_STORE_ID.

        Raises:
            ValueError: If vector store ID or API key is not provided
        """
        self.vector_store_id = vector_store_id or Config.OPENAI_VECTOR_STORE_ID

        if not self.vector_store_id:
            raise ValueError(
                "OpenAI vector store ID is required. "
                "Run upload_to_openai.py to create a vector store."
            )

        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI RAG mode")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self._loaded = False
        self._file_count = 0

        # Verify vector store exists
        self._verify_vector_store()

    @timer("OpenAI KB: Verify vector store", log_level="INFO")
    def _verify_vector_store(self) -> None:
        """
        Verify that the vector store exists and is ready.

        Raises:
            ValueError: If vector store doesn't exist or is not ready
        """
        logger.info(f"Verifying OpenAI vector store: {self.vector_store_id}")

        try:
            vs = self.client.vector_stores.retrieve(self.vector_store_id)

            if vs.status == 'completed':
                self._file_count = vs.file_counts.completed
                logger.info(
                    f"✓ Vector store ready: {self._file_count} files indexed"
                )
                self._loaded = True
            elif vs.status == 'in_progress':
                self._file_count = vs.file_counts.completed
                logger.warning(
                    f"Vector store is still processing. "
                    f"{vs.file_counts.completed}/{vs.file_counts.total} files complete"
                )
                self._loaded = True  # Allow usage even if still processing
            elif vs.status == 'failed':
                raise ValueError("Vector store processing failed")
            else:
                raise ValueError(f"Vector store has unexpected status: {vs.status}")

        except Exception as e:
            logger.error(f"Failed to verify vector store: {e}")
            raise ValueError(
                f"Cannot access vector store '{self.vector_store_id}': {e}\n"
                f"Please check that the vector store ID is correct and you have access."
            )

    def build_context(self, user_message: str, k: int = None) -> str:
        """
        For OpenAI RAG mode, context building is handled by OpenAI's file_search tool.
        This method returns a marker string that signals to use the vector store.

        Args:
            user_message: The student's question (not used, for interface compatibility)
            k: Number of chunks (not used, for interface compatibility)

        Returns:
            Marker string indicating OpenAI RAG mode
        """
        if not self._loaded:
            raise ValueError("Vector store not loaded")

        # Return a special marker that signals to the chat route to use OpenAI's file_search
        # The actual context retrieval will happen within the OpenAI API call
        return f"[OPENAI_VECTOR_STORE:{self.vector_store_id}]"

    def get_vector_store_id(self) -> str:
        """
        Get the vector store ID for use with OpenAI's file_search tool.

        Returns:
            The vector store ID string
        """
        return self.vector_store_id

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing vector store statistics
        """
        if not self._loaded:
            return {
                "provider": "openai",
                "status": "not_loaded"
            }

        try:
            vs = self.client.vector_stores.retrieve(self.vector_store_id)

            return {
                "provider": "openai",
                "vector_store_id": self.vector_store_id,
                "status": vs.status,
                "total_files": vs.file_counts.total,
                "completed_files": vs.file_counts.completed,
                "in_progress_files": vs.file_counts.in_progress,
                "failed_files": vs.file_counts.failed,
                "created_at": vs.created_at,
                "name": vs.name
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "provider": "openai",
                "vector_store_id": self.vector_store_id,
                "error": str(e)
            }

    def is_loaded(self) -> bool:
        """
        Check if the vector store is ready.

        Returns:
            True if vector store is loaded and ready, False otherwise
        """
        return self._loaded
