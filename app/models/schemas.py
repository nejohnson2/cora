"""
Pydantic Models and Schemas

Defines the data models used for API requests and responses.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class ChatIn(BaseModel):
    """
    Schema for incoming chat messages.

    Attributes:
        message: The user's message text
    """
    message: str = Field(..., min_length=1, max_length=5000, description="User's message")


class ChatResponse(BaseModel):
    """
    Schema for chat response.

    Attributes:
        reply: The AI tutor's response
    """
    reply: str = Field(..., description="AI tutor's response")


class HealthResponse(BaseModel):
    """
    Schema for health check endpoint response.

    Attributes:
        status: Overall system status
        version: Application version
        llm_provider: Configured LLM provider
        kb_loaded: Whether knowledge base is loaded
        kb_chunks: Number of chunks in knowledge base
        services: Status of individual services
    """
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Application version")
    llm_provider: str = Field(..., description="LLM provider in use")
    kb_loaded: bool = Field(..., description="Knowledge base loaded status")
    kb_chunks: int = Field(..., description="Number of KB chunks")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")


class KBStatsResponse(BaseModel):
    """
    Schema for knowledge base statistics response.

    Attributes:
        total_chunks: Total number of text chunks
        total_sources: Number of unique source documents
        embedding_dimension: Dimension of embedding vectors
        sources: List of source document names
        avg_chunk_length: Average length of text chunks in characters
    """
    total_chunks: int = Field(..., description="Total number of chunks")
    total_sources: int = Field(..., description="Number of unique sources")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    avg_chunk_length: Optional[float] = Field(None, description="Average chunk length")


class SessionResetResponse(BaseModel):
    """
    Schema for session reset response.

    Attributes:
        ok: Whether the reset was successful
        message: Optional status message
    """
    ok: bool = Field(True, description="Success status")
    message: Optional[str] = Field(None, description="Status message")
