"""
System Monitoring Routes

Provides health checks, statistics, and system status endpoints.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse, KBStatsResponse
from app.services import KnowledgeBaseService, SessionService
from app.utils.performance import get_tracker
from config import Config
from app import __version__

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["system"])

# Services (will be injected by main.py)
kb_service: KnowledgeBaseService = None
session_service: SessionService = None


def init_system_routes(kb: KnowledgeBaseService, session: SessionService):
    """
    Initialize system routes with required services.

    Args:
        kb: Knowledge base service instance
        session: Session service instance
    """
    global kb_service, session_service
    kb_service = kb
    session_service = session


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring system status.

    Returns comprehensive status including:
    - Overall system health
    - Application version
    - LLM provider configuration
    - Knowledge base status
    - Individual service statuses

    Returns:
        JSON response with health check data
    """
    logger.debug("Health check requested")

    # Check KB status
    kb_loaded = kb_service.is_loaded() if kb_service else False
    kb_chunks = len(kb_service.items) if kb_service and kb_loaded else 0

    # Determine overall status
    status = "healthy" if kb_loaded else "degraded"

    # Service statuses
    services = {
        "knowledge_base": "ok" if kb_loaded else "error",
        "session_manager": "ok" if session_service else "error",
        "llm_provider": Config.LLM_PROVIDER
    }

    response = HealthResponse(
        status=status,
        version=__version__,
        llm_provider=Config.LLM_PROVIDER,
        kb_loaded=kb_loaded,
        kb_chunks=kb_chunks,
        services=services
    )

    logger.info(f"Health check: {status}")
    return response


@router.get("/stats/kb", response_model=KBStatsResponse)
async def kb_stats():
    """
    Get statistics about the knowledge base.

    Provides detailed information about:
    - Number of chunks and sources
    - Embedding dimensions
    - Source document list
    - Average chunk size

    Returns:
        JSON response with KB statistics

    Raises:
        500 error if KB is not loaded
    """
    logger.debug("KB stats requested")

    if not kb_service or not kb_service.is_loaded():
        return JSONResponse(
            status_code=500,
            content={"error": "Knowledge base not loaded"}
        )

    stats = kb_service.get_statistics()

    response = KBStatsResponse(
        total_chunks=stats["total_chunks"],
        total_sources=stats["total_sources"],
        embedding_dimension=stats["embedding_dimension"],
        sources=stats["sources"],
        avg_chunk_length=stats.get("avg_chunk_length")
    )

    logger.info(f"KB stats: {stats['total_chunks']} chunks from {stats['total_sources']} sources")
    return response


@router.get("/stats/sessions")
async def session_stats():
    """
    Get statistics about active sessions.

    Provides information about:
    - Total number of sessions
    - Number of active conversations
    - Message counts
    - Average messages per session

    Returns:
        JSON response with session statistics
    """
    logger.debug("Session stats requested")

    if not session_service:
        return JSONResponse(
            status_code=500,
            content={"error": "Session service not available"}
        )

    stats = session_service.get_statistics()

    logger.info(
        f"Session stats: {stats['total_sessions']} sessions, "
        f"{stats['total_messages']} messages"
    )

    return JSONResponse(content=stats)


@router.get("/config")
async def get_config():
    """
    Get sanitized configuration information.

    Returns configuration without sensitive data (API keys, etc.)

    Returns:
        JSON response with safe configuration data
    """
    logger.debug("Config requested")

    safe_config = {
        "llm_provider": Config.LLM_PROVIDER,
        "max_turns": Config.MAX_TURNS,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "context_chunks": Config.CONTEXT_CHUNKS,
        "kb_embedding_dim": Config.KB_EMBEDDING_DIM,
        "log_level": Config.LOG_LEVEL,
        "version": __version__
    }

    # Add provider-specific safe config
    if Config.LLM_PROVIDER == "openai":
        safe_config["openai_chat_model"] = Config.OPENAI_CHAT_MODEL
        safe_config["openai_embedding_model"] = Config.OPENAI_EMBEDDING_MODEL
    elif Config.LLM_PROVIDER in ["ollama_local", "ollama_remote"]:
        safe_config["ollama_chat_model"] = Config.OLLAMA_CHAT_MODEL
        safe_config["ollama_embedding_model"] = Config.OLLAMA_EMBEDDING_MODEL
        safe_config["ollama_timeout"] = Config.OLLAMA_TIMEOUT

    return JSONResponse(content=safe_config)


@router.get("/stats/performance")
async def performance_stats():
    """
    Get performance timing statistics.

    Shows timing measurements for various operations to help
    identify bottlenecks.

    Returns:
        JSON response with performance statistics
    """
    logger.debug("Performance stats requested")

    tracker = get_tracker()
    stats = tracker.report(log_level="DEBUG")

    return JSONResponse(content={
        "performance_metrics": stats,
        "note": "Metrics are cumulative since server start. Times in milliseconds."
    })
