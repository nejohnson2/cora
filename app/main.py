"""
AI Tutor Main Application

FastAPI application with improved architecture:
- Separated services (KB, sessions, LLM)
- Modular route handlers
- Template-based frontend
- Comprehensive error handling
- Health monitoring endpoints
"""

import logging
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import Config
from app.services import KnowledgeBaseService, OpenAIKnowledgeBaseService, SessionService
from app.routes import pages_router, chat_router, system_router
from app.routes.chat import init_chat_routes
from app.routes.system import init_system_routes
from app.utils.performance import timer_context
from app import __version__

# ===== Configure Logging =====
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.SERVER_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== Application Initialization =====
logger.info("=" * 60)
logger.info(f"AI Tutor Server v{__version__} initializing...")
logger.info("=" * 60)

# Display configuration
Config.display()

# ===== Create FastAPI App =====
app = FastAPI(
    title="AI Tutor",
    description="Intelligent tutoring system with knowledge base and LLM integration",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== Mount Static Files =====
# Serve CSS, JS, and other static assets
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Mounted static files from /static")
else:
    logger.warning("Static directory not found - static files will not be served")

# ===== Initialize Services =====
startup_start = time.perf_counter()
try:
    # Initialize Knowledge Base Service based on RAG_MODE
    with timer_context("Startup: Initialize Knowledge Base Service", log_level="INFO"):
        logger.info(f"Initializing Knowledge Base Service (mode: {Config.RAG_MODE})...")

        if Config.RAG_MODE == 'local':
            kb_service = KnowledgeBaseService()
            stats = kb_service.get_statistics()
            logger.info(f"✓ Local KB loaded: {stats['total_chunks']} chunks from {stats['total_sources']} sources")
        elif Config.RAG_MODE == 'openai':
            kb_service = OpenAIKnowledgeBaseService()
            stats = kb_service.get_statistics()
            logger.info(f"✓ OpenAI KB loaded: Vector Store {stats.get('vector_store_id', 'N/A')}")
            logger.info(f"  Status: {stats.get('status', 'unknown')}, Files: {stats.get('completed_files', 0)}")
        else:
            raise ValueError(f"Invalid RAG_MODE: {Config.RAG_MODE}")

    # Initialize Session Service
    with timer_context("Startup: Initialize Session Service", log_level="INFO"):
        logger.info("Initializing Session Service...")
        session_service = SessionService()
        logger.info("✓ Session service ready")

    # Initialize route handlers with services
    with timer_context("Startup: Initialize route handlers", log_level="INFO"):
        init_chat_routes(kb_service, session_service)
        init_system_routes(kb_service, session_service)
        logger.info("✓ Route handlers initialized")

    total_startup = time.perf_counter() - startup_start
    logger.info(f"⏱️  Total startup time: {total_startup*1000:.2f}ms")

except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    logger.error("Server will start but may not function properly")
    raise

# ===== Register Routers =====
app.include_router(pages_router)
app.include_router(chat_router)
app.include_router(system_router)

logger.info("✓ All routes registered")

# ===== Error Handlers =====

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions with user-friendly error messages.

    Args:
        request: The request that caused the error
        exc: The HTTP exception

    Returns:
        JSON error response
    """
    logger.error(f"HTTP {exc.status_code} error on {request.url.path}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors (e.g., invalid input).

    Args:
        request: The request that caused the error
        exc: The validation error

    Returns:
        JSON error response with validation details
    """
    logger.error(f"Validation error on {request.url.path}: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request data",
            "details": exc.errors(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected errors gracefully.

    Args:
        request: The request that caused the error
        exc: The exception

    Returns:
        JSON error response
    """
    logger.exception(f"Unexpected error on {request.url.path}: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """
    Run tasks on application startup.
    """
    logger.info("=" * 60)
    logger.info("AI Tutor Server started successfully!")
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info("  Pages:")
    logger.info("    - GET  /              (redirect to /chat)")
    logger.info("    - GET  /chat          (regular chat, no logging)")
    logger.info("    - GET  /study         (study mode, with logging)")
    logger.info("  Chat API:")
    logger.info("    - POST /api/chat      (send message)")
    logger.info("    - POST /api/study/chat")
    logger.info("    - POST /api/reset     (reset session)")
    logger.info("    - POST /api/study/reset")
    logger.info("  System:")
    logger.info("    - GET  /health        (health check)")
    logger.info("    - GET  /stats/kb      (KB statistics)")
    logger.info("    - GET  /stats/sessions (session stats)")
    logger.info("    - GET  /config        (configuration)")
    logger.info("  Docs:")
    logger.info("    - GET  /docs          (Swagger UI)")
    logger.info("    - GET  /redoc         (ReDoc)")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run cleanup tasks on application shutdown.
    """
    logger.info("AI Tutor Server shutting down...")

    # Log final statistics
    if session_service:
        stats = session_service.get_statistics()
        logger.info(f"Final session stats: {stats}")

    logger.info("Server shutdown complete")


# ===== Middleware (Optional) =====

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests (optional middleware).

    Args:
        request: The incoming request
        call_next: The next middleware/handler

    Returns:
        The response from the next handler
    """
    # Log request
    logger.debug(f"{request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response status
    logger.debug(f"{request.method} {request.url.path} -> {response.status_code}")

    return response


# ===== Export for ASGI Servers =====
# This allows running with: uvicorn app.main:app

if __name__ == "__main__":
    # This block is only for direct execution (not recommended in production)
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
