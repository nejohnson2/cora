"""
API Routes

Contains all HTTP endpoint handlers organized by functionality.
"""

from .pages import router as pages_router
from .chat import router as chat_router
from .system import router as system_router

__all__ = ['pages_router', 'chat_router', 'system_router']
