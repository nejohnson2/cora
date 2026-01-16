"""
Page Rendering Routes

Handles serving HTML pages for the chat interface.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Create router
router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def root():
    """
    Redirect root to /chat page.

    Returns:
        HTML redirect response
    """
    return HTMLResponse(
        content='<meta http-equiv="refresh" content="0; url=/chat">',
        status_code=200
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    Serve the regular chat interface (no conversation logging).

    Args:
        request: FastAPI Request object

    Returns:
        Rendered HTML page for the AI tutor chat interface
    """
    logger.info("Serving /chat page")
    return templates.TemplateResponse("chat.html", {"request": request})


@router.get("/study", response_class=HTMLResponse)
async def study_page(request: Request):
    """
    Serve the study chat interface (WITH conversation logging).

    Logs all conversations to JSONL file for research purposes.
    Users are informed via the UI that their chats will be stored.

    Args:
        request: FastAPI Request object

    Returns:
        Rendered HTML page for the study mode AI tutor interface
    """
    logger.info("Serving /study page")
    return templates.TemplateResponse("study.html", {"request": request})
