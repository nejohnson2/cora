"""
Chat API Routes

Handles chat message processing and conversation management.
"""

import logging
import time
from typing import List, Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import ChatIn, ChatResponse, SessionResetResponse
from app.services import KnowledgeBaseService, SessionService
from app.services.openai_kb_service import OpenAIKnowledgeBaseService
from app.utils.performance import timer, timer_context
from config import Config
from app.services.llm_client import create_llm_client
from openai import OpenAI

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize services (will be set by main.py)
kb_service: KnowledgeBaseService = None
session_service: SessionService = None
llm_client = None

# System prompt
SYSTEM_PROMPT = ""
SESSION_COOKIE = Config.SESSION_COOKIE_NAME


def init_chat_routes(kb: KnowledgeBaseService, session: SessionService):
    """
    Initialize chat routes with required services.

    This is called by main.py during app startup.

    Args:
        kb: Knowledge base service instance
        session: Session service instance
    """
    global kb_service, session_service, llm_client, SYSTEM_PROMPT
    kb_service = kb
    session_service = session
    llm_client = create_llm_client()

    # Load system prompt
    SYSTEM_PROMPT = Config.load_system_prompt() or _get_default_prompt()
    logger.info(f"Using {'custom' if Config.SYSTEM_PROMPT_FILE else 'default'} system prompt")


def _get_default_prompt() -> str:
    """Get the default system prompt if no custom prompt is configured."""
    return """
You are a classroom AI tutor for a specific course.

Core rules:
- Use a Socratic teaching method: ask guiding questions, request the student's reasoning, and help them plan steps.
- Do not provide final answers, complete solutions, or exact wording that could be submitted as the student's work.
- If the student asks for a direct answer, refuse briefly and switch to guidance through questions and hints.
- Keep responses focused, kind, and concise.
- When referencing course materials, only use the provided context. If the context does not contain the needed info, say so and ask clarifying questions.

Style:
- Start by clarifying the student's goal and what they have tried.
- Prefer short numbered questions and micro-hints.
""".strip()


def _get_session_id(request: Request) -> str:
    """
    Get or create session ID from request cookies.

    Args:
        request: FastAPI Request object

    Returns:
        Session ID string
    """
    session_id = request.cookies.get(SESSION_COOKIE)
    return session_service.get_or_create_session(session_id)


@timer("Chat: Generate OpenAI RAG response (total)", log_level="INFO")
def _generate_openai_rag_response(session_id: str, user_message: str, vector_store_id: str) -> str:
    """
    Generate a tutor response using OpenAI Responses API with file_search tool.

    This function uses OpenAI's Responses API with Vector Store file_search
    capability to automatically retrieve relevant context and generate responses.

    Note: Migrated from deprecated Assistants API to Responses API (Aug 2025).
    The Assistants API will be sunset on August 26, 2026.

    Args:
        session_id: Unique session identifier
        user_message: The student's question or message
        vector_store_id: OpenAI vector store ID containing the knowledge base

    Returns:
        The AI tutor's response text

    Raises:
        HTTPException: If OpenAI API call fails
    """
    logger.info(f"Processing OpenAI RAG request for session {session_id[:8]}...")

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        # Build conversation input with history
        # The Responses API uses 'input' (not 'messages') and 'instructions' (not system message)
        conversation_input: List[Dict[str, str]] = []

        # Add conversation history
        history = session_service.get_history(session_id)
        conversation_input.extend(history)
        logger.debug(f"Added {len(history)} historical messages to context")

        # Add current user message
        conversation_input.append({"role": "user", "content": user_message})

        # Use OpenAI Responses API with file_search
        # This is the replacement for the deprecated Assistants API
        logger.debug(f"Calling Responses API with file_search tool")

        # Call Responses API with file_search tool
        # Note: Using instructions for system prompt, input for conversation history
        response = client.responses.create(
            model=Config.OPENAI_CHAT_MODEL,
            input=conversation_input,
            instructions=SYSTEM_PROMPT,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id]
            }],
            store=False  # Don't store conversation on OpenAI's servers
        )

        # Extract the response text from the output
        # The Responses API provides output_text as a convenience method
        assistant_message = response.output_text

        if not assistant_message:
            raise RuntimeError("No text response found in OpenAI API output")

        logger.info(f"Generated OpenAI RAG response ({len(assistant_message)} chars)")

        return assistant_message

    except Exception as e:
        logger.error(f"OpenAI RAG API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@timer("Chat: Generate tutor response (total)", log_level="INFO")
def _generate_tutor_response(session_id: str, user_message: str) -> str:
    """
    Generate a tutor response using LLM with knowledge base context.

    Process:
    1. Retrieve relevant knowledge base chunks for the user's message
    2. Build message array with system prompt, context, history, and new message
    3. Call LLM API to generate response
    4. Return the assistant's message

    Args:
        session_id: Unique session identifier
        user_message: The student's question or message

    Returns:
        The AI tutor's response text

    Raises:
        HTTPException: If LLM API call fails
    """
    start_time = time.perf_counter()
    logger.info(f"Processing tutor request for session {session_id[:8]}...")
    logger.debug(f"User message: {user_message[:200]}...")

    try:
        # Retrieve relevant context from knowledge base
        context = kb_service.build_context(user_message)
        kb_time = time.perf_counter() - start_time
        logger.info(f"⏱️  KB context retrieval: {kb_time*1000:.2f}ms")

        # Check if we're using OpenAI RAG mode
        use_openai_rag = context.startswith("[OPENAI_VECTOR_STORE:")

        if use_openai_rag:
            # Extract vector store ID from marker
            vector_store_id = context.split(":")[1].rstrip("]")
            logger.info(f"Using OpenAI RAG mode with vector store: {vector_store_id}")

            # For OpenAI RAG, use OpenAI API directly with file_search tool
            llm_start = time.perf_counter()
            assistant_message = _generate_openai_rag_response(
                session_id, user_message, vector_store_id
            )
            llm_time = time.perf_counter() - llm_start
            logger.info(f"⏱️  OpenAI RAG call: {llm_time*1000:.2f}ms")
        else:
            # Use local RAG mode - build message array with context
            with timer_context("Chat: Build message array", log_level="DEBUG"):
                messages: List[Dict[str, str]] = []

                # Add system prompts
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
                messages.append({
                    "role": "system",
                    "content": f"Course context (use this, do not invent):\n\n{context}"
                })

                # Add conversation history for this session
                history = session_service.get_history(session_id)
                messages.extend(history)
                logger.debug(f"Added {len(history)} historical messages to context")

                # Add current user message
                messages.append({"role": "user", "content": user_message})

            # Call LLM API
            llm_start = time.perf_counter()
            logger.debug(f"Calling LLM API with {len(messages)} messages")

            with timer_context("Chat: LLM API call", log_level="INFO"):
                assistant_message = llm_client.chat_completion(messages)

            llm_time = time.perf_counter() - llm_start
            logger.info(f"⏱️  LLM API call: {llm_time*1000:.2f}ms")
            logger.info(f"Generated response ({len(assistant_message)} chars)")
            logger.debug(f"Assistant response: {assistant_message[:200]}...")

        total_time = time.perf_counter() - start_time
        logger.info(f"⏱️  Total request time: {total_time*1000:.2f}ms")

        return assistant_message

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_api(request: Request, payload: ChatIn):
    """
    Handle chat messages for the regular (non-logging) interface.

    Process:
    1. Get or create session ID
    2. Add user message to history
    3. Generate AI response with knowledge base context
    4. Add assistant message to history
    5. Return response (NO logging to disk)

    Args:
        request: FastAPI Request object
        payload: ChatIn model with the user's message

    Returns:
        JSON response with the AI tutor's reply
    """
    session_id = _get_session_id(request)
    logger.info(f"[/api/chat] Processing message for session {session_id[:8]}...")

    # Update in-memory history with user message
    session_service.append_message(session_id, "user", payload.message)

    # Generate tutor response
    assistant_msg = _generate_tutor_response(session_id, payload.message)

    # Update history with assistant response
    session_service.append_message(session_id, "assistant", assistant_msg)

    # Return response (note: NO logging to disk for this endpoint)
    response = JSONResponse({"reply": assistant_msg})
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    logger.info(f"[/api/chat] Completed request for session {session_id[:8]}...")
    return response


@router.post("/study/chat", response_model=ChatResponse)
async def study_chat_api(request: Request, payload: ChatIn):
    """
    Handle chat messages for the study (logging) interface.

    Process:
    1. Get or create session ID
    2. Add user message to history
    3. Generate AI response with knowledge base context
    4. Add assistant message to history
    5. LOG conversation to disk (key difference from /api/chat)
    6. Return response

    Args:
        request: FastAPI Request object
        payload: ChatIn model with the user's message

    Returns:
        JSON response with the AI tutor's reply
    """
    session_id = _get_session_id(request)
    logger.info(f"[/api/study/chat] Processing message for session {session_id[:8]}...")

    # Update in-memory history with user message
    session_service.append_message(session_id, "user", payload.message)

    # Generate tutor response
    assistant_msg = _generate_tutor_response(session_id, payload.message)

    # Update history with assistant response
    session_service.append_message(session_id, "assistant", assistant_msg)

    # IMPORTANT: Persist transcript for consented study participants
    session_service.log_conversation(
        session_id=session_id,
        user_msg=payload.message,
        assistant_msg=assistant_msg,
        consented=True
    )

    response = JSONResponse({"reply": assistant_msg})
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    logger.info(f"[/api/study/chat] Completed request for session {session_id[:8]}...")
    return response


@router.post("/reset", response_model=SessionResetResponse)
async def reset_api(request: Request):
    """
    Clear conversation history for a session (regular chat mode).

    Args:
        request: FastAPI Request object

    Returns:
        JSON response indicating success
    """
    session_id = _get_session_id(request)
    logger.info(f"[/api/reset] Resetting history for session {session_id[:8]}...")
    session_service.reset_session(session_id)

    response = JSONResponse({"ok": True, "message": "Session reset successfully"})
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


@router.post("/study/reset", response_model=SessionResetResponse)
async def study_reset_api(request: Request):
    """
    Clear conversation history for a session (study mode).

    Note: This only clears in-memory history. Past logged conversations
    remain in the JSONL file.

    Args:
        request: FastAPI Request object

    Returns:
        JSON response indicating success
    """
    session_id = _get_session_id(request)
    logger.info(f"[/api/study/reset] Resetting history for session {session_id[:8]}...")
    session_service.reset_session(session_id)

    response = JSONResponse({"ok": True, "message": "Session reset successfully"})
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response
