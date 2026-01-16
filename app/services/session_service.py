"""
Session Management Service

Handles user session management including:
- Session creation and tracking
- Conversation history storage
- Session cleanup and expiration
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Literal
from collections import defaultdict

from config import Config

logger = logging.getLogger(__name__)

# Type hint for message roles
Role = Literal["user", "assistant", "system"]


class SessionService:
    """
    Service for managing user sessions and conversation history.

    Currently uses in-memory storage. Sessions are lost on server restart.
    In a production environment, this could be backed by Redis or a database.
    """

    def __init__(self):
        """Initialize the session service."""
        # In-memory storage: session_id -> list of messages
        self._history: Dict[str, List[Dict[str, str]]] = {}

        # Track session metadata (creation time, last activity)
        self._metadata: Dict[str, Dict[str, float]] = {}

        # Configuration
        self._max_turns = Config.MAX_TURNS
        self._log_dir = Config.LOG_DIR
        self._consent_log_path = Config.get_consent_log_path()

        # Ensure log directory exists
        os.makedirs(self._log_dir, exist_ok=True)

        logger.info("Session service initialized (in-memory storage)")

    def create_session_id(self) -> str:
        """
        Generate a new unique session ID.

        Returns:
            UUID-based session ID as a hex string
        """
        session_id = uuid.uuid4().hex
        self._metadata[session_id] = {
            "created_at": time.time(),
            "last_activity": time.time()
        }
        logger.info(f"Created new session: {session_id[:8]}...")
        return session_id

    def validate_session_id(self, session_id: Optional[str]) -> bool:
        """
        Validate a session ID.

        Args:
            session_id: The session ID to validate

        Returns:
            True if session ID is valid, False otherwise
        """
        if not session_id:
            return False

        if not isinstance(session_id, str):
            return False

        # Check minimum length for UUID hex
        if len(session_id) < 10:
            return False

        return True

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Existing session ID or None

        Returns:
            Valid session ID (either existing or newly created)
        """
        if self.validate_session_id(session_id) and session_id in self._metadata:
            # Update last activity
            self._metadata[session_id]["last_activity"] = time.time()
            logger.debug(f"Using existing session: {session_id[:8]}...")
            return session_id

        # Create new session
        return self.create_session_id()

    def append_message(
        self,
        session_id: str,
        role: Role,
        content: str
    ) -> None:
        """
        Add a message to the session's conversation history.

        Automatically trims history to keep only the most recent MAX_TURNS
        conversation pairs to manage memory usage.

        Args:
            session_id: Unique identifier for the session
            role: Message role ("user", "assistant", or "system")
            content: The message content
        """
        # Initialize history for new sessions
        if session_id not in self._history:
            self._history[session_id] = []
            logger.debug(f"Initialized history for session {session_id[:8]}...")

        # Append the new message
        self._history[session_id].append({
            "role": role,
            "content": content
        })

        # Update last activity
        if session_id in self._metadata:
            self._metadata[session_id]["last_activity"] = time.time()

        # Keep only the last MAX_TURNS*2 messages (user+assistant pairs)
        max_msgs = self._max_turns * 2
        if len(self._history[session_id]) > max_msgs:
            trimmed_count = len(self._history[session_id]) - max_msgs
            self._history[session_id] = self._history[session_id][-max_msgs:]
            logger.debug(
                f"Trimmed {trimmed_count} old messages from session {session_id[:8]}..."
            )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            List of message dictionaries, or empty list if session not found
        """
        history = self._history.get(session_id, [])
        logger.debug(f"Retrieved {len(history)} messages for session {session_id[:8]}...")
        return history

    def reset_session(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Preserves session metadata but removes all messages.

        Args:
            session_id: Unique identifier for the session
        """
        if session_id in self._history:
            msg_count = len(self._history[session_id])
            self._history.pop(session_id, None)
            logger.info(
                f"Reset history for session {session_id[:8]}... "
                f"(removed {msg_count} messages)"
            )

            # Update last activity
            if session_id in self._metadata:
                self._metadata[session_id]["last_activity"] = time.time()
        else:
            logger.debug(f"No history to reset for session {session_id[:8]}...")

    def delete_session(self, session_id: str) -> None:
        """
        Completely delete a session and its metadata.

        Args:
            session_id: Unique identifier for the session
        """
        self._history.pop(session_id, None)
        self._metadata.pop(session_id, None)
        logger.info(f"Deleted session {session_id[:8]}...")

    def log_conversation(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        consented: bool = False
    ) -> None:
        """
        Log a conversation turn to disk (if consented).

        Only logs when consented=True (i.e., in /study mode).

        Args:
            session_id: Unique session identifier
            user_msg: The student's message
            assistant_msg: The tutor's response
            consented: Whether user has consented to logging
        """
        if not consented:
            return

        event = {
            "ts_unix": time.time(),
            "session_id": session_id,
            "user": user_msg,
            "assistant": assistant_msg,
        }

        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(self._consent_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            logger.info(f"Logged conversation turn for session {session_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_session_count(self) -> int:
        """
        Get the total number of active sessions.

        Returns:
            Number of sessions currently in memory
        """
        return len(self._metadata)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, any]]:
        """
        Get metadata about a specific session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with session metadata, or None if not found
        """
        if session_id not in self._metadata:
            return None

        meta = self._metadata[session_id]
        history_length = len(self._history.get(session_id, []))

        return {
            "session_id": session_id,
            "created_at": meta["created_at"],
            "last_activity": meta["last_activity"],
            "message_count": history_length,
            "age_seconds": time.time() - meta["created_at"]
        }

    def cleanup_old_sessions(self, max_age_seconds: int = 86400) -> int:
        """
        Remove sessions that haven't been active recently.

        Args:
            max_age_seconds: Maximum age in seconds (default: 24 hours)

        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        sessions_to_remove = []

        for session_id, meta in self._metadata.items():
            age = current_time - meta["last_activity"]
            if age > max_age_seconds:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.delete_session(session_id)

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

        return len(sessions_to_remove)

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about session usage.

        Returns:
            Dictionary with session statistics
        """
        if not self._metadata:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "total_messages": 0,
                "avg_messages_per_session": 0
            }

        total_messages = sum(len(hist) for hist in self._history.values())
        avg_messages = total_messages / len(self._history) if self._history else 0

        return {
            "total_sessions": len(self._metadata),
            "active_sessions": len(self._history),
            "total_messages": total_messages,
            "avg_messages_per_session": round(avg_messages, 2)
        }
