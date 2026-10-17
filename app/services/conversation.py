"""
app/services/conversation.py — In-memory conversation history management.

NOTE: History is stored in-process memory and is NOT persistent across server
restarts. This is intentional for simplicity. For durable history, add a Redis
or SQLite backend. Session IDs isolate conversations; there are no user accounts.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias: a turn is {"role": "user"|"assistant", "content": str, "timestamp": float}
Turn = Dict[str, str]

# Global in-memory storage: session_id → list of turns
_sessions: Dict[str, List[Turn]] = defaultdict(list)

# Maximum number of turns (user + assistant pairs) to keep per session
DEFAULT_MAX_TURNS = 20


def add_turn(session_id: str, role: str, content: str, max_turns: Optional[int] = None) -> None:
    """
    Append a turn to the session history.

    Args:
        session_id: Unique identifier for the conversation session.
        role: "user" or "assistant".
        content: The message content.
        max_turns: Maximum number of individual messages to retain (sliding window).
                   Defaults to DEFAULT_MAX_TURNS * 2 (each turn has user + assistant).
    """
    if not session_id or not session_id.strip():
        return

    limit = (max_turns or DEFAULT_MAX_TURNS) * 2  # pairs × 2 messages
    history = _sessions[session_id]
    history.append({
        "role": role,
        "content": content,
        "timestamp": str(time.time()),
    })

    # Sliding window: drop oldest messages
    if len(history) > limit:
        _sessions[session_id] = history[-limit:]


def get_history(session_id: str) -> List[Turn]:
    """
    Return the conversation history for a session.

    Returns an empty list for unknown or new sessions.
    """
    return list(_sessions.get(session_id, []))


def get_history_for_prompt(session_id: str, max_turns: int = 6) -> List[Dict[str, str]]:
    """
    Return the last *max_turns* turns as simple {"role", "content"} dicts
    suitable for inclusion in a generation prompt.
    """
    history = get_history(session_id)
    # Keep last max_turns messages
    recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
    return [{"role": t["role"], "content": t["content"]} for t in recent]


def clear_session(session_id: str) -> bool:
    """
    Clear all history for a session.

    Returns True if the session existed and was cleared, False otherwise.
    """
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info("Cleared session: %s", session_id)
        return True
    return False


def get_session_count() -> int:
    """Return the number of active in-memory sessions."""
    return len(_sessions)


def list_sessions() -> List[str]:
    """Return all known session IDs (for diagnostics only)."""
    return list(_sessions.keys())
