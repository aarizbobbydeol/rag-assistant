"""In-process conversation memory: a bounded LRU of short chat histories.

Sessions are deliberately kept in the process rather than in the vector store or
a database: they are small, they expire quickly, and losing them on a restart
costs a user nothing but the context of the current thread.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.models import Turn


@dataclass
class _Session:
    """Turns of one conversation plus the moment it was last used."""

    turns: list[Turn] = field(default_factory=list)
    touched_at: float = 0.0


class SessionStore:
    """Thread-safe LRU + TTL store of per-session chat turns.

    Recency order and last-touch order are kept identical - every read and every
    write moves its session to the end of the ``OrderedDict`` - so eviction and
    expiry share one ordering and ``purge_expired`` can stop at the first live
    session instead of scanning the whole map.

    A non-positive ``ttl_s`` disables expiry; the ``max_sessions`` cap still
    applies, which is what keeps the store bounded in that configuration.
    """

    def __init__(self, max_sessions: int = 1000, ttl_s: int = 21600) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_s = float(ttl_s)
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    # -- reads ---------------------------------------------------------- #
    def history(self, session_id: str | None, limit: int) -> list[Turn]:
        """Last ``limit`` turns of ``session_id``, oldest first.

        An unknown session, a missing id and a non-positive limit are all normal
        conditions for a stateless request, so they yield an empty list rather
        than an error.
        """
        if not session_id or limit <= 0:
            return []
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            self._touch_locked(session_id, session)
            # Copies: callers hand these to prompt builders that are free to
            # rewrite them, and a mutation must not reach the stored history.
            return [turn.model_copy() for turn in session.turns[-limit:]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- writes --------------------------------------------------------- #
    def append(self, session_id: str | None, turn: Turn) -> None:
        """Record one turn. A missing ``session_id`` means "do not remember"."""
        if not session_id:
            return
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session()
                self._sessions[session_id] = session
            session.turns.append(turn.model_copy())
            self._touch_locked(session_id, session)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def reset(self, session_id: str | None) -> bool:
        """Drop a conversation; ``True`` if there was one to drop."""
        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def purge_expired(self) -> int:
        """Evict every session idle for longer than the TTL; returns the count."""
        with self._lock:
            return self._purge_locked()

    # -- internals (caller holds the lock) ------------------------------- #
    def _touch_locked(self, session_id: str, session: _Session) -> None:
        session.touched_at = time.monotonic()
        self._sessions.move_to_end(session_id)

    def _purge_locked(self) -> int:
        if self.ttl_s <= 0:
            return 0
        cutoff = time.monotonic() - self.ttl_s
        removed = 0
        while self._sessions:
            session_id = next(iter(self._sessions))
            if self._sessions[session_id].touched_at > cutoff:
                break
            del self._sessions[session_id]
            removed += 1
        return removed
