"""Where routines live: one SQLite row per user and trigger.

Not in the checkpointer's memory.db: that connection can hold a write transaction
open during a run, and a second writer on the same file would fail with "database
is locked". Not in Qdrant either: a trigger needs an exact match, not a similar one.
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass

from ai_companion.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Routine:
    trigger: str
    steps: list[str]


def normalize_trigger(text: str) -> str:
    """Reduce text to its bare words, so 'Guten Morgen, Bob!' equals 'guten morgen'."""
    return " ".join(word for word in re.findall(r"\w+", text.casefold()) if word != "bob")


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(settings.ROUTINES_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.ROUTINES_DB_PATH)
    try:
        with conn:  # commits on success, rolls back on an error
            conn.execute(
                "CREATE TABLE IF NOT EXISTS routines ("
                "session_id TEXT NOT NULL, trigger_key TEXT NOT NULL, trigger_text TEXT NOT NULL, "
                "steps TEXT NOT NULL, PRIMARY KEY (session_id, trigger_key))"
            )
            yield conn
    finally:
        conn.close()


def save(session_id: str, trigger: str, steps: list[str]) -> None:
    """Save a routine. A routine with the same trigger is replaced."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO routines VALUES (?, ?, ?, ?)",
            (session_id, normalize_trigger(trigger), trigger, json.dumps(steps, ensure_ascii=False)),
        )


def find(session_id: str, message: str) -> Routine | None:
    """The routine this message triggers, if any: the whole message must be the trigger."""
    key = normalize_trigger(message)
    if not key:
        return None
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT trigger_text, steps FROM routines WHERE session_id = ? AND trigger_key = ?",
                (session_id, key),
            ).fetchone()
    except (sqlite3.Error, OSError) as e:
        # This runs for every message: a broken file must not stop Bob from chatting.
        logger.warning(f"Could not read routines: {e}")
        return None
    return Routine(row[0], json.loads(row[1])) if row else None


def list_all(session_id: str) -> list[Routine]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT trigger_text, steps FROM routines WHERE session_id = ? ORDER BY trigger_key",
            (session_id,),
        ).fetchall()
    return [Routine(trigger, json.loads(steps)) for trigger, steps in rows]


def delete(session_id: str, trigger: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM routines WHERE session_id = ? AND trigger_key = ?",
            (session_id, normalize_trigger(trigger)),
        )
