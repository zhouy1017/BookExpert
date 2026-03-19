"""
Local Embedding, Summary, Score & Feedback Caches
===================================================
All backed by a single SQLite database at db/cache.db.

Classes
-------
EmbeddingCache  – stores per-chunk float vectors keyed by (file_hash, chunk_index)
SummaryCache    – stores text results (summaries, reviews) keyed by string key
FeedbackCache   – stores user review feedback and scoring preferences per book
"""

import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = "db/cache.db"


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """
    Stores embedding vectors keyed by (file_hash, chunk_index).

    Schema: embedding_cache(file_hash TEXT, chunk_index INTEGER, vector TEXT)
    """
    def __init__(self, db_path: str = _DB_PATH):
        self._lock = threading.Lock()
        self._conn = _get_conn(db_path)
        self._init()

    def _init(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                file_hash   TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                vector      TEXT    NOT NULL,
                PRIMARY KEY (file_hash, chunk_index)
            )
        """)
        self._conn.commit()

    def get(self, file_hash: str, chunk_index: int) -> Optional[List[float]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embedding_cache WHERE file_hash=? AND chunk_index=?",
                (file_hash, chunk_index)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, file_hash: str, chunk_index: int, vector: List[float]):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (file_hash, chunk_index, vector) VALUES (?,?,?)",
                (file_hash, chunk_index, json.dumps(vector))
            )
            self._conn.commit()

    def has_all(self, file_hash: str, total_chunks: int) -> bool:
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embedding_cache WHERE file_hash=?", (file_hash,)
            ).fetchone()[0]
        return count >= total_chunks


# ---------------------------------------------------------------------------
# Summary / Review Cache
# ---------------------------------------------------------------------------
class SummaryCache:
    """
    Stores arbitrary text results (summaries, review JSON, locked scores) by string key.

    Key conventions (all keyed by file_hash or doc-derived hash):
      summary_{hash}   – full-book summary text
      review_{hash}    – full review JSON string
      scores_{hash}    – locked score JSON {overall_score, dimensions}
    """
    def __init__(self, db_path: str = _DB_PATH):
        self._lock = threading.Lock()
        self._conn = _get_conn(db_path)
        self._init()

    def _init(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_cache (
                cache_key TEXT PRIMARY KEY,
                value     TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM summary_cache WHERE cache_key=?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO summary_cache (cache_key, value) VALUES (?,?)",
                (key, value)
            )
            self._conn.commit()

    # -- Convenience: locked scores -------------------------------------------
    def get_scores(self, file_hash: str) -> Optional[Dict]:
        raw = self.get(f"scores_{file_hash}")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    def save_scores(self, file_hash: str, scores: Dict):
        self.put(f"scores_{file_hash}", json.dumps(scores, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Feedback Cache
# ---------------------------------------------------------------------------
class FeedbackCache:
    """
    Stores user review feedback and scoring preferences per book.

    Schema: user_review_feedback(
        book_hash TEXT PK, score_overrides TEXT, extra_strengths TEXT,
        extra_weaknesses TEXT, user_comments TEXT, scoring_prefs TEXT, updated_at TEXT
    )
    """
    def __init__(self, db_path: str = _DB_PATH):
        self._lock = threading.Lock()
        self._conn = _get_conn(db_path)
        self._init()

    def _init(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_review_feedback (
                book_hash        TEXT PRIMARY KEY,
                score_overrides  TEXT,
                extra_strengths  TEXT,
                extra_weaknesses TEXT,
                user_comments    TEXT,
                scoring_prefs    TEXT,
                updated_at       TEXT
            )
        """)
        self._conn.commit()

    def get_feedback(self, book_hash: str) -> Optional[Dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT score_overrides, extra_strengths, extra_weaknesses, "
                "user_comments, scoring_prefs FROM user_review_feedback WHERE book_hash=?",
                (book_hash,)
            ).fetchone()
        if not row:
            return None
        return {
            "score_overrides":  json.loads(row[0]) if row[0] else {},
            "extra_strengths":  json.loads(row[1]) if row[1] else [],
            "extra_weaknesses": json.loads(row[2]) if row[2] else [],
            "user_comments":    row[3] or "",
            "scoring_prefs":    row[4] or "",
        }

    def save_feedback(self, book_hash: str, data: Dict):
        from datetime import datetime
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO user_review_feedback
                    (book_hash, score_overrides, extra_strengths, extra_weaknesses,
                     user_comments, scoring_prefs, updated_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                book_hash,
                json.dumps(data.get("score_overrides", {}),  ensure_ascii=False),
                json.dumps(data.get("extra_strengths", []),  ensure_ascii=False),
                json.dumps(data.get("extra_weaknesses", []), ensure_ascii=False),
                data.get("user_comments", ""),
                data.get("scoring_prefs", ""),
                datetime.utcnow().isoformat(),
            ))
            self._conn.commit()
