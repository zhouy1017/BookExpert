"""
Local Embedding & Summary Cache
================================
Avoids redundant Google/DeepSeek API calls by persisting computed results
to a local SQLite database keyed by SHA-256 of the source file bytes.

Classes
-------
EmbeddingCache  – stores per-chunk float vectors
SummaryCache    – stores full-text results (summaries, reviews) by key
"""

import json
import logging
import sqlite3
import threading
from typing import List, Optional

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

    Schema
    ------
    embedding_cache(file_hash TEXT, chunk_index INTEGER, vector TEXT,
                    PRIMARY KEY (file_hash, chunk_index))
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
        """Return cached vector or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embedding_cache WHERE file_hash=? AND chunk_index=?",
                (file_hash, chunk_index)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, file_hash: str, chunk_index: int, vector: List[float]):
        """Persist a vector."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (file_hash, chunk_index, vector) VALUES (?,?,?)",
                (file_hash, chunk_index, json.dumps(vector))
            )
            self._conn.commit()

    def has_all(self, file_hash: str, total_chunks: int) -> bool:
        """Returns True when every chunk index 0..total_chunks-1 is cached."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embedding_cache WHERE file_hash=?",
                (file_hash,)
            ).fetchone()[0]
        return count >= total_chunks

    def get_all(self, file_hash: str, total_chunks: int) -> Optional[List[List[float]]]:
        """Returns ordered list of all vectors if fully cached, else None."""
        if not self.has_all(file_hash, total_chunks):
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_index, vector FROM embedding_cache WHERE file_hash=? ORDER BY chunk_index",
                (file_hash,)
            ).fetchall()
        if len(rows) < total_chunks:
            return None
        return [json.loads(r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Summary / Review Cache
# ---------------------------------------------------------------------------
class SummaryCache:
    """
    Stores arbitrary text results (summaries, reviews) keyed by a string key.

    Key conventions:
      summary_{file_hash}   – full-book summary
      review_{file_hash}    – book review JSON string
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
