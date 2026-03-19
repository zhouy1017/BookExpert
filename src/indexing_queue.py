"""
Background Indexing Queue
==========================
Provides a module-level singleton worker thread and job queue so that
document indexing continues in the background regardless of which Streamlit
tab or conversation the user is viewing.

Usage
-----
    from src.indexing_queue import enqueue, get_progress, is_idle, is_doc_indexed

    enqueue(doc_name, chunks, file_hash, searcher, embedding_cache)

    # In the UI render loop:
    progress = get_progress()   # dict of doc_name → {done, total, status, error}
"""

import logging
import queue
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (survive Streamlit re-runs in same process)
# ---------------------------------------------------------------------------
_job_queue: queue.Queue = queue.Queue()

# progress dict: { doc_name: {done, total, status, error, pct} }
# "status" values: "queued" | "indexing" | "done" | "error"
_progress: Dict[str, Dict] = {}
_progress_lock = threading.Lock()

_worker_started = False
_worker_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _set_progress(doc_name: str, **kwargs):
    with _progress_lock:
        if doc_name not in _progress:
            _progress[doc_name] = {"done": 0, "total": 1, "status": "queued", "error": None, "pct": 0.0}
        _progress[doc_name].update(kwargs)
        done  = _progress[doc_name].get("done", 0)
        total = _progress[doc_name].get("total", 1) or 1
        _progress[doc_name]["pct"] = done / total


def _worker_loop():
    """Daemon worker that processes indexing jobs one at a time."""
    while True:
        try:
            job = _job_queue.get(timeout=30)
        except queue.Empty:
            continue

        doc_name      = job["doc_name"]
        chunks        = job["chunks"]
        file_hash     = job["file_hash"]
        searcher      = job["searcher"]
        emb_cache     = job["embedding_cache"]

        logger.info(f"[IndexQueue] Starting indexing: {doc_name}")
        _set_progress(doc_name, status="indexing", done=0, total=len(chunks), error=None)

        def on_progress(done: int, total: int):
            _set_progress(doc_name, done=done, total=total)

        try:
            searcher.add_documents(
                chunks,
                doc_name=doc_name,
                file_hash=file_hash,
                embedding_cache=emb_cache,
                progress_callback=on_progress,
            )
            _set_progress(doc_name, status="done", done=len(chunks), total=len(chunks))
            logger.info(f"[IndexQueue] Completed indexing: {doc_name}")
        except Exception as e:
            logger.error(f"[IndexQueue] Error indexing {doc_name}: {e}")
            _set_progress(doc_name, status="error", error=str(e))
        finally:
            _job_queue.task_done()


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True, name="IndexingWorker")
            t.start()
            _worker_started = True
            logger.info("[IndexQueue] Worker thread started.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enqueue(
    doc_name: str,
    chunks: list,
    file_hash: str,
    searcher: Any,
    embedding_cache: Any,
):
    """Add a document to the indexing queue."""
    _ensure_worker()
    _set_progress(doc_name, status="queued", done=0, total=len(chunks), error=None)
    _job_queue.put({
        "doc_name":       doc_name,
        "chunks":         chunks,
        "file_hash":      file_hash,
        "searcher":       searcher,
        "embedding_cache": embedding_cache,
    })
    logger.info(f"[IndexQueue] Enqueued: {doc_name} ({len(chunks)} chunks). "
                f"Queue size: {_job_queue.qsize()}")


def get_progress() -> Dict[str, Dict]:
    """Return a snapshot of all indexing job progress states."""
    with _progress_lock:
        return {k: dict(v) for k, v in _progress.items()}


def is_idle() -> bool:
    """True when queue is empty and no job is actively being processed."""
    return _job_queue.empty() and all(
        v["status"] in ("done", "error", "queued") for v in _progress.values()
    )


def any_active() -> bool:
    """True if any job is currently being indexed or waiting in queue."""
    if not _job_queue.empty():
        return True
    with _progress_lock:
        return any(v["status"] in ("queued", "indexing") for v in _progress.values())


def get_completed_docs() -> list:
    """Return list of doc_names that finished indexing successfully."""
    with _progress_lock:
        return [k for k, v in _progress.items() if v["status"] == "done"]


def clear_completed():
    """Remove finished/errored entries from progress dict."""
    with _progress_lock:
        done_keys = [k for k, v in _progress.items() if v["status"] in ("done", "error")]
        for k in done_keys:
            del _progress[k]
