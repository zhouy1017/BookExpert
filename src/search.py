# Hybrid Search (BM25 + Semantic)
import os
import sqlite3
import logging
import uuid
import jieba
from typing import Any, Callable, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, VectorParams, PointStruct

from src.indexing import Indexer

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(self, db_path: str = "db"):
        os.makedirs(db_path, exist_ok=True)
        self.qdrant_path = os.path.join(db_path, "qdrant_storage")
        self.sqlite_path = os.path.join(db_path, "metadata.db")
        self.collection_name = "book_chunks"

        # Initialize clients
        self.qclient = QdrantClient(path=self.qdrant_path)
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.indexer = Indexer()

        self.bm25_corpus_ids = []
        self.bm25_corpus_texts = []
        self.bm25_corpus_docnames = []
        self.bm25 = None

        self._init_db()
        self._load_bm25()

    # ------------------------------------------------------------------
    # Schema / Init
    # ------------------------------------------------------------------
    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_name     TEXT,
                text_content TEXT
            )
        ''')
        self.conn.commit()

        # Init Qdrant Collection
        if not self.qclient.collection_exists(self.collection_name):
            self.qclient.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )

    def _load_bm25(self):
        """Loads all chunks from SQLite and builds the BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed")
            return

        logger.info("Building BM25 index from SQLite...")
        c = self.conn.cursor()
        c.execute("SELECT chunk_id, doc_name, text_content FROM chunks")
        rows = c.fetchall()

        self.bm25_corpus_ids = [r[0] for r in rows]
        self.bm25_corpus_docnames = [r[1] for r in rows]
        self.bm25_corpus_texts = [r[2] for r in rows]

        tokenized_corpus = [list(jieba.cut(text)) for text in self.bm25_corpus_texts]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(rows)} chunks.")

    def _rebuild_bm25(self):
        """Re-initialize BM25 from the in-memory corpus lists."""
        from rank_bm25 import BM25Okapi
        if self.bm25_corpus_texts:
            tokenized_corpus = [list(jieba.cut(text)) for text in self.bm25_corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    # ------------------------------------------------------------------
    # Document Management
    # ------------------------------------------------------------------
    def get_indexed_documents(self) -> List[str]:
        """Return a sorted list of distinct doc_name values."""
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT doc_name FROM chunks ORDER BY doc_name")
        return [row[0] for row in c.fetchall()]

    def get_chunks_for_doc(self, doc_name: str) -> List[str]:
        """Return all text chunks for a specific document."""
        c = self.conn.cursor()
        c.execute("SELECT text_content FROM chunks WHERE doc_name=? ORDER BY rowid", (doc_name,))
        return [row[0] for row in c.fetchall()]

    def delete_document(self, doc_name: str):
        """Remove a document entirely from Qdrant, SQLite, and BM25."""
        logger.info(f"Deleting document: {doc_name}")

        # 1. Delete from Qdrant by payload filter
        try:
            self.qclient.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
                ),
            )
        except Exception as e:
            logger.warning(f"Qdrant delete warning: {e}")

        # 2. Delete from SQLite
        c = self.conn.cursor()
        c.execute("DELETE FROM chunks WHERE doc_name=?", (doc_name,))
        self.conn.commit()

        # 3. Remove from in-memory BM25 corpus
        new_ids, new_texts, new_docnames = [], [], []
        for cid, txt, dn in zip(self.bm25_corpus_ids, self.bm25_corpus_texts, self.bm25_corpus_docnames):
            if dn != doc_name:
                new_ids.append(cid)
                new_texts.append(txt)
                new_docnames.append(dn)
        self.bm25_corpus_ids = new_ids
        self.bm25_corpus_texts = new_texts
        self.bm25_corpus_docnames = new_docnames
        self._rebuild_bm25()
        logger.info(f"Document '{doc_name}' deleted.")

    def add_documents(
        self,
        chunks: List[str],
        doc_name: str = "unknown",
        file_hash: Optional[str] = None,
        embedding_cache=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Embed text chunks and store vectors in Qdrant, text in SQLite.
        Supports embedding cache and progress reporting.
        """
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} documents to searcher...")
        embeddings = self.indexer.embed_documents(
            chunks,
            file_hash=file_hash,
            embedding_cache=embedding_cache,
            progress_callback=progress_callback,
        )

        points = []
        c = self.conn.cursor()
        new_corpus_texts = []
        new_corpus_ids = []
        new_corpus_docnames = []

        for i, text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            vector = embeddings[i]

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"doc_name": doc_name}
                )
            )
            c.execute(
                "INSERT INTO chunks (chunk_id, doc_name, text_content) VALUES (?, ?, ?)",
                (chunk_id, doc_name, text)
            )
            new_corpus_ids.append(chunk_id)
            new_corpus_texts.append(text)
            new_corpus_docnames.append(doc_name)

        self.conn.commit()
        self.qclient.upsert(collection_name=self.collection_name, points=points)

        # Update in-memory BM25
        self.bm25_corpus_ids.extend(new_corpus_ids)
        self.bm25_corpus_texts.extend(new_corpus_texts)
        self.bm25_corpus_docnames.extend(new_corpus_docnames)
        self._rebuild_bm25()

        logger.info("Successfully added documents.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        limit: int = 5,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining Vector (Qdrant) and Keyword (BM25) using RRF.

        Parameters
        ----------
        query      : search query string
        limit      : number of results to return
        doc_filter : if provided, restrict results to these doc_name values only
        """
        if not self.bm25_corpus_ids or self.bm25 is None:
            return []

        k = 60  # RRF constant
        rrf_scores: Dict[str, float] = {}

        # --- 1. Vector Search ---
        qdrant_filter = None
        if doc_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(key="doc_name", match=MatchAny(any=doc_filter))]
            )

        query_vector = self.indexer.embed_query(query)
        qdrant_response = self.qclient.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit * 2,
        )
        for rank, hit in enumerate(qdrant_response.points):
            cid = str(hit.id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1.0)

        # --- 2. BM25 Search ---
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Build allowed set for BM25 if filtering
        allowed_ids: Optional[set] = None
        if doc_filter:
            allowed_ids = {
                cid for cid, dn in zip(self.bm25_corpus_ids, self.bm25_corpus_docnames)
                if dn in doc_filter
            }

        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True
        )[:limit * 2]

        for rank, idx in enumerate(top_bm25_indices):
            cid = str(self.bm25_corpus_ids[idx])
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if bm25_scores[idx] > 0:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1.0)

        # --- 3. Combine & fetch ---
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]

        final_results = []
        c = self.conn.cursor()
        for cid in sorted_cids:
            c.execute("SELECT text_content, doc_name FROM chunks WHERE chunk_id=?", (str(cid),))
            row = c.fetchone()
            if row:
                final_results.append({
                    "chunk_id": cid,
                    "text": row[0],
                    "doc_name": row[1],
                    "score": rrf_scores[cid],
                })

        return final_results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def clear_all_documents(self):
        """
        Remove ALL indexed documents from both Qdrant (vectors) and SQLite (metadata).
        Rebuilds the BM25 index with empty state.
        """
        try:
            # Drop and recreate the Qdrant collection
            self.qclient.delete_collection(self.collection_name)
            self._init_db()   # re-creates Qdrant collection + SQLite table
        except Exception as e:
            logger.warning(f"clear_all_documents: Qdrant reset warning: {e}")

        try:
            c = self.conn.cursor()
            c.execute("DELETE FROM chunks")
            self.conn.commit()
        except Exception as e:
            logger.error(f"clear_all_documents: SQLite clear error: {e}")

        # Reset in-memory BM25 corpus
        self.bm25_corpus_ids       = []
        self.bm25_corpus_texts     = []
        self.bm25_corpus_docnames  = []
        self.bm25                  = None
        logger.info("clear_all_documents: all index data cleared.")
