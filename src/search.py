# Hybrid Search (BM25 + Semantic)
import os
import sqlite3
import logging
import uuid
import jieba
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

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
        self.bm25 = None
        
        self._init_db()
        self._load_bm25()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_name TEXT,
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
        logger.info("Building BM25 index from SQLite...")
        c = self.conn.cursor()
        c.execute("SELECT chunk_id, text_content FROM chunks")
        rows = c.fetchall()
        
        self.bm25_corpus_ids = [r[0] for r in rows]
        self.bm25_corpus_texts = [r[1] for r in rows]
        
        tokenized_corpus = [list(jieba.cut(text)) for text in self.bm25_corpus_texts]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(rows)} chunks.")

    def add_documents(self, chunks: List[str], doc_name: str = "unknown"):
        """Embeds text chunks, storing vectors in Qdrant and text in SQLite."""
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} documents to searcher...")
        # Get embeddings
        embeddings = self.indexer.embed_documents(chunks)
        
        points = []
        c = self.conn.cursor()
        
        new_corpus_texts = []
        new_corpus_ids = []
        
        for i, text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            vector = embeddings[i]
            
            # Prepare Qdrant Point
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"doc_name": doc_name}
                )
            )
            
            # Insert to SQLite
            c.execute("INSERT INTO chunks (chunk_id, doc_name, text_content) VALUES (?, ?, ?)",
                      (chunk_id, doc_name, text))
                      
            new_corpus_ids.append(chunk_id)
            new_corpus_texts.append(text)
            
        # Commit SQLite
        self.conn.commit()
        
        # Upsert to Qdrant
        self.qclient.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Re-build BM25 in memory dynamically
        self.bm25_corpus_ids.extend(new_corpus_ids)
        self.bm25_corpus_texts.extend(new_corpus_texts)
        if self.bm25_corpus_texts:
            tokenized_corpus = [list(jieba.cut(text)) for text in self.bm25_corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Successfully added documents.")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining Vector (Qdrant) and Keyword (BM25) using RRF."""
        if not self.bm25_corpus_ids or self.bm25 is None:
            return []
            
        # 1. Vector Search using the modern query_points API
        query_vector = self.indexer.embed_query(query)
        qdrant_response = self.qclient.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit * 2  # get more candidates for RRF
        )
        qdrant_results = qdrant_response.points
        
        # Rank dict: chunk_id -> rank score
        rrf_scores: Dict[str, float] = {}
        k = 60 # RRF constant
        
        # Score Vector results
        for rank, hit in enumerate(qdrant_results):
            cid = str(hit.id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1.0)
            
        # 2. BM25 Search
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top limit*2 from BM25
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True)
        top_bm25_indices = top_bm25_indices[:limit*2]
        
        for rank, idx in enumerate(top_bm25_indices):
            if bm25_scores[idx] > 0: # only score if it actually matches
                cid = str(self.bm25_corpus_ids[idx])
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1.0)
                
        # Combine and sort
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]
        
        # Fetch text for the final results
        final_results = []
        c = self.conn.cursor()
        for cid in sorted_cids:
            c.execute("SELECT text_content, doc_name FROM chunks WHERE chunk_id = ?", (str(cid),))
            row = c.fetchone()
            if row:
                final_results.append({
                    "chunk_id": cid,
                    "text": row[0],
                    "doc_name": row[1],
                    "score": rrf_scores[cid]
                })
                
        return final_results

