# Indexing and Embedding Generation

import os
import logging
from typing import Callable, List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.rate_limiter import GOOGLE_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

# How many texts to embed per single API call (keeps request payloads small)
_EMBED_BATCH_SIZE = 5


class Indexer:
    def __init__(self):
        # Load API key from d:/BookExpert/google.apikey
        try:
            with open("d:/BookExpert/google.apikey", "r", encoding="utf-8") as f:
                self.api_key = f.read().strip().rstrip(".")
                os.environ["GOOGLE_API_KEY"] = self.api_key
        except Exception as e:
            logger.error("Could not read google.apikey")
            raise e

        logger.info("Initializing Google Generative AI Embeddings (gemini-embedding-001)")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.api_key
        )

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a small batch — the rate limiter waits before the call."""
        GOOGLE_LIMITER.wait()
        return self.embeddings.embed_documents(texts)

    def embed_documents(
        self,
        texts: List[str],
        file_hash: Optional[str] = None,
        embedding_cache=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        Embed texts with optional caching and progress reporting.

        Parameters
        ----------
        texts            : list of text strings to embed
        file_hash        : SHA-256 of the source file; used as cache key
        embedding_cache  : EmbeddingCache instance; skips API if hit
        progress_callback: called as (completed: int, total: int) after each batch
        """
        total = len(texts)
        logger.info(
            f"Generating embeddings for {total} chunks "
            f"(batches of {_EMBED_BATCH_SIZE}, ~{60/_EMBED_BATCH_SIZE:.0f}s/batch at 15 RPM)"
        )

        all_vectors: List[List[float]] = [None] * total

        # --- Fill from cache first ---
        uncached_indices: List[int] = []
        if file_hash and embedding_cache:
            for i in range(total):
                vec = embedding_cache.get(file_hash, i)
                if vec is not None:
                    all_vectors[i] = vec
                else:
                    uncached_indices.append(i)
            if len(uncached_indices) < total:
                logger.info(
                    f"  Embedding cache: {total - len(uncached_indices)}/{total} chunks reused, "
                    f"{len(uncached_indices)} need API calls."
                )
        else:
            uncached_indices = list(range(total))

        # --- API calls for uncached chunks ---
        completed = total - len(uncached_indices)
        if progress_callback:
            progress_callback(completed, total)

        for batch_start in range(0, len(uncached_indices), _EMBED_BATCH_SIZE):
            batch_idxs = uncached_indices[batch_start: batch_start + _EMBED_BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idxs]
            vectors = self._embed_batch(batch_texts)

            for local_i, global_i in enumerate(batch_idxs):
                all_vectors[global_i] = vectors[local_i]
                if file_hash and embedding_cache:
                    embedding_cache.put(file_hash, global_i, vectors[local_i])

            completed += len(batch_idxs)
            logger.info(f"  Embedded {completed}/{total} chunks")
            if progress_callback:
                progress_callback(completed, total)

        return all_vectors

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=60)
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string with rate limiting."""
        GOOGLE_LIMITER.wait()
        return self.embeddings.embed_query(query)
