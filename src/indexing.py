# Indexing and Embedding Generation
# Supports dual-model Gemini embedding with sliding-window rate limiting.

import logging
import os
from typing import Callable, List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.rate_limiter import GEMINI_EMBEDDING_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

# Texts per single API call — keeps payloads small and token estimates accurate
_EMBED_BATCH_SIZE = 5


class Indexer:
    def __init__(self):
        try:
            with open("d:/BookExpert/google.apikey", "r", encoding="utf-8") as f:
                self.api_key = f.read().strip().rstrip(".")
                os.environ["GOOGLE_API_KEY"] = self.api_key
        except Exception as e:
            logger.error("Could not read google.apikey")
            raise e

        # Both embedding model instances are created up-front;
        # GeminiEmbeddingLimiter decides which one to use at call time.
        logger.info("Initializing Gemini embedding models (primary + fallback)")
        self._model_instances = {
            "models/gemini-embedding-001": GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_key,
            ),
            "models/gemini-embedding-002": GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-002",
                google_api_key=self.api_key,
            ),
        }
        # Default embeddings object (used for embed_query compatibility)
        self.embeddings = self._model_instances["models/gemini-embedding-001"]

    def _get_model(self, model_name: str) -> GoogleGenerativeAIEmbeddings:
        return self._model_instances.get(model_name, self.embeddings)

    @retry_on_rate_limit(max_attempts=8, wait_min=5, wait_max=120)
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts. GeminiEmbeddingLimiter:
          1. Picks the model (primary or fallback) that has quota.
          2. Records the usage for rate tracking.
          3. Blocks (sleeps) if both models are exhausted.
        """
        model_name = GEMINI_EMBEDDING_LIMITER.wait_and_get_model(texts)
        model_obj  = self._get_model(model_name)
        logger.debug(f"Embedding {len(texts)} texts with {model_name}")
        return model_obj.embed_documents(texts)

    def embed_documents(
        self,
        texts: List[str],
        file_hash: Optional[str] = None,
        embedding_cache=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        Embed texts with optional cache and progress reporting.

        Parameters
        ----------
        texts            : list of text strings to embed
        file_hash        : SHA-256 of source file; used as cache key
        embedding_cache  : EmbeddingCache instance; skips API if hit
        progress_callback: called as (completed, total) after each batch
        """
        total = len(texts)
        logger.info(f"Embedding {total} chunks (batch={_EMBED_BATCH_SIZE})")

        all_vectors: List[Optional[List[float]]] = [None] * total

        # --- Fill from cache first ---
        uncached_indices: List[int] = []
        if file_hash and embedding_cache:
            for i in range(total):
                vec = embedding_cache.get(file_hash, i)
                if vec is not None:
                    all_vectors[i] = vec
                else:
                    uncached_indices.append(i)
            cached_count = total - len(uncached_indices)
            if cached_count:
                logger.info(f"  Cache hit: {cached_count}/{total} chunks reused.")
        else:
            uncached_indices = list(range(total))

        # --- Report cache-filled progress ---
        completed = total - len(uncached_indices)
        if progress_callback:
            progress_callback(completed, total)

        # --- API calls for uncached chunks ---
        for batch_start in range(0, len(uncached_indices), _EMBED_BATCH_SIZE):
            batch_idxs  = uncached_indices[batch_start: batch_start + _EMBED_BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idxs]
            vectors     = self._embed_batch(batch_texts)

            for local_i, global_i in enumerate(batch_idxs):
                all_vectors[global_i] = vectors[local_i]
                if file_hash and embedding_cache:
                    embedding_cache.put(file_hash, global_i, vectors[local_i])

            completed += len(batch_idxs)
            logger.info(f"  Embedded {completed}/{total}")
            if progress_callback:
                progress_callback(completed, total)

        return all_vectors

    @retry_on_rate_limit(max_attempts=8, wait_min=5, wait_max=60)
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string using the rate limiter."""
        model_name = GEMINI_EMBEDDING_LIMITER.wait_and_get_model([query])
        return self._get_model(model_name).embed_query(query)
