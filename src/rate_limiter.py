"""
API Rate Limiting Utility
=========================
Provides:
- RateLimiter: a token-bucket style rate limiter (requests per minute) with
               inter-call sleep delays to stay under the API quota.
- retry_on_rate_limit: a decorator using tenacity to automatically retry
                        requests that hit HTTP 429 / rate-limit errors with
                        exponential backoff.

Default limits (conservative):
  Google free tier:  15 RPM  (actual limit is 60-100 RPM; 15 gives headroom)
  DeepSeek paid:     50 RPM  (actual limit varies; 50 is safe for most plans)
"""

import time
import logging
import threading
from functools import wraps
from typing import Callable, Any

from tenacity import (
    retry,
    retry_if_exception,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: detect rate-limit / transient errors worth retrying
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "resource_exhausted", "quota", "unavailable", "too many requests"))


# ---------------------------------------------------------------------------
# Decorator: exponential-backoff retry on rate-limit errors
# ---------------------------------------------------------------------------
def retry_on_rate_limit(max_attempts: int = 6, wait_min: float = 4, wait_max: float = 60):
    """
    Decorator that retries the wrapped function with exponential back-off
    when a rate-limit error (429 / ResourceExhausted) is detected.
    """
    return retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=wait_min, max=wait_max),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Token-bucket Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Usage:
        limiter = RateLimiter(rpm=15)   # 15 requests per minute
        limiter.wait()                  # call before each API request
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._min_interval = 60.0 / rpm   # seconds between requests
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Block until the next API call is allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            sleep_time = self._min_interval - elapsed
            if sleep_time > 0:
                logger.debug(f"Rate limiter sleeping {sleep_time:.2f}s ...")
                time.sleep(sleep_time)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Singleton limiters — import these wherever you make API calls
# ---------------------------------------------------------------------------
# Google free tier: gemini-embedding-001 → 60 RPM (empirically safe at 15 RPM)
GOOGLE_LIMITER = RateLimiter(rpm=15)

# DeepSeek paid tier: safe default 50 RPM
DEEPSEEK_LIMITER = RateLimiter(rpm=50)
