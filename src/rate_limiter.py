"""
API Rate Limiting Utilities
============================
Provides:
- RateLimiter:                sliding-window RPM limiter (DeepSeek, etc.)
                               + get_status() for UI display
- GeminiEmbeddingLimiter:     sliding-window RPM + TPM + RPD limiter for Google
                               Gemini embedding APIs, with dual-model fallback
                               + get_status() for UI display
- retry_on_rate_limit:        tenacity decorator for 429 / ResourceExhausted retries

Gemini free-tier limits for gemini-embedding-001 / gemini-embedding-002:
    100 RPM  (requests per minute)
     30 000 TPM  (tokens per minute)
      1 000 RPD  (requests per day)

Token estimation: ~1 token ≈ 4 characters of UTF-8 text (conservative proxy).
"""

import datetime
import logging
import threading
import time
from collections import deque
from datetime import date
from typing import Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4   # conservative Gemini token estimate


# ---------------------------------------------------------------------------
# Helper: detect rate-limit / transient errors worth retrying
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "rate limit", "resource_exhausted", "quota",
        "unavailable", "too many requests", "exhausted",
    ))


def is_quota_error(exc: BaseException) -> bool:
    """True when exc represents an API quota / billing exhaustion (for UI alerts)."""
    msg = str(exc).lower()
    return any(k in msg for k in (
        "quota", "resource_exhausted", "429", "too many requests",
        "billing", "exhausted",
    ))


# ---------------------------------------------------------------------------
# Decorator: exponential-backoff retry on rate-limit errors
# ---------------------------------------------------------------------------
def retry_on_rate_limit(max_attempts: int = 6, wait_min: float = 4, wait_max: float = 60):
    """
    Decorator that retries with exponential back-off when a rate-limit error
    (429 / ResourceExhausted) is detected.
    """
    return retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=wait_min, max=wait_max),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Sliding-Window RPM Rate Limiter (DeepSeek / generic)
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Thread-safe sliding-window rate limiter (requests per minute).

    Tracks actual call timestamps so get_status() can report remaining capacity.

    Usage:
        limiter = RateLimiter(rpm=50)
        limiter.wait()            # call before each API request
        info = limiter.get_status()  # {"rpm_limit", "rpm_used", "rpm_remaining"}
    """
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._min_interval = 60.0 / rpm
        self._lock          = threading.Lock()
        self._last_call: float = 0.0
        self._window: deque[float] = deque()   # monotonic timestamps of last 60 s

    def wait(self):
        """Block until the next API call is allowed, then record it."""
        with self._lock:
            now = time.monotonic()
            sleep_time = self._min_interval - (now - self._last_call)
            if sleep_time > 0:
                logger.debug(f"RateLimiter sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            now = time.monotonic()
            # prune window
            cutoff = now - 60.0
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
            self._window.append(now)
            self._last_call = now

    def get_status(self) -> Dict:
        """Return current sliding-window usage within the last 60 seconds."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            # count without mutating (don't prune here — wait() will)
            used = sum(1 for t in self._window if t >= cutoff)
        remaining = max(0, self.rpm - used)
        return {
            "rpm_limit":     self.rpm,
            "rpm_used":      used,
            "rpm_remaining": remaining,
            "pct_used":      used / self.rpm if self.rpm else 0.0,
        }


# ---------------------------------------------------------------------------
# Per-model sliding-window bucket (RPM + TPM + RPD)
# ---------------------------------------------------------------------------
class _ModelBucket:
    def __init__(self, model_name: str, rpm: int, tpm: int, rpd: int):
        self.model_name = model_name
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd

        self._lock          = threading.Lock()
        self._req_times:    deque[float]         = deque()
        self._tok_log:      deque[Tuple[float, int]] = deque()
        self._day_count:    int  = 0
        self._day_stamp:    date = date.today()

    def _prune_window(self, now: float):
        cutoff = now - 60.0
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._tok_log and self._tok_log[0][0] < cutoff:
            self._tok_log.popleft()

    def _reset_day_if_needed(self):
        today = date.today()
        if today != self._day_stamp:
            self._day_stamp = today
            self._day_count = 0

    def can_accept(self, token_estimate: int) -> bool:
        with self._lock:
            now = time.monotonic()
            self._prune_window(now)
            self._reset_day_if_needed()
            if self._day_count >= self.rpd:
                return False
            if len(self._req_times) >= self.rpm:
                return False
            if sum(t for _, t in self._tok_log) + token_estimate > self.tpm:
                return False
            return True

    def record(self, token_estimate: int):
        with self._lock:
            now = time.monotonic()
            self._prune_window(now)
            self._reset_day_if_needed()
            self._req_times.append(now)
            self._tok_log.append((now, token_estimate))
            self._day_count += 1

    def seconds_until_slot(self, token_estimate: int) -> float:
        with self._lock:
            now = time.monotonic()
            self._prune_window(now)
            self._reset_day_if_needed()

            if self._day_count >= self.rpd:
                tomorrow = datetime.date.today().toordinal() + 1
                midnight  = datetime.datetime.combine(
                    datetime.date.fromordinal(tomorrow), datetime.time.min
                )
                return max(0.0, (midnight - datetime.datetime.now()).total_seconds())

            wait = 0.0
            if len(self._req_times) >= self.rpm:
                wait = max(wait, 60.0 - (now - self._req_times[0]) + 0.05)
            used_tpm = sum(t for _, t in self._tok_log)
            if used_tpm + token_estimate > self.tpm and self._tok_log:
                wait = max(wait, 60.0 - (now - self._tok_log[0][0]) + 0.05)
            return wait

    def get_status(self) -> Dict:
        with self._lock:
            now = time.monotonic()
            self._prune_window(now)
            self._reset_day_if_needed()
            rpm_used = len(self._req_times)
            tpm_used = sum(t for _, t in self._tok_log)
            rpd_used = self._day_count
        return {
            "model":          self.model_name,
            "rpm_used":       rpm_used,
            "rpm_remaining":  max(0, self.rpm - rpm_used),
            "rpm_limit":      self.rpm,
            "tpm_used":       tpm_used,
            "tpm_remaining":  max(0, self.tpm - tpm_used),
            "tpm_limit":      self.tpm,
            "rpd_used":       rpd_used,
            "rpd_remaining":  max(0, self.rpd - rpd_used),
            "rpd_limit":      self.rpd,
            "exhausted":      (
                rpm_used >= self.rpm
                or tpm_used >= self.tpm
                or rpd_used >= self.rpd
            ),
        }


# ---------------------------------------------------------------------------
# Gemini Embedding Rate Limiter — Dual-model with RPM + TPM + RPD tracking
# ---------------------------------------------------------------------------
class GeminiEmbeddingLimiter:
    """
    Dual-model rate limiter for Google Gemini embedding APIs.

    Logic:
      1. Try primary model.
      2. If primary quota full → fall back to secondary.
      3. If both exhausted → sleep until earliest slot available.

    get_status() returns live quota snapshot for both models.
    """
    def __init__(
        self,
        primary_model:  str = "models/gemini-embedding-001",
        fallback_model: str = "models/gemini-embedding-002",
        rpm: int = 100,
        tpm: int = 30_000,
        rpd: int = 1_000,
    ):
        self._primary  = _ModelBucket(primary_model,  rpm, tpm, rpd)
        self._fallback = _ModelBucket(fallback_model, rpm, tpm, rpd)

    def _estimate_tokens(self, texts: List[str]) -> int:
        return max(1, sum(len(t) for t in texts) // _CHARS_PER_TOKEN)

    def wait_and_get_model(self, texts: List[str]) -> str:
        """Block until a model slot is available; returns model name to use."""
        token_estimate = self._estimate_tokens(texts)

        while True:
            if self._primary.can_accept(token_estimate):
                self._primary.record(token_estimate)
                return self._primary.model_name

            if self._fallback.can_accept(token_estimate):
                self._fallback.record(token_estimate)
                logger.warning(
                    f"GeminiLimiter: primary quota full, using fallback "
                    f"({self._fallback.model_name})"
                )
                return self._fallback.model_name

            wait_p = self._primary.seconds_until_slot(token_estimate)
            wait_f = self._fallback.seconds_until_slot(token_estimate)
            sleep_for = max(min(wait_p, wait_f), 1.0)
            logger.warning(
                f"GeminiLimiter: both models at quota. "
                f"Sleeping {sleep_for:.1f}s."
            )
            time.sleep(sleep_for)

    def get_status(self) -> Dict:
        """Return quota snapshot for both models, plus combined summary."""
        p = self._primary.get_status()
        f = self._fallback.get_status()
        both_exhausted = p["exhausted"] and f["exhausted"]
        return {
            "primary":        p,
            "fallback":       f,
            "both_exhausted": both_exhausted,
            # Aggregate remaining (useful for a single progress indicator)
            "total_rpm_remaining": p["rpm_remaining"] + f["rpm_remaining"],
            "total_rpd_remaining": p["rpd_remaining"] + f["rpd_remaining"],
        }


# ---------------------------------------------------------------------------
# Singleton limiters
# ---------------------------------------------------------------------------
GEMINI_EMBEDDING_LIMITER = GeminiEmbeddingLimiter(
    primary_model  = "models/gemini-embedding-001",
    fallback_model = "models/gemini-embedding-002",
    rpm = 100,
    tpm = 30_000,
    rpd = 1_000,
)

DEEPSEEK_LIMITER = RateLimiter(rpm=50)

# Backward-compatible alias
GOOGLE_LIMITER = RateLimiter(rpm=100)
