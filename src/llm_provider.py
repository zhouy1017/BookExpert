"""
Dual-LLM Provider
==================
Primary:  Google Gemini 3.1 Flash Lite (via ChatGoogleGenerativeAI)
Fallback: DeepSeek Chat (via ChatOpenAI)

On any quota / API error from the primary, the provider automatically retries
with the fallback model.  A RateLimiter is applied before each call.

Usage
-----
    from src.llm_provider import get_llm_provider

    provider = get_llm_provider()          # cached singleton
    response = provider.invoke(messages)   # returns AIMessage; handles fallback
"""

import logging
import os
from typing import List, Optional

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# ─── Lazy imports so missing optional packages raise at call time, not import ──
def _make_gemini(api_key: str, model: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,   # Gemini doesn't support system role natively
    )


def _make_deepseek(api_key: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=temperature,
    )


# ─── Primary model name ─────────────────────────────────────────────────────
GEMINI_CHAT_MODEL = "gemini-3.1-flash-lite-preview"


class LLMProvider:
    """
    Wraps a primary (Gemini) and fallback (DeepSeek) LLM.

    Call `provider.invoke(messages)` — it behaves exactly like the underlying
    LangChain chat model's `.invoke()`.  On quota/API errors from the primary it
    automatically falls back to DeepSeek.
    """

    def __init__(self, temperature: float = 0.3):
        self._temperature = temperature
        self._primary_name  = GEMINI_CHAT_MODEL
        self._fallback_name = "deepseek-chat"

        # Load API keys
        try:
            with open("d:/BookExpert/google.apikey", "r", encoding="utf-8") as f:
                self._google_key = f.read().strip().rstrip(".")
        except Exception as e:
            logger.warning(f"Could not read google.apikey — Gemini primary unavailable: {e}")
            self._google_key = None

        try:
            with open("d:/BookExpert/deepseek.apikey", "r", encoding="utf-8") as f:
                self._deepseek_key = f.read().strip().rstrip(".")
                os.environ["DEEPSEEK_API_KEY"] = self._deepseek_key
        except Exception as e:
            logger.error(f"Could not read deepseek.apikey: {e}")
            self._deepseek_key = None

        # Build LLM objects lazily to avoid import errors at startup
        self._primary  = None
        self._fallback = None
        self._primary_failed = False   # once set, skip primary permanently

        if self._google_key:
            try:
                self._primary = _make_gemini(self._google_key, self._primary_name, temperature)
                logger.info(f"LLMProvider: primary model = {self._primary_name}")
            except Exception as e:
                logger.warning(f"LLMProvider: failed to initialise Gemini primary: {e}")
                self._primary_failed = True

        if self._deepseek_key:
            try:
                self._fallback = _make_deepseek(self._deepseek_key, temperature)
                logger.info("LLMProvider: fallback model = deepseek-chat")
            except Exception as e:
                logger.error(f"LLMProvider: failed to initialise DeepSeek fallback: {e}")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #
    def invoke(self, messages: List[BaseMessage]) -> "BaseMessage":
        """
        Invoke the primary LLM; fall back to DeepSeek on any error.
        Raises RuntimeError if both models fail.
        """
        from src.rate_limiter import GEMINI_LLM_LIMITER, DEEPSEEK_LIMITER

        if self._primary and not self._primary_failed:
            try:
                GEMINI_LLM_LIMITER.wait()
                result = self._primary.invoke(messages)
                logger.debug(f"LLMProvider: response from {self._primary_name}")
                return result
            except Exception as e:
                if _is_quota_or_unavailable(e):
                    logger.warning(
                        f"LLMProvider: primary {self._primary_name} quota/error "
                        f"({_brief(e)}) — falling back to DeepSeek."
                    )
                else:
                    logger.error(f"LLMProvider: primary error — {e!r} — falling back.")

        if self._fallback:
            DEEPSEEK_LIMITER.wait()
            return self._fallback.invoke(messages)

        raise RuntimeError(
            "LLMProvider: both primary (Gemini) and fallback (DeepSeek) are unavailable."
        )

    @property
    def active_model_name(self) -> str:
        if self._primary and not self._primary_failed:
            return self._primary_name
        if self._fallback:
            return self._fallback_name
        return "none"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _is_quota_or_unavailable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "quota", "resource_exhausted", "rate_limit",
        "too many requests", "exhausted", "unavailable",
    ))


def _brief(exc: BaseException) -> str:
    return str(exc)[:120]


# ─── Singleton factory ────────────────────────────────────────────────────────
_PROVIDER_CACHE: dict = {}


def get_llm_provider(temperature: float = 0.3) -> LLMProvider:
    """Return a cached LLMProvider for the given temperature."""
    if temperature not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[temperature] = LLMProvider(temperature)
    return _PROVIDER_CACHE[temperature]
