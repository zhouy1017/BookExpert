# Map-Reduce Summarization and QA via DeepSeek
import os
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.rate_limiter import DEEPSEEK_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

# Truncate individual chunks for the map step to avoid hitting context window limits
_MAX_CHUNK_CHARS = 8000


class BookSummarizer:
    def __init__(self):
        try:
            with open("d:/BookExpert/deepseek.apikey", "r", encoding="utf-8") as f:
                self.api_key = f.read().strip().rstrip(".")
                os.environ["DEEPSEEK_API_KEY"] = self.api_key
        except Exception as e:
            logger.error("Could not read deepseek.apikey")
            raise e

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.3
        )

    # ------------------------------------------------------------------
    # Map-Reduce Summarization
    # ------------------------------------------------------------------
    def summarize_large_document(
        self,
        texts: List[str],
        batch_size: int = 20,
        file_hash: Optional[str] = None,
        summary_cache=None,
    ) -> str:
        """
        Map-Reduce summarization over a list of text chunks.

        Checks SummaryCache first using key ``summary_{file_hash}`` if provided.
        Writes the result back to cache after computation.
        """
        if not texts:
            return "No text to summarize."

        # --- Cache read ---
        cache_key = f"summary_{file_hash}" if file_hash else None
        if cache_key and summary_cache:
            cached = summary_cache.get(cache_key)
            if cached:
                logger.info("Summary cache hit — returning stored summary.")
                return cached

        logger.info(f"[MAP] Summarizing {len(texts)} chunks in batches of {batch_size}...")

        # MAP step
        intermediate_summaries: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for chunk in batch:
                truncated = chunk[:_MAX_CHUNK_CHARS]
                summary = self._invoke_map(truncated)
                intermediate_summaries.append(summary)
            logger.info(f"  Mapped {min(i + batch_size, len(texts))}/{len(texts)} chunks.")

        logger.info(f"[REDUCE] Reducing {len(intermediate_summaries)} intermediate summaries...")

        # REDUCE step — recursively reduce until single summary
        while len(intermediate_summaries) > 1:
            next_level: List[str] = []
            for i in range(0, len(intermediate_summaries), batch_size):
                batch = intermediate_summaries[i:i + batch_size]
                combined = "\n\n".join([f"[Segment {i + j + 1}]: {s}" for j, s in enumerate(batch)])
                reduced = self._invoke_reduce(combined)
                next_level.append(reduced)
            intermediate_summaries = next_level
            logger.info(f"  Reduce pass produced {len(intermediate_summaries)} summaries.")

        result = intermediate_summaries[0] if intermediate_summaries else "Could not generate summary."

        # --- Cache write ---
        if cache_key and summary_cache:
            summary_cache.put(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Multi-Turn QA with full conversation history + optional attachment
    # ------------------------------------------------------------------
    def answer_question(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        attachment_text: str = "",
    ) -> str:
        """
        Answer a question using retrieved context, full conversation history,
        and an optional in-chat file attachment.

        Parameters
        ----------
        query          : the new user question
        context_chunks : list of dicts with 'text', 'doc_name', 'score'
        history        : list of {role: "user"|"assistant", content: str} dicts
                         representing prior turns (oldest first)
        attachment_text: raw text extracted from a chat-attached file
        """
        if not context_chunks and not attachment_text:
            return "No relevant context found in the book to answer your question."

        history = history or []

        # --- Build context string from retrieved book chunks ---
        book_context = "\n\n".join([
            f"Excerpt from '{c.get('doc_name', 'book')}' (Score: {c.get('score', 0):.2f}):\n{c['text']}"
            for c in context_chunks
        ]) if context_chunks else ""

        # --- Compose messages ---
        messages = []

        # 1. System: role definition
        messages.append(SystemMessage(content=(
            "你是书籍专家（BookExpert），一位精通书籍内容的中文AI助手，能进行多轮对话。"
            "请仅根据提供的书籍摘录和附件内容用简体中文回答用户的问题。"
            "若内容中未包含答案，请直接说明无法从书中找到相关信息。"
        )))

        # 2. System: book context (if any)
        if book_context:
            messages.append(SystemMessage(content=f"--- 书籍相关摘录 ---\n{book_context}"))

        # 3. System: attachment context (if any)
        if attachment_text:
            snippet = attachment_text[:12000]  # guard context window
            messages.append(SystemMessage(content=f"--- 📎 用户附件内容（仅本会话有效）---\n{snippet}"))

        # 4. Full conversation history
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 5. New question
        messages.append(HumanMessage(content=query))

        logger.info(
            f"Answering question with {len(context_chunks)} chunks, "
            f"{len(history)} history turns, attachment={'yes' if attachment_text else 'no'}."
        )
        return self._invoke_with_messages(messages)

    # ------------------------------------------------------------------
    # Internal rate-limited + retry-wrapped LLM invocations
    # ------------------------------------------------------------------
    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_map(self, chunk: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content="你是一位书籍分析专家，请用简体中文简要总结以下书籍段落，重点提炼关键事件、人物和核心思想。"),
            HumanMessage(content=chunk),
        ]
        return self.llm.invoke(messages).content

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_reduce(self, summaries: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content="你是一位书籍分析专家。你将收到一本书各段落的摘要。请用简体中文将它们整合成一份完整、连贯的全书摘要。"),
            HumanMessage(content=summaries),
        ]
        return self.llm.invoke(messages).content

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_with_messages(self, messages: list) -> str:
        DEEPSEEK_LIMITER.wait()
        return self.llm.invoke(messages).content
