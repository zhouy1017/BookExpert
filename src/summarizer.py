# Map-Reduce Summarization and QA via DeepSeek — RAG-first design
import os
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.rate_limiter import DEEPSEEK_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

# Truncate individual chunks for the map step to avoid hitting context window limits
_MAX_CHUNK_CHARS = 8000

# Broad summary seed queries — used for RAG-guided summarization
_SUMMARY_QUERIES = [
    "主要人物 故事情节 关键事件",
    "核心主题 思想内涵 中心思想",
    "故事背景 时代环境 社会背景",
    "结局 高潮 转折点",
    "作者观点 价值观 启示",
]
_SUMMARY_CHUNKS_PER_QUERY = 10


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
    # RAG-Guided Map-Reduce Summarization
    # ------------------------------------------------------------------
    def rag_summarize(
        self,
        doc_name: str,
        searcher: Any,
        top_k: int = 40,
        batch_size: int = 10,
        file_hash: Optional[str] = None,
        summary_cache=None,
    ) -> str:
        """
        RAG-guided summarization: retrieves the most relevant chunks from the
        indexed corpus using broad seed queries, then runs map-reduce only on
        those chunks. Falls back to full-chunk summarization if fewer than 10
        chunks are in the index.

        Parameters
        ----------
        doc_name     : indexed document name
        searcher     : HybridSearcher instance
        top_k        : max total chunks to retrieve
        batch_size   : chunks per reduce batch
        file_hash    : stable hash for caching
        summary_cache: SummaryCache instance
        """
        if not doc_name:
            return "No document specified."

        # --- Cache read ---
        cache_key = f"summary_{file_hash}" if file_hash else None
        if cache_key and summary_cache:
            cached = summary_cache.get(cache_key)
            if cached:
                logger.info(f"Summary cache hit for '{doc_name}'")
                return cached

        # --- RAG retrieval from indexed corpus ---
        seen_ids: set = set()
        retrieved_texts: List[str] = []

        per_query = max(4, top_k // len(_SUMMARY_QUERIES))
        for query in _SUMMARY_QUERIES:
            try:
                results = searcher.search(query, limit=per_query, doc_filter=[doc_name])
                for r in results:
                    cid = r.get("chunk_id", r["text"][:40])
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        retrieved_texts.append(r["text"])
                        if len(retrieved_texts) >= top_k:
                            break
            except Exception as e:
                logger.warning(f"RAG retrieval error for summary query '{query}': {e}")
            if len(retrieved_texts) >= top_k:
                break

        # Fallback: not enough retrieved chunks — use stored chunks directly
        if len(retrieved_texts) < 10:
            logger.info(f"RAG retrieved only {len(retrieved_texts)} chunks for '{doc_name}'; "
                        "falling back to full chunk list.")
            retrieved_texts = searcher.get_chunks_for_doc(doc_name)

        if not retrieved_texts:
            return f"未在索引中找到《{doc_name}》的内容，请先完成索引后再生成摘要。"

        logger.info(f"[MAP-REDUCE] Summarizing {len(retrieved_texts)} RAG-retrieved chunks for '{doc_name}'")
        result = self.summarize_large_document(retrieved_texts, batch_size=batch_size)

        # --- Cache write ---
        if cache_key and summary_cache:
            summary_cache.put(cache_key, result)

        return result

    def summarize_large_document(self, texts: List[str], batch_size: int = 20) -> str:
        """
        Core Map-Reduce summarization over a list of text strings.
        Called by rag_summarize with pre-filtered RAG chunks.
        """
        if not texts:
            return "No text to summarize."

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

        # REDUCE step
        while len(intermediate_summaries) > 1:
            next_level: List[str] = []
            for i in range(0, len(intermediate_summaries), batch_size):
                batch = intermediate_summaries[i:i + batch_size]
                combined = "\n\n".join([f"[Segment {i + j + 1}]: {s}" for j, s in enumerate(batch)])
                reduced = self._invoke_reduce(combined)
                next_level.append(reduced)
            intermediate_summaries = next_level
            logger.info(f"  Reduce pass produced {len(intermediate_summaries)} summaries.")

        return intermediate_summaries[0] if intermediate_summaries else "Could not generate summary."

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
        Answer a question using retrieved context from the indexed corpus,
        the full conversation history, and an optional in-chat file attachment.

        Parameters
        ----------
        query          : the new user question
        context_chunks : retrieved dicts with 'text', 'doc_name', 'score' (from HybridSearcher)
        history        : list of {role, content} dicts — full prior turns (oldest first)
        attachment_text: text from a per-conversation chat attachment (not indexed)
        """
        if not context_chunks and not attachment_text:
            return "在已索引的书籍中未找到相关信息，请先上传并索引书籍，或上传附件进行提问。"

        history = history or []

        # Build book context from RAG results
        book_context = "\n\n".join([
            f"来自《{c.get('doc_name', '书籍')}》(相关度 {c.get('score', 0):.2f}):\n{c['text']}"
            for c in context_chunks
        ]) if context_chunks else ""

        # Compose messages
        messages = []

        # 1. System: role
        messages.append(SystemMessage(content=(
            "你是书籍专家（BookExpert），一位精通书籍内容的中文AI助手，能进行多轮深度对话。"
            "请仅根据提供的书籍索引摘录和用户附件用简体中文回答问题。"
            "若内容中未包含答案，请直接告知，不要编造信息。"
        )))

        # 2. System: book context from indexed corpus
        if book_context:
            messages.append(SystemMessage(content=f"--- 索引库相关摘录 ---\n{book_context}"))

        # 3. System: attachment (session-only, not indexed)
        if attachment_text:
            snippet = attachment_text[:12000]
            messages.append(SystemMessage(content=f"--- 📎 用户附件（仅本会话，未入书库）---\n{snippet}"))

        # 4. Full history
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
            f"Answering: {len(context_chunks)} RAG chunks, "
            f"{len(history)} history turns, attachment={'yes' if attachment_text else 'no'}"
        )
        return self._invoke_with_messages(messages)

    # ------------------------------------------------------------------
    # Rate-limited LLM calls
    # ------------------------------------------------------------------
    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_map(self, chunk: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content=(
                "你是一位书籍分析专家，请用简体中文简要总结以下书籍段落，"
                "重点提炼关键事件、人物和核心思想。"
            )),
            HumanMessage(content=chunk),
        ]
        return self.llm.invoke(messages).content

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_reduce(self, summaries: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content=(
                "你是一位书籍分析专家。你将收到一本书各段落的摘要。"
                "请用简体中文将它们整合成一份完整、连贯的全书摘要。"
            )),
            HumanMessage(content=summaries),
        ]
        return self.llm.invoke(messages).content

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_with_messages(self, messages: list) -> str:
        DEEPSEEK_LIMITER.wait()
        return self.llm.invoke(messages).content
