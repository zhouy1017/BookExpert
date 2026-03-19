# Map-Reduce Summarization and QA via LLMProvider (Gemini primary, DeepSeek fallback)
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm_provider import get_llm_provider

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


class BookSummarizer:
    def __init__(self):
        self._provider = get_llm_provider(temperature=0.3)

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
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        RAG-guided summarization: retrieves the most relevant chunks from the
        indexed corpus using broad seed queries, then runs map-reduce only on
        those chunks. Falls back to full-chunk summarization if fewer than 10
        chunks are in the index.

        progress_callback(step, total, label) is called at each stage so
        the caller can drive a live progress bar.
        """
        if not doc_name:
            return "No document specified."

        # --- Cache read ---
        cache_key = f"summary_{file_hash}" if file_hash else None
        if cache_key and summary_cache:
            cached = summary_cache.get(cache_key)
            if cached:
                logger.info(f"Summary cache hit for '{doc_name}'")
                if progress_callback:
                    progress_callback(1, 1, "摘要缓存命中 ✅")
                return cached

        n_queries = len(_SUMMARY_QUERIES)
        if progress_callback:
            progress_callback(0, n_queries + 1, f"RAG 检索中… (0/{n_queries} 词条)")

        # --- RAG retrieval from indexed corpus ---
        seen_ids: set = set()
        retrieved_texts: List[str] = []
        per_query = max(4, top_k // n_queries)

        for qi, query in enumerate(_SUMMARY_QUERIES):
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
            if progress_callback:
                progress_callback(
                    qi + 1, n_queries + 1,
                    f"RAG 检索… ({qi+1}/{n_queries} 词条，已获取 {len(retrieved_texts)} 段落)"
                )
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
        result = self.summarize_large_document(
            retrieved_texts,
            batch_size=batch_size,
            progress_callback=progress_callback,
            rag_step_offset=n_queries,
        )

        # --- Cache write ---
        if cache_key and summary_cache:
            summary_cache.put(cache_key, result)

        return result

    def summarize_large_document(
        self,
        texts: List[str],
        batch_size: int = 20,
        progress_callback: Optional[Callable] = None,
        rag_step_offset: int = 0,
    ) -> str:
        """
        Core Map-Reduce summarization over a list of text strings.
        Called by rag_summarize with pre-filtered RAG chunks.
        progress_callback(step, total, label) is optional; rag_step_offset
        accounts for already-completed RAG retrieval steps.
        """
        if not texts:
            return "No text to summarize."

        n_map = len(texts)
        total_steps = rag_step_offset + n_map + 1  # +1 for reduce

        logger.info(f"[MAP] Summarizing {n_map} chunks in batches of {batch_size}...")

        # MAP step
        intermediate_summaries: List[str] = []
        for i, chunk in enumerate(texts):
            truncated = chunk[:_MAX_CHUNK_CHARS]
            summary   = self._invoke_map(truncated)
            intermediate_summaries.append(summary)
            if progress_callback:
                progress_callback(
                    rag_step_offset + i + 1,
                    total_steps,
                    f"Map 阶段: 正在提炼第 {i+1}/{n_map} 段落…"
                )

        logger.info(f"[REDUCE] Reducing {len(intermediate_summaries)} intermediate summaries...")
        if progress_callback:
            progress_callback(
                rag_step_offset + n_map,
                total_steps,
                f"Reduce 阶段: 合并 {len(intermediate_summaries)} 个提炼…"
            )

        # REDUCE step
        while len(intermediate_summaries) > 1:
            next_level: List[str] = []
            for i in range(0, len(intermediate_summaries), batch_size):
                batch    = intermediate_summaries[i:i + batch_size]
                combined = "\n\n".join([f"[Segment {i + j + 1}]: {s}" for j, s in enumerate(batch)])
                reduced  = self._invoke_reduce(combined)
                next_level.append(reduced)
            intermediate_summaries = next_level
            logger.info(f"  Reduce pass produced {len(intermediate_summaries)} summaries.")

        if progress_callback:
            progress_callback(total_steps, total_steps, "摘要完成 ✅")

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
        """
        if not context_chunks and not attachment_text:
            return "在已索引的书籍中未找到相关信息，请先上传并索引书籍，或上传附件进行提问。"

        history = history or []

        book_context = "\n\n".join([
            f"来自《{c.get('doc_name', '书籍')}》(相关度 {c.get('score', 0):.2f}):\n{c['text']}"
            for c in context_chunks
        ]) if context_chunks else ""

        messages = []
        messages.append(SystemMessage(content=(
            "你是书籍专家（BookExpert），一位精通书籍内容的中文AI助手，能进行多轮深度对话。"
            "请仅根据提供的书籍索引摘录和用户附件用简体中文回答问题。"
            "若内容中未包含答案，请直接告知，不要编造信息。"
        )))

        if book_context:
            messages.append(SystemMessage(content=f"--- 索引库相关摘录 ---\n{book_context}"))

        if attachment_text:
            snippet = attachment_text[:12000]
            messages.append(SystemMessage(content=f"--- 📎 用户附件（仅本会话，未入书库）---\n{snippet}"))

        for turn in history:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))

        logger.info(
            f"Answering: {len(context_chunks)} RAG chunks, "
            f"{len(history)} history turns, attachment={'yes' if attachment_text else 'no'}"
        )
        return self._provider.invoke(messages).content

    # ------------------------------------------------------------------
    # LLM helper calls (via LLMProvider — handles Gemini → DeepSeek fallback)
    # ------------------------------------------------------------------
    def _invoke_map(self, chunk: str) -> str:
        messages = [
            SystemMessage(content=(
                "你是一位书籍分析专家，请用简体中文简要总结以下书籍段落，"
                "重点提炼关键事件、人物和核心思想。"
            )),
            HumanMessage(content=chunk),
        ]
        return self._provider.invoke(messages).content

    def _invoke_reduce(self, summaries: str) -> str:
        messages = [
            SystemMessage(content=(
                "你是一位书籍分析专家。你将收到一本书各段落的摘要。"
                "请用简体中文将它们整合成一份完整、连贯的全书摘要。"
            )),
            HumanMessage(content=summaries),
        ]
        return self._provider.invoke(messages).content
