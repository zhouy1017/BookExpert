"""
Book Review System
==================
Uses DeepSeek to generate a structured 100-point critic review for an indexed
book. Results are cached in SummaryCache under the key ``review_{file_hash}``.

Review JSON schema
------------------
{
    "overall_score": <int 0-100>,
    "dimensions": {
        "内容深度":   <int 0-20>,
        "文笔与表达": <int 0-20>,
        "结构与逻辑": <int 0-20>,
        "思想价值":   <int 0-20>,
        "可读性":     <int 0-20>
    },
    "strengths":      [<str>, ...],
    "weaknesses":     [<str>, ...],
    "critic_summary": <str>
}
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.rate_limiter import DEEPSEEK_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是一位资深文学评论家和书籍评审专家。你将收到一本书的部分段落摘录（代表性样本）。
请基于这些内容，对该书进行专业评估，并**严格**以如下 JSON 格式返回，不要输出任何 JSON 之外的内容：

{
  "overall_score": <整数，0-100>,
  "dimensions": {
    "内容深度":   <整数，0-20>,
    "文笔与表达": <整数，0-20>,
    "结构与逻辑": <整数，0-20>,
    "思想价值":   <整数，0-20>,
    "可读性":     <整数，0-20>
  },
  "strengths":      ["优点1", "优点2", "优点3"],
  "weaknesses":     ["不足1", "不足2"],
  "critic_summary": "2-4句话的完整评论总结，用简体中文书写。"
}

评分维度说明（各20分，合计100分）：
- 内容深度：主题的广度、深度与原创性
- 文笔与表达：语言运用、风格与表现力
- 结构与逻辑：章节编排、叙事/论证的逻辑性
- 思想价值：对读者的启发、影响与现实意义
- 可读性：流畅度、趣味性与吸引力
"""

_MAX_SAMPLE_CHUNKS = 30
_MAX_CHARS_PER_CHUNK = 800


class BookReviewer:
    def __init__(self):
        try:
            with open("d:/BookExpert/deepseek.apikey", "r", encoding="utf-8") as f:
                api_key = f.read().strip().rstrip(".")
                os.environ["DEEPSEEK_API_KEY"] = api_key
        except Exception as e:
            logger.error("Could not read deepseek.apikey")
            raise e

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.4,
        )

    def review_book(
        self,
        chunks: List[str],
        doc_name: str,
        file_hash: Optional[str] = None,
        summary_cache=None,
    ) -> Dict:
        """
        Generate a structured 100-point review for a book.

        Returns a dict matching the review JSON schema. Checks and writes
        SummaryCache when file_hash is provided.
        """
        cache_key = f"review_{file_hash}" if file_hash else None

        # --- Cache read ---
        if cache_key and summary_cache:
            cached = summary_cache.get(cache_key)
            if cached:
                logger.info(f"Review cache hit for {doc_name}")
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    logger.warning("Corrupt review cache entry; re-generating.")

        # --- Sample chunks evenly ---
        if len(chunks) <= _MAX_SAMPLE_CHUNKS:
            sample = chunks
        else:
            indices = sorted(random.sample(range(len(chunks)), _MAX_SAMPLE_CHUNKS))
            sample = [chunks[i] for i in indices]

        excerpts = "\n\n---\n\n".join(
            f"[摘录 {i+1}]\n{c[:_MAX_CHARS_PER_CHUNK]}" for i, c in enumerate(sample)
        )
        human_msg = f"书名：《{doc_name}》\n\n以下是书中的代表性段落：\n\n{excerpts}"

        # --- LLM call ---
        result_text = self._invoke_review(human_msg)

        # --- Parse JSON (with fallback) ---
        review = self._parse_review(result_text, doc_name)

        # --- Cache write ---
        if cache_key and summary_cache:
            summary_cache.put(cache_key, json.dumps(review, ensure_ascii=False))

        return review

    @retry_on_rate_limit(max_attempts=5, wait_min=5, wait_max=60)
    def _invoke_review(self, human_text: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]
        return self.llm.invoke(messages).content

    @staticmethod
    def _parse_review(text: str, doc_name: str) -> Dict:
        """Parse the LLM JSON output; return a safe fallback dict on error."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(cleaned)
            # Validate and coerce to expected shape
            dims = data.get("dimensions", {})
            return {
                "overall_score": int(data.get("overall_score", 0)),
                "dimensions": {
                    "内容深度":   int(dims.get("内容深度", 0)),
                    "文笔与表达": int(dims.get("文笔与表达", 0)),
                    "结构与逻辑": int(dims.get("结构与逻辑", 0)),
                    "思想价值":   int(dims.get("思想价值", 0)),
                    "可读性":     int(dims.get("可读性", 0)),
                },
                "strengths":      data.get("strengths", []),
                "weaknesses":     data.get("weaknesses", []),
                "critic_summary": data.get("critic_summary", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse review JSON for '{doc_name}': {e}\nRaw: {text[:300]}")
            return {
                "overall_score": 0,
                "dimensions": {k: 0 for k in ["内容深度", "文笔与表达", "结构与逻辑", "思想价值", "可读性"]},
                "strengths": [],
                "weaknesses": [],
                "critic_summary": f"书评生成失败，请重试。\n错误：{e}",
            }
