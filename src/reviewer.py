"""
Book Review System — RAG-first, professional critic quality
============================================================
Uses DeepSeek to generate a structured 100-point critic review for an indexed
book. Retrieves content exclusively from the indexed corpus via HybridSearcher
(never re-reads raw files). Scores are locked after first generation to ensure
consistency. Supports user-feedback-guided regeneration.

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
    "dimension_comments": {
        "内容深度":   "<analytical paragraph>",
        ...
    },
    "strengths":      [<str>, ...],
    "weaknesses":     [<str>, ...],
    "critic_essay":   "<4+ paragraph professional essay>"
}
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.rate_limiter import DEEPSEEK_LIMITER, retry_on_rate_limit

logger = logging.getLogger(__name__)

# Dimension seed queries — used to retrieve thematically diverse chunks via RAG
_DIMENSION_QUERIES = [
    "主题思想 核心观点 思想价值",
    "文笔风格 语言表达 写作技巧",
    "故事结构 章节安排 叙事逻辑",
    "人物塑造 角色发展 情节推进",
    "可读性 节奏感 阅读体验",
]
_CHUNKS_PER_QUERY = 8         # retrieve up to 8 chunks per seed query
_MAX_CHARS_PER_CHUNK = 600    # truncate each chunk to save tokens

_SYSTEM_PROMPT = """\
你是一位资深文学评论家，拥有深厚的中文文学和世界文学背景，曾为权威文学期刊撰稿。
你将收到一本书的代表性段落。请基于这些内容，以专业、严谨而富有洞察力的风格完成完整书评。

**严格**以如下 JSON 格式返回，不得输出任何 JSON 之外的内容：

{
  "overall_score": <整数，0-100>,
  "dimensions": {
    "内容深度":   <整数，0-20>,
    "文笔与表达": <整数，0-20>,
    "结构与逻辑": <整数，0-20>,
    "思想价值":   <整数，0-20>,
    "可读性":     <整数，0-20>
  },
  "dimension_comments": {
    "内容深度":   "<150字以上的专业分析段落，结合具体书中内容>",
    "文笔与表达": "<150字以上的专业分析段落>",
    "结构与逻辑": "<150字以上的专业分析段落>",
    "思想价值":   "<150字以上的专业分析段落>",
    "可读性":     "<150字以上的专业分析段落>"
  },
  "strengths":      ["具体优点1（需结合书中内容）", "具体优点2", "具体优点3"],
  "weaknesses":     ["具体不足1（需结合书中内容）", "具体不足2"],
  "critic_essay":   "<至少4段、每段不少于100字的完整书评文章，涵盖：文学背景与定位、主题与思想深度分析、叙事技巧与语言风格、文化与社会意义、对读者的建议>"
}

评分维度（各20分，合计100分）：
- 内容深度：主题广度、原创性、论述深度
- 文笔与表达：语言质量、风格独特性、表现力
- 结构与逻辑：章节编排、叙事/论证的内在逻辑
- 思想价值：对读者的启发性、现实意义、人文关怀
- 可读性：节奏感、趣味性、读者粘性
"""

_FEEDBACK_SYSTEM_PROMPT = """\
你是一位资深文学评论家。你之前已为某书打出了固定评分，现在需要根据读者反馈重新撰写评论文字。
注意：评分数字（overall_score 与 dimensions）已经锁定，你**不得修改评分数字**。
你只需基于读者的意见重新撰写 dimension_comments、strengths、weaknesses 和 critic_essay。

严格以 JSON 格式返回，字段与原格式完全相同。不得输出 JSON 以外的任何内容。
"""


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
            temperature=0.5,
        )

    # ------------------------------------------------------------------
    # RAG chunk retrieval from indexed corpus
    # ------------------------------------------------------------------
    def _retrieve_review_chunks(self, searcher: Any, doc_name: str) -> List[str]:
        """
        Pull thematically diverse chunks from the indexed corpus using
        dimension-targeted seed queries. Deduplicates by chunk_id.
        """
        seen_ids = set()
        chunks_text = []
        for query in _DIMENSION_QUERIES:
            try:
                results = searcher.search(query, limit=_CHUNKS_PER_QUERY, doc_filter=[doc_name])
                for r in results:
                    cid = r.get("chunk_id", r["text"][:40])
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        chunks_text.append(r["text"][:_MAX_CHARS_PER_CHUNK])
            except Exception as e:
                logger.warning(f"Review chunk retrieval error for query '{query}': {e}")
        logger.info(f"Retrieved {len(chunks_text)} unique chunks for review of '{doc_name}'")
        return chunks_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def review_book(
        self,
        searcher: Any,
        doc_name: str,
        file_hash: Optional[str] = None,
        summary_cache=None,
    ) -> Dict:
        """
        Generate a structured review. Retrieves content from the indexed corpus
        via searcher (RAG). Locks scores after first generation.
        """
        cache_key = f"review_{file_hash}" if file_hash else None

        # --- Full review cache read ---
        if cache_key and summary_cache:
            cached = summary_cache.get(cache_key)
            if cached:
                logger.info(f"Review cache hit for '{doc_name}'")
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    logger.warning("Corrupt review cache; re-generating.")

        # --- Retrieve chunks from indexed corpus ---
        chunks = self._retrieve_review_chunks(searcher, doc_name)
        if not chunks:
            # Fallback: try fetching raw stored chunks from SQLite
            chunks = searcher.get_chunks_for_doc(doc_name)[:30]

        if not chunks:
            return self._empty_review("索引中未找到该书的内容，请先完成索引后再生成书评。")

        # --- Check for locked scores ---
        locked_scores = None
        if file_hash and summary_cache:
            locked_scores = summary_cache.get_scores(file_hash)

        excerpts = self._format_excerpts(chunks)
        score_hint = ""
        if locked_scores:
            score_hint = (
                f"\n\n【注意】此书的评分已经确定，请严格使用以下评分，不得修改：\n"
                f"综合评分={locked_scores['overall_score']}，"
                + "，".join(f"{k}={v}" for k, v in locked_scores.get("dimensions", {}).items())
            )

        human_msg = f"书名：《{doc_name}》{score_hint}\n\n以下是从索引库检索的代表性段落：\n\n{excerpts}"
        result_text = self._invoke_review(_SYSTEM_PROMPT, human_msg)
        review = self._parse_review(result_text, doc_name)

        # --- Lock scores on first successful generation ---
        if file_hash and summary_cache and not locked_scores:
            summary_cache.save_scores(file_hash, {
                "overall_score": review["overall_score"],
                "dimensions":    review["dimensions"],
            })

        # --- Cache full review ---
        if cache_key and summary_cache:
            summary_cache.put(cache_key, json.dumps(review, ensure_ascii=False))

        return review

    def regenerate_with_feedback(
        self,
        searcher: Any,
        doc_name: str,
        file_hash: str,
        feedback: Dict,
        scoring_prefs: str,
        summary_cache=None,
    ) -> Dict:
        """
        Re-generate critic text (not scores) based on user feedback.
        Scores are always loaded from cache and kept locked.
        """
        locked_scores = None
        if summary_cache:
            locked_scores = summary_cache.get_scores(file_hash)

        if not locked_scores:
            # First time — do a full generation to establish scores
            return self.review_book(searcher, doc_name, file_hash, summary_cache)

        chunks = self._retrieve_review_chunks(searcher, doc_name)
        if not chunks:
            chunks = searcher.get_chunks_for_doc(doc_name)[:30]

        excerpts = self._format_excerpts(chunks)

        # Build feedback section
        score_overrides   = feedback.get("score_overrides", {})
        extra_strengths   = feedback.get("extra_strengths", [])
        extra_weaknesses  = feedback.get("extra_weaknesses", [])
        user_comments     = feedback.get("user_comments", "")

        feedback_text = (
            f"书名：《{doc_name}》\n\n"
            f"【锁定评分 — 不得修改】\n"
            f"综合评分={locked_scores['overall_score']}，"
            + "，".join(f"{k}={v}" for k, v in locked_scores["dimensions"].items())
        )

        if score_overrides:
            feedback_text += f"\n\n【读者调整评分】\n{json.dumps(score_overrides, ensure_ascii=False)}"
        if extra_strengths:
            feedback_text += f"\n\n【读者补充优点】\n" + "\n".join(f"- {s}" for s in extra_strengths)
        if extra_weaknesses:
            feedback_text += f"\n\n【读者补充不足】\n" + "\n".join(f"- {w}" for w in extra_weaknesses)
        if user_comments:
            feedback_text += f"\n\n【读者评论与意见】\n{user_comments}"
        if scoring_prefs:
            feedback_text += f"\n\n【读者评分偏好（用于未来评分参考）】\n{scoring_prefs}"

        feedback_text += f"\n\n【书籍摘录】\n{excerpts}"

        result_text = self._invoke_review(_FEEDBACK_SYSTEM_PROMPT, feedback_text)
        review = self._parse_review(result_text, doc_name)

        # Always restore locked scores
        review["overall_score"] = locked_scores["overall_score"]
        review["dimensions"]    = locked_scores["dimensions"]

        # Apply user score overrides on top
        if "overall_score" in score_overrides:
            review["overall_score"] = int(score_overrides["overall_score"])
        for dim, val in score_overrides.get("dimensions", {}).items():
            if dim in review["dimensions"]:
                review["dimensions"][dim] = int(val)

        # Cache the updated review
        if summary_cache:
            cache_key = f"review_{file_hash}"
            summary_cache.put(cache_key, json.dumps(review, ensure_ascii=False))

        return review

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_excerpts(chunks: List[str]) -> str:
        return "\n\n---\n\n".join(
            f"[摘录 {i+1}]\n{c}" for i, c in enumerate(chunks)
        )

    @retry_on_rate_limit(max_attempts=5, wait_min=5, wait_max=90)
    def _invoke_review(self, system_prompt: str, human_text: str) -> str:
        DEEPSEEK_LIMITER.wait()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_text),
        ]
        return self.llm.invoke(messages).content

    @staticmethod
    def _parse_review(text: str, doc_name: str) -> Dict:
        """Parse LLM JSON; return safe fallback on error."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            end = -1 if lines[-1].strip() in ("```", "```json") else len(lines)
            cleaned = "\n".join(lines[1:end])
        try:
            data = json.loads(cleaned)
            dims = data.get("dimensions", {})
            dim_comments = data.get("dimension_comments", {})
            dim_keys = ["内容深度", "文笔与表达", "结构与逻辑", "思想价值", "可读性"]
            return {
                "overall_score":      int(data.get("overall_score", 0)),
                "dimensions":         {k: int(dims.get(k, 0)) for k in dim_keys},
                "dimension_comments": {k: dim_comments.get(k, "") for k in dim_keys},
                "strengths":          data.get("strengths", []),
                "weaknesses":         data.get("weaknesses", []),
                "critic_essay":       data.get("critic_essay", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse review JSON for '{doc_name}': {e}\nRaw: {text[:400]}")
            return BookReviewer._empty_review(f"书评解析失败，请重试。\n错误：{e}")

    @staticmethod
    def _empty_review(msg: str = "") -> Dict:
        dim_keys = ["内容深度", "文笔与表达", "结构与逻辑", "思想价值", "可读性"]
        return {
            "overall_score":      0,
            "dimensions":         {k: 0 for k in dim_keys},
            "dimension_comments": {k: "" for k in dim_keys},
            "strengths":          [],
            "weaknesses":         [],
            "critic_essay":       msg,
        }
