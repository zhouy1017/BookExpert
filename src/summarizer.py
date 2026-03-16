# Map-Reduce Summarization and QA via DeepSeek
import os
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

        # LCEL map chain: summarize a single chunk
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位书籍分析专家，请用简体中文简要总结以下书籍段落，重点提炼关键事件、人物和核心思想。"),
            ("human", "{chunk}")
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()

        # LCEL reduce chain: merge intermediate summaries into one final summary
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位书籍分析专家。你将收到一本书各段落的摘要。请用简体中文将它们整合成一份完整、连贯的全书摘要。"),
            ("human", "{summaries}")
        ])
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()

        # LCEL QA chain: answer a question given context excerpts
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
                "你是书籍专家（BookExpert），一位精通书籍内容的中文AI助手。"
                "请仅根据以下提供的书籍摘录用简体中文回答用户的问题。"
                "若摘录中未包含答案，请直接说明无法从书中找到相关信息。"),
            ("human", "--- 书籍摘录 ---\n{context}\n\n问题：{question}")
        ])
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def summarize_large_document(self, texts: List[str], batch_size: int = 20) -> str:
        """
        Runs a Map-Reduce summarization over a list of text chunks.
        - Map: each chunk is summarized individually.
        - Reduce: intermediate summaries are merged in passes into a final global summary.
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

        # REDUCE step — recursively reduce until we have a single summary
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

    def answer_question(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answers a question given retrieved context chunks using a LCEL QA chain."""
        if not context_chunks:
            return "No relevant context found in the book to answer your question."

        context_str = "\n\n".join([
            f"Excerpt from '{c.get('doc_name', 'book')}' (Score: {c.get('score', 0):.2f}):\n{c['text']}"
            for c in context_chunks
        ])

        logger.info(f"Answering question based on {len(context_chunks)} retrieved chunks.")
        return self._invoke_qa(context_str, query)

    # ------------------------------------------------------------------
    # Internal rate-limited + retry-wrapped LLM invocations
    # ------------------------------------------------------------------
    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_map(self, chunk: str) -> str:
        DEEPSEEK_LIMITER.wait()
        return self.map_chain.invoke({"chunk": chunk})

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_reduce(self, summaries: str) -> str:
        DEEPSEEK_LIMITER.wait()
        return self.reduce_chain.invoke({"summaries": summaries})

    @retry_on_rate_limit(max_attempts=6, wait_min=5, wait_max=90)
    def _invoke_qa(self, context: str, question: str) -> str:
        DEEPSEEK_LIMITER.wait()
        return self.qa_chain.invoke({"context": context, "question": question})
