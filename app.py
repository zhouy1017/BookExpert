import hashlib
import os
import tempfile
import uuid
from typing import Dict, List, Optional

import streamlit as st

from src.cache import EmbeddingCache, SummaryCache
from src.chunking import ChineseTextSplitter
from src.extractors import DocumentProcessor
from src.reviewer import BookReviewer
from src.search import HybridSearcher
from src.summarizer import BookSummarizer

st.set_page_config(page_title="书籍专家 BookExpert", layout="wide", page_icon="📚")

# ─────────────────────────────────────────
# Cached singletons
# ─────────────────────────────────────────
@st.cache_resource
def get_processor():    return DocumentProcessor()
@st.cache_resource
def get_chunker():      return ChineseTextSplitter()
@st.cache_resource
def get_searcher():     return HybridSearcher()
@st.cache_resource
def get_summarizer():   return BookSummarizer()
@st.cache_resource
def get_reviewer():     return BookReviewer()
@st.cache_resource
def get_emb_cache():    return EmbeddingCache()
@st.cache_resource
def get_sum_cache():    return SummaryCache()


# ─────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────
def _init_session():
    """Initialise all session-state keys on first run."""
    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.conversations = {
            first_id: {
                "name": "对话 1",
                "history": [],          # list of {role, content}
                "scope_docs": [],       # [] means all docs
                "attachment_text": "",  # text of uploaded chat file
                "attachment_name": "",
            }
        }
        st.session_state.active_conv_id = first_id

    if "conv_counter" not in st.session_state:
        st.session_state.conv_counter = 1


def _active_conv() -> Dict:
    return st.session_state.conversations[st.session_state.active_conv_id]


def _new_conversation():
    st.session_state.conv_counter += 1
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {
        "name": f"对话 {st.session_state.conv_counter}",
        "history": [],
        "scope_docs": [],
        "attachment_text": "",
        "attachment_name": "",
    }
    st.session_state.active_conv_id = cid


def _delete_conversation(cid: str):
    convs = st.session_state.conversations
    if len(convs) == 1:
        return  # keep at least one
    del convs[cid]
    if st.session_state.active_conv_id == cid:
        st.session_state.active_conv_id = next(iter(convs))


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_tmp(uploaded_file) -> str:
    """Save uploaded file to a temp path, extract text, clean up."""
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return get_processor().extract_text(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _build_export_md(conv: Dict, last_results: List) -> str:
    """Build a Markdown export of the last Q&A turn."""
    history = conv["history"]
    if len(history) < 2:
        return "# 未找到可导出的对话内容"
    question = history[-2]["content"]
    answer   = history[-1]["content"]
    lines = [f"# BookExpert — Q&A 导出\n", f"## 问题\n{question}\n", f"## AI 回答\n{answer}\n"]
    if last_results:
        lines.append("## 引用书籍摘录\n")
        for i, r in enumerate(last_results):
            lines.append(
                f"### 摘录 {i+1} — 来源：{r['doc_name']} (相关度 {r['score']:.4f})\n{r['text']}\n"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────
# Main app
# ─────────────────────────────────────────
def main():
    _init_session()

    processor  = get_processor()
    chunker    = get_chunker()
    searcher   = get_searcher()
    summarizer = get_summarizer()
    reviewer   = get_reviewer()
    emb_cache  = get_emb_cache()
    sum_cache  = get_sum_cache()

    # ══════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════
    with st.sidebar:
        st.title("📚 书籍专家")
        st.caption("中文书籍智能问答系统")
        st.divider()

        # ── 1. Book Library ──────────────────
        st.header("📂 书库管理")
        uploaded_file = st.file_uploader(
            "上传书籍（PDF / DOCX / DOC）",
            type=["pdf", "docx", "doc"],
            key="book_uploader",
        )
        if uploaded_file and st.button("⚙️ 处理并建立索引"):
            file_bytes = uploaded_file.getvalue()
            file_hash  = _sha256(file_bytes)

            # --- Extract ---
            with st.spinner("正在提取文本…"):
                try:
                    suffix = "." + uploaded_file.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    text = processor.extract_text(tmp_path)
                    st.success(f"提取成功！共 {len(text):,} 个字符。")
                except Exception as e:
                    st.error(f"文本提取失败：{e}")
                    return
                finally:
                    try: os.remove(tmp_path)
                    except: pass

            # --- Chunk ---
            with st.spinner("正在分块…"):
                chunks = chunker.split_text(text)
                st.success(f"生成 {len(chunks)} 个文本块。")

            # --- Embed & Index with progress bar ---
            st.write("⚙️ 正在向量化和索引…")
            prog_bar = st.progress(0, text="初始化…")
            status_txt = st.empty()

            def on_progress(done: int, total: int):
                pct = done / total if total else 1.0
                prog_bar.progress(pct, text=f"已处理 {done}/{total} 块")
                status_txt.caption(f"{'✅ 缓存命中' if done == total and pct < 1.01 else ''}")

            try:
                searcher.add_documents(
                    chunks,
                    doc_name=uploaded_file.name,
                    file_hash=file_hash,
                    embedding_cache=emb_cache,
                    progress_callback=on_progress,
                )
                prog_bar.progress(1.0, text="完成！")
                # Check if fully from cache
                cached_count = sum(
                    1 for i in range(len(chunks)) if emb_cache.get(file_hash, i) is not None
                )
                if cached_count == len(chunks):
                    st.info("✅ 已使用本地嵌入缓存，无需重新计算向量。")
                st.success("✅ 索引建立完成！")
            except Exception as e:
                st.error(f"索引失败：{e}")

        st.divider()

        # ── 2. Indexed Document List ──────────
        st.header("📚 已索引文档")
        indexed_docs = searcher.get_indexed_documents()
        if not indexed_docs:
            st.caption("暂无已索引文档。")
        else:
            for doc in indexed_docs:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 **{doc}**")
                if col2.button("🗑️", key=f"del_{doc}", help=f"删除 {doc}"):
                    searcher.delete_document(doc)
                    st.rerun()

        st.divider()

        # ── 3. Conversation List ──────────────
        st.header("💬 对话管理")
        if st.button("＋ 新对话", use_container_width=True):
            _new_conversation()
            st.rerun()

        for cid, conv in list(st.session_state.conversations.items()):
            is_active = cid == st.session_state.active_conv_id
            c1, c2 = st.columns([5, 1])
            label = f"{'▶ ' if is_active else ''}{conv['name']}"
            if c1.button(label, key=f"sel_{cid}", use_container_width=True):
                st.session_state.active_conv_id = cid
                st.rerun()
            if c2.button("✕", key=f"dconv_{cid}", help="删除该对话"):
                _delete_conversation(cid)
                st.rerun()

    # ══════════════════════════════════════
    # MAIN TABS
    # ══════════════════════════════════════
    tab_chat, tab_summary, tab_review = st.tabs(["💬 智能问答", "📝 全书摘要", "⭐ 书评"])

    # ──────────────────────────────────────
    # TAB 1: Multi-turn Chat
    # ──────────────────────────────────────
    with tab_chat:
        conv = _active_conv()
        indexed_docs = searcher.get_indexed_documents()

        # Conversation header
        col_title, col_clear = st.columns([6, 1])
        col_title.subheader(f"💬 {conv['name']}")
        if col_clear.button("🗑️ 清空", key="clear_conv"):
            conv["history"] = []
            conv["attachment_text"] = ""
            conv["attachment_name"] = ""
            st.rerun()

        # Scope selector
        if indexed_docs:
            selected_scope = st.multiselect(
                "📖 对话范围（留空 = 全部书籍）",
                options=indexed_docs,
                default=conv.get("scope_docs", []),
                key=f"scope_{st.session_state.active_conv_id}",
            )
            conv["scope_docs"] = selected_scope
        else:
            selected_scope = []

        # Chat attachment
        with st.expander("📎 上传附件（仅本对话有效，不入书库）", expanded=bool(conv["attachment_name"])):
            attach_file = st.file_uploader(
                "选择文件",
                type=["pdf", "docx", "doc", "txt"],
                key=f"attach_{st.session_state.active_conv_id}",
            )
            if attach_file:
                if attach_file.name != conv.get("attachment_name"):
                    with st.spinner("正在提取附件内容…"):
                        try:
                            conv["attachment_text"] = _extract_tmp(attach_file)
                            conv["attachment_name"] = attach_file.name
                            st.success(f"附件 '{attach_file.name}' 已读取（{len(conv['attachment_text']):,} 字符）")
                        except Exception as e:
                            st.error(f"附件提取失败：{e}")
            if conv["attachment_name"]:
                st.info(f"当前附件：📎 {conv['attachment_name']}")
                if st.button("移除附件", key="rm_attach"):
                    conv["attachment_text"] = ""
                    conv["attachment_name"] = ""
                    st.rerun()

        st.divider()

        # Render conversation history
        for msg in conv["history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Keep last search results for export
        last_results_key = f"last_results_{st.session_state.active_conv_id}"
        if last_results_key not in st.session_state:
            st.session_state[last_results_key] = []

        # Chat input
        user_input = st.chat_input("向书籍专家提问…")
        if user_input:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)

            doc_filter = conv["scope_docs"] if conv["scope_docs"] else None

            # Retrieve relevant chunks
            with st.spinner("检索相关段落…"):
                results = searcher.search(user_input, limit=5, doc_filter=doc_filter)
                st.session_state[last_results_key] = results

            # Show retrieved excerpts in an expander
            if results:
                with st.expander(f"📚 找到 {len(results)} 条相关段落", expanded=False):
                    for i, r in enumerate(results):
                        st.markdown(f"**段落 {i+1}** — 来源：`{r['doc_name']}` (相关度 `{r['score']:.4f}`)")
                        st.caption(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("DeepSeek 正在思考…"):
                    answer = summarizer.answer_question(
                        query=user_input,
                        context_chunks=results,
                        history=conv["history"],
                        attachment_text=conv.get("attachment_text", ""),
                    )
                st.markdown(answer)

                # Export button under the answer
                export_md = _build_export_md(
                    {**conv, "history": conv["history"] + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": answer},
                    ]},
                    results
                )
                st.download_button(
                    "📥 导出本次问答（Markdown）",
                    data=export_md,
                    file_name="bookexpert_qa.md",
                    mime="text/markdown",
                    key=f"export_{uuid.uuid4()}",
                )

            # Persist to history
            conv["history"].append({"role": "user",      "content": user_input})
            conv["history"].append({"role": "assistant", "content": answer})

    # ──────────────────────────────────────
    # TAB 2: Per-Document Summary
    # ──────────────────────────────────────
    with tab_summary:
        st.header("📝 全书摘要")
        indexed_docs = searcher.get_indexed_documents()

        if not indexed_docs:
            st.info("请先在侧边栏上传并索引书籍文件。")
        else:
            selected_doc = st.selectbox("选择要摘要的书籍", options=indexed_docs, key="summary_doc_select")
            st.warning("⚠️ 首次生成摘要需要一定时间；结果将缓存供下次秒级复用。")

            # We need file_hash for summary cache; we approximate by hashing doc_name
            # (real hash stored during indexing is not persisted here; we use a stable key)
            cache_key_doc = hashlib.sha256(selected_doc.encode()).hexdigest()

            # Check summary cache
            cached_summary = sum_cache.get(f"summary_{cache_key_doc}")
            if cached_summary:
                st.info("✅ 已使用本地摘要缓存。")
                st.markdown("### 全书摘要")
                st.write(cached_summary)
            else:
                if st.button("📖 生成全书摘要"):
                    chunks = searcher.get_chunks_for_doc(selected_doc)
                    if not chunks:
                        st.error("未找到该文档的文本块。")
                    else:
                        with st.spinner(f"正在为《{selected_doc}》生成摘要（Map-Reduce）…"):
                            summary = summarizer.summarize_large_document(
                                chunks,
                                file_hash=cache_key_doc,
                                summary_cache=sum_cache,
                            )
                        st.markdown("### 全书摘要")
                        st.write(summary)

    # ──────────────────────────────────────
    # TAB 3: Book Review
    # ──────────────────────────────────────
    with tab_review:
        st.header("⭐ 书评系统")
        indexed_docs = searcher.get_indexed_documents()

        if not indexed_docs:
            st.info("请先在侧边栏上传并索引书籍文件。")
        else:
            selected_doc = st.selectbox("选择要评分的书籍", options=indexed_docs, key="review_doc_select")
            cache_key_doc = hashlib.sha256(selected_doc.encode()).hexdigest()
            cached_review_str = sum_cache.get(f"review_{cache_key_doc}")

            review_data = None
            if cached_review_str:
                import json
                try:
                    review_data = json.loads(cached_review_str)
                    st.info("✅ 已使用本地书评缓存。")
                except Exception:
                    pass

            btn_label = "🔄 重新生成书评" if review_data else "⭐ 生成书评"
            if st.button(btn_label):
                chunks = searcher.get_chunks_for_doc(selected_doc)
                if not chunks:
                    st.error("未找到该文档的文本块。")
                else:
                    with st.spinner(f"DeepSeek 正在评审《{selected_doc}》…"):
                        review_data = reviewer.review_book(
                            chunks,
                            doc_name=selected_doc,
                            file_hash=cache_key_doc,
                            summary_cache=sum_cache,
                        )

            if review_data:
                score = review_data.get("overall_score", 0)
                dims  = review_data.get("dimensions", {})

                # --- Score badge ---
                score_color = "#2ecc71" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:24px 0 8px'>
                      <span style='font-size:80px; font-weight:900; color:{score_color};'>{score}</span>
                      <span style='font-size:36px; color:#aaa;'> / 100</span>
                      <p style='font-size:18px; color:#888; margin-top:4px;'>综合评分</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --- Dimension metrics ---
                st.subheader("📊 维度评分（各满分 20 分）")
                dim_cols = st.columns(len(dims))
                for col, (dim_name, dim_score) in zip(dim_cols, dims.items()):
                    col.metric(label=dim_name, value=f"{dim_score} / 20")

                st.divider()

                # --- Strengths & Weaknesses ---
                col_s, col_w = st.columns(2)
                with col_s:
                    st.subheader("✅ 优点")
                    for s in review_data.get("strengths", []):
                        st.markdown(f"- {s}")
                with col_w:
                    st.subheader("⚠️ 不足")
                    for w in review_data.get("weaknesses", []):
                        st.markdown(f"- {w}")

                st.divider()

                # --- Critic Summary ---
                st.subheader("📝 评论总结")
                st.info(review_data.get("critic_summary", ""))

                # --- Export ---
                import json
                st.download_button(
                    "📥 导出书评（JSON）",
                    data=json.dumps(review_data, ensure_ascii=False, indent=2),
                    file_name=f"review_{selected_doc}.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
