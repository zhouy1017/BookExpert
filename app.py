import hashlib
import json
import os
import tempfile
import uuid
from typing import Dict, List, Optional

import streamlit as st

import src.indexing_queue as iq
from src.cache import EmbeddingCache, FeedbackCache, SummaryCache
from src.chunking import ChineseTextSplitter
from src.extractors import DocumentProcessor
from src.rate_limiter import DEEPSEEK_LIMITER, GEMINI_EMBEDDING_LIMITER, is_quota_error
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
@st.cache_resource
def get_fb_cache():     return FeedbackCache()


# ─────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────
def _init_session():
    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.conversations = {
            first_id: {
                "name": "对话 1",
                "history": [],
                "scope_docs": [],
                "attachment_text": "",
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
        return
    del convs[cid]
    if st.session_state.active_conv_id == cid:
        st.session_state.active_conv_id = next(iter(convs))


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_tmp(uploaded_file) -> str:
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
    return ""



def _doc_hash(doc_name: str) -> str:
    """Stable hash for a doc_name — used for summary/review cache keys."""
    return hashlib.sha256(doc_name.encode()).hexdigest()


# ─────────────────────────────────────────
# Quota status toolbar
# ─────────────────────────────────────────
def _color_for_pct(pct: float) -> str:
    if pct >= 0.85: return "#e74c3c"
    if pct >= 0.60: return "#f39c12"
    return "#2ecc71"


def _render_quota_toolbar():
    """Render a compact API quota status panel in the sidebar."""
    g_status = GEMINI_EMBEDDING_LIMITER.get_status()
    d_status  = DEEPSEEK_LIMITER.get_status()

    with st.expander("📊 API 额度状态", expanded=False):
        # ── Gemini Embedding ───────────────────────────────────────
        st.caption("**Gemini Embedding（嵌入模型）**")
        for label, model_s in [("主模型 001", g_status["primary"]), ("备用模型 002", g_status["fallback"])]:
            rpm_pct = 1 - model_s["rpm_remaining"] / max(model_s["rpm_limit"], 1)
            tpm_pct = 1 - model_s["tpm_remaining"] / max(model_s["tpm_limit"], 1)
            rpd_pct = 1 - model_s["rpd_remaining"] / max(model_s["rpd_limit"], 1)
            worst   = max(rpm_pct, tpm_pct, rpd_pct)
            icon    = "🔴" if worst >= 0.85 else ("🟡" if worst >= 0.60 else "🟢")
            status_tag = " 🚫 已耗尽" if model_s["exhausted"] else ""
            st.markdown(
                f"{icon} **{label}**{status_tag}  "
                f"RPM {model_s['rpm_remaining']}/{model_s['rpm_limit']} · "
                f"TPM {model_s['tpm_remaining']:,}/{model_s['tpm_limit']:,} · "
                f"RPD {model_s['rpd_remaining']}/{model_s['rpd_limit']}"
            )
            st.progress(min(worst, 1.0))

        if g_status["both_exhausted"]:
            st.error("⚠️ 两个嵌入模型的配额均已耗尽，索引任务将暂停等待配额重置。")

        st.divider()

        # ── DeepSeek LLM ───────────────────────────────────────────
        st.caption("**DeepSeek LLM（对话模型）**")
        rpm_used = d_status["rpm_used"]
        rpm_lim  = d_status["rpm_limit"]
        rpm_rem  = d_status["rpm_remaining"]
        pct      = d_status["pct_used"]
        icon     = "🔴" if pct >= 0.85 else ("🟡" if pct >= 0.60 else "🟢")
        st.markdown(f"{icon} **RPM** {rpm_rem}/{rpm_lim} 剩余（已用 {rpm_used}）")
        st.progress(min(pct, 1.0))



def _build_export_md(question: str, answer: str, results: List) -> str:
    lines = ["# BookExpert — Q&A 导出\n", f"## 问题\n{question}\n", f"## AI 回答\n{answer}\n"]
    if results:
        lines.append("## 引用书籍摘录\n")
        for i, r in enumerate(results):
            lines.append(
                f"### 摘录 {i+1} — 来源：{r['doc_name']} (相关度 {r['score']:.4f})\n{r['text']}\n"
            )
    return "\n".join(lines)


def _render_indexing_queue_sidebar():
    """Render the live indexing queue progress in the sidebar."""
    progress = iq.get_progress()
    if not progress:
        return

    active = {k: v for k, v in progress.items() if v["status"] in ("queued", "indexing")}
    done   = {k: v for k, v in progress.items() if v["status"] == "done"}
    errors = {k: v for k, v in progress.items() if v["status"] == "error"}

    if active or done or errors:
        st.subheader("⚙️ 索引队列")

    for doc_name, info in active.items():
        label = f"{'🔄' if info['status'] == 'indexing' else '⏳'} {doc_name}"
        st.markdown(f"**{label}**")
        pct = info.get("pct", 0.0)
        st.progress(pct, text=f"{info['done']}/{info['total']} 块")

    for doc_name, info in done.items():
        st.success(f"✅ {doc_name} — 索引完成")

    for doc_name, info in errors.items():
        st.error(f"❌ {doc_name}: {info.get('error', '未知错误')}")

    if done or errors:
        if st.button("清除已完成记录", key="clear_q"):
            iq.clear_completed()
            st.rerun()


# ─────────────────────────────────────────
# Main app
# ─────────────────────────────────────────
def main():
    _init_session()

    # Auto-refresh every 2s when indexing is active (safe polling)
    if iq.any_active():
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, key="idx_refresh")
        except ImportError:
            pass  # graceful degradation if package not installed

    searcher   = get_searcher()
    summarizer = get_summarizer()
    reviewer   = get_reviewer()
    emb_cache  = get_emb_cache()
    sum_cache  = get_sum_cache()
    fb_cache   = get_fb_cache()

    # ══════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════
    with st.sidebar:
        st.title("📚 书籍专家")
        st.caption("中文书籍智能问答系统")
        st.divider()
        _render_quota_toolbar()

        # ── Book Library ──────────────────
        st.header("📂 书库管理")
        uploaded_file = st.file_uploader(
            "上传书籍（PDF / DOCX / DOC）",
            type=["pdf", "docx", "doc"],
            key="book_uploader",
        )
        if uploaded_file and st.button("⚙️ 加入索引队列"):
            file_bytes = uploaded_file.getvalue()
            file_hash  = _sha256(file_bytes)

            # Check full embedding cache hit
            with st.spinner("正在分块..."):
                try:
                    suffix = "." + uploaded_file.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    text = get_processor().extract_text(tmp_path)
                    chunks = get_chunker().split_text(text)
                    st.success(f"已分为 {len(chunks)} 块，加入后台索引队列…")
                except Exception as e:
                    st.error(f"提取失败：{e}")
                    chunks = []
                finally:
                    try: os.remove(tmp_path)
                    except: pass

            if chunks:
                all_cached = emb_cache.has_all(file_hash, len(chunks))
                if all_cached and uploaded_file.name in searcher.get_indexed_documents():
                    st.info("✅ 此书已索引且向量缓存完整，无需重新处理。")
                else:
                    iq.enqueue(
                        doc_name=uploaded_file.name,
                        chunks=chunks,
                        file_hash=file_hash,
                        searcher=searcher,
                        embedding_cache=emb_cache,
                    )
                    st.info(f"📥 《{uploaded_file.name}》已加入索引队列。")
                    st.rerun()

        st.divider()

        # ── Indexing Queue Progress ───────
        _render_indexing_queue_sidebar()

        # ── Indexed Document List ─────────
        st.header("📚 已索引文档")
        indexed_docs = searcher.get_indexed_documents()
        if not indexed_docs:
            st.caption("暂无已索引文档。")
        else:
            for doc in indexed_docs:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 {doc}")
                if col2.button("🗑️", key=f"del_{doc}", help=f"删除 {doc}"):
                    searcher.delete_document(doc)
                    st.rerun()

        st.divider()

        # ── Conversation List ─────────────
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
    # TAB 1: Multi-Turn Chat
    # ──────────────────────────────────────
    with tab_chat:
        conv = _active_conv()
        indexed_docs = searcher.get_indexed_documents()
        indexing_active = iq.any_active()

        col_title, col_clear = st.columns([6, 1])
        col_title.subheader(f"💬 {conv['name']}")
        if col_clear.button("🗑️ 清空", key="clear_conv"):
            conv["history"] = []
            conv["attachment_text"] = ""
            conv["attachment_name"] = ""
            st.rerun()

        # Guard rail: no docs indexed
        if not indexed_docs and not conv.get("attachment_text"):
            if indexing_active:
                st.warning("📥 正在建立索引，索引完成后即可开始问答。")
            else:
                st.info("💡 请先在左侧书库管理中上传并索引书籍，或上传附件后直接提问。")

        # Scope selector
        if indexed_docs:
            selected_scope = st.multiselect(
                "📖 对话范围（留空 = 全部书籍）",
                options=indexed_docs,
                default=[d for d in conv.get("scope_docs", []) if d in indexed_docs],
                key=f"scope_{st.session_state.active_conv_id}",
            )
            conv["scope_docs"] = selected_scope
        else:
            selected_scope = []

        # Chat attachment
        with st.expander("📎 上传附件（仅本对话有效，不入书库）", expanded=bool(conv["attachment_name"])):
            attach_file = st.file_uploader(
                "选择文件", type=["pdf", "docx", "doc", "txt"],
                key=f"attach_{st.session_state.active_conv_id}",
            )
            if attach_file and attach_file.name != conv.get("attachment_name"):
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

        # Render history
        for msg in conv["history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Guard: disable input only if no indexed docs AND no attachment AND indexing active
        chat_disabled = (not indexed_docs and not conv.get("attachment_text") and indexing_active)
        user_input = st.chat_input(
            "向书籍专家提问…" if not chat_disabled else "索引进行中，请稍候…",
            disabled=chat_disabled,
        )

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            doc_filter = conv["scope_docs"] if conv["scope_docs"] else None

            with st.spinner("从索引库检索相关段落…"):
                results = searcher.search(user_input, limit=5, doc_filter=doc_filter) if indexed_docs else []

            last_results_key = f"last_results_{st.session_state.active_conv_id}"
            st.session_state[last_results_key] = results

            if results:
                with st.expander(f"📚 找到 {len(results)} 条相关段落", expanded=False):
                    for i, r in enumerate(results):
                        st.markdown(f"**段落 {i+1}** — `{r['doc_name']}` (相关度 `{r['score']:.4f}`)")
                        st.caption(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))

            with st.chat_message("assistant"):
                try:
                    with st.spinner("DeepSeek 正在思考…"):
                        answer = summarizer.answer_question(
                            query=user_input,
                            context_chunks=results,
                            history=conv["history"],
                            attachment_text=conv.get("attachment_text", ""),
                        )
                    st.markdown(answer)
                    export_md = _build_export_md(user_input, answer, results)
                    st.download_button(
                        "📥 导出本次问答（Markdown）",
                        data=export_md,
                        file_name="bookexpert_qa.md",
                        mime="text/markdown",
                        key=f"export_{uuid.uuid4()}",
                    )
                    conv["history"].append({"role": "user",      "content": user_input})
                    conv["history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    if is_quota_error(e):
                        st.toast("🚫 DeepSeek API 配额已耗尽，请稍候再试。", icon="⚠️")
                        st.error(f"请求失败：DeepSeek API 配额不足。\n详情：{e}")
                    else:
                        st.toast("问答出错，请查看错误信息。", icon="❌")
                        st.error(f"错误：{e}")

    # ──────────────────────────────────────
    # TAB 2: Per-Document Summary (RAG)
    # ──────────────────────────────────────
    with tab_summary:
        st.header("📝 全书摘要（RAG 驱动）")
        indexed_docs = searcher.get_indexed_documents()

        if not indexed_docs:
            if iq.any_active():
                st.warning("📥 正在建立索引，完成后可生成摘要。")
            else:
                st.info("请先在左侧书库上传并索引书籍。")
        else:
            selected_doc = st.selectbox("选择要摘要的书籍", options=indexed_docs, key="summary_doc")
            dh = _doc_hash(selected_doc)
            st.info("💡 摘要使用 RAG 从索引库检索最相关段落后进行 Map-Reduce，更精准且节省 Token。")
            st.warning("首次生成需要一定时间；结果将缓存，下次秒级复用。")

            cached = sum_cache.get(f"summary_{dh}")
            if cached:
                st.success("✅ 已使用本地摘要缓存。")
                st.markdown("### 全书摘要")
                st.write(cached)
                if st.button("🔄 重新生成摘要"):
                    sum_cache.put(f"summary_{dh}", "")  # invalidate
                    st.rerun()
            else:
                if st.button("📖 生成全书摘要"):
                    # Guard: check doc is fully indexed (not mid-queue)
                    progress = iq.get_progress()
                    if selected_doc in progress and progress[selected_doc]["status"] in ("queued", "indexing"):
                        st.warning("该书仍在索引中，请等待索引完成后再生成摘要。")
                    else:
                        try:
                            with st.spinner(f"正在为《{selected_doc}》进行 RAG 检索 + Map-Reduce 摘要…"):
                                summary = summarizer.rag_summarize(
                                    doc_name=selected_doc,
                                    searcher=searcher,
                                    file_hash=dh,
                                    summary_cache=sum_cache,
                                )
                            st.markdown("### 全书摘要")
                            st.write(summary)
                        except Exception as e:
                            if is_quota_error(e):
                                st.toast("🚫 Gemini 嵌入 API 配额已耗尽，摄要失败。", icon="⚠️")
                                st.error(f"摘要失败：Gemini 或 DeepSeek API 配额不足。\n详情：{e}")
                            else:
                                st.toast("摘要生成出错。", icon="❌")
                                st.error(f"错误：{e}")

    # ──────────────────────────────────────
    # TAB 3: Book Review
    # ──────────────────────────────────────
    with tab_review:
        st.header("⭐ 书评系统")
        indexed_docs = searcher.get_indexed_documents()

        if not indexed_docs:
            if iq.any_active():
                st.warning("📥 正在建立索引，完成后可生成书评。")
            else:
                st.info("请先在左侧书库上传并索引书籍。")
        else:
            selected_doc = st.selectbox("选择要评分的书籍", options=indexed_docs, key="review_doc")
            dh = _doc_hash(selected_doc)

            # Guard: check indexing status
            progress = iq.get_progress()
            doc_indexing = selected_doc in progress and progress[selected_doc]["status"] in ("queued", "indexing")
            if doc_indexing:
                st.warning(f"《{selected_doc}》仍在索引中（{progress[selected_doc]['done']}/{progress[selected_doc]['total']} 块），请等待完成后生成书评。")
            else:
                # Load cached review
                cached_review_str = sum_cache.get(f"review_{dh}")
                review_data = None
                if cached_review_str:
                    try:
                        review_data = json.loads(cached_review_str)
                    except Exception:
                        pass

                # Score lock indicator
                locked_scores = sum_cache.get_scores(dh)
                score_locked = locked_scores is not None

                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("⭐ 生成书评" if not review_data else "🔄 重新生成书评"):
                    try:
                        with st.spinner(f"DeepSeek 正在评审《{selected_doc}》（RAG 检索中）…"):
                            review_data = reviewer.review_book(
                                searcher=searcher,
                                doc_name=selected_doc,
                                file_hash=dh,
                                summary_cache=sum_cache,
                            )
                        score_locked = True
                        st.rerun()
                    except Exception as e:
                        if is_quota_error(e):
                            st.toast("🚫 API 配额已耗尽，书评生成失败。", icon="⚠️")
                            st.error(f"书评失败：API 配额不足。\n详情：{e}")
                        else:
                            st.toast("书评生成出错。", icon="❌")
                            st.error(f"错误：{e}")


                if col_btn2.button("🔓 重置评分锁", disabled=not score_locked, help="清除锁定分数，下次重新生成"):
                    sum_cache.put(f"scores_{dh}", "")
                    sum_cache.put(f"review_{dh}", "")
                    score_locked = False
                    review_data = None
                    st.rerun()

                if review_data:
                    score = review_data.get("overall_score", 0)
                    dims  = review_data.get("dimensions", {})

                    # Score badge
                    score_color = "#2ecc71" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
                    lock_badge = " 🔒" if score_locked else ""
                    st.markdown(
                        f"""<div style='text-align:center;padding:20px 0 4px'>
                            <span style='font-size:72px;font-weight:900;color:{score_color};'>{score}</span>
                            <span style='font-size:32px;color:#aaa;'>&nbsp;/ 100{lock_badge}</span>
                            <p style='font-size:15px;color:#888;margin-top:2px;'>综合评分{'（已锁定，重新生成不变）' if score_locked else ''}</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    # Dimension grid
                    st.subheader("📊 维度评分（各满分 20 分）")
                    dim_comments = review_data.get("dimension_comments", {})
                    dim_cols = st.columns(len(dims))
                    for col, (dim_name, dim_score) in zip(dim_cols, dims.items()):
                        col.metric(dim_name, f"{dim_score} / 20")
                        if dim_comments.get(dim_name):
                            col.caption(dim_comments[dim_name][:120] + "…")

                    # Strengths & Weaknesses
                    st.divider()
                    col_s, col_w = st.columns(2)
                    with col_s:
                        st.subheader("✅ 优点")
                        for s in review_data.get("strengths", []):
                            st.markdown(f"- {s}")
                    with col_w:
                        st.subheader("⚠️ 不足")
                        for w in review_data.get("weaknesses", []):
                            st.markdown(f"- {w}")

                    # Critic essay
                    st.divider()
                    st.subheader("📝 专业书评")
                    st.markdown(review_data.get("critic_essay", ""))

                    # Detailed dimension analysis
                    with st.expander("🔍 各维度详细分析"):
                        for dim_name, comment in dim_comments.items():
                            if comment:
                                st.markdown(f"**{dim_name}**")
                                st.write(comment)

                    # Export
                    st.download_button(
                        "📥 导出书评（JSON）",
                        data=json.dumps(review_data, ensure_ascii=False, indent=2),
                        file_name=f"review_{selected_doc}.json",
                        mime="application/json",
                    )

                    st.divider()

                    # ── User Feedback Panel (reactive sliders) ────────────────
                    st.subheader("✏️ 读者反馈与评分纠偏")
                    st.caption("调整分项评分 → 综合评分自动求和；调整综合评分 → 分项均匀分配。所有调整自动记忆。")

                    existing_fb = fb_cache.get_feedback(dh) or {}
                    dim_names   = list(dims.keys())
                    n_dims      = len(dim_names)

                    # ── Initialize session state for this doc's sliders ───────
                    ss_overall = f"_fb_overall_{dh}"
                    if ss_overall not in st.session_state:
                        st.session_state[ss_overall] = int(
                            existing_fb.get("score_overrides", {}).get("overall_score", score)
                        )
                    for dname, dval in dims.items():
                        ss_dim = f"_fb_dim_{dname}_{dh}"
                        if ss_dim not in st.session_state:
                            st.session_state[ss_dim] = int(
                                existing_fb.get("score_overrides", {})
                                           .get("dimensions", {})
                                           .get(dname, dval)
                            )

                    # ── Callbacks for bidirectional sync ──────────────────────
                    def _on_dim_changed(doc_hash=dh, d_names=dim_names):
                        """Recalculate overall as sum of dim scores (capped at 100)."""
                        total = sum(
                            st.session_state[f"_fb_dim_{d}_{doc_hash}"]
                            for d in d_names
                        )
                        st.session_state[f"_fb_overall_{doc_hash}"] = min(total, 100)

                    def _on_overall_changed(doc_hash=dh, d_names=dim_names):
                        """Distribute overall score evenly across dimension sliders."""
                        target    = st.session_state[f"_fb_overall_{doc_hash}"]
                        per_dim   = target // len(d_names)
                        remainder = target % len(d_names)
                        for idx, d in enumerate(d_names):
                            st.session_state[f"_fb_dim_{d}_{doc_hash}"] = (
                                min(per_dim + (1 if idx < remainder else 0), 20)
                            )

                    # ── Overall slider ────────────────────────────────────────
                    st.markdown("**综合评分**")
                    st.slider(
                        "综合评分", 0, 100,
                        key=ss_overall,
                        on_change=_on_overall_changed,
                        label_visibility="collapsed",
                    )
                    st.caption(
                        f"当前分项之和：{sum(st.session_state[f'_fb_dim_{d}_{dh}'] for d in dim_names)}"
                    )

                    # ── Dimension sliders ─────────────────────────────────────
                    st.markdown("**分项评分调整**")
                    dim_cols2 = st.columns(n_dims)
                    for col2, dname in zip(dim_cols2, dim_names):
                        col2.slider(
                            dname, 0, 20,
                            key=f"_fb_dim_{dname}_{dh}",
                            on_change=_on_dim_changed,
                        )

                    # ── Text feedback fields ──────────────────────────────────
                    fb_strengths  = st.text_area(
                        "补充优点（每行一条）",
                        value="\n".join(existing_fb.get("extra_strengths", [])),
                        key=f"_fb_str_{dh}",
                    )
                    fb_weaknesses = st.text_area(
                        "补充不足（每行一条）",
                        value="\n".join(existing_fb.get("extra_weaknesses", [])),
                        key=f"_fb_wk_{dh}",
                    )
                    fb_comment = st.text_area(
                        "对书评的意见或补充",
                        value=existing_fb.get("user_comments", ""),
                        key=f"_fb_cmt_{dh}",
                    )

                    if st.button("🔄 根据反馈重新生成书评", key=f"_fb_submit_{dh}"):
                        # Read current slider values from session state
                        fb_overall_val  = st.session_state[ss_overall]
                        fb_dim_vals     = {
                            d: st.session_state[f"_fb_dim_{d}_{dh}"]
                            for d in dim_names
                        }
                        dim_deltas = [
                            f"{d}调整为{fb_dim_vals[d]}分"
                            for d, orig in dims.items()
                            if fb_dim_vals.get(d) != orig
                        ]
                        auto_prefs = "；".join(dim_deltas) if dim_deltas else ""

                        feedback_data = {
                            "score_overrides": {
                                "overall_score": fb_overall_val,
                                "dimensions":    fb_dim_vals,
                            },
                            "extra_strengths":  [s.strip() for s in fb_strengths.splitlines() if s.strip()],
                            "extra_weaknesses": [w.strip() for w in fb_weaknesses.splitlines() if w.strip()],
                            "user_comments":    fb_comment,
                            "scoring_prefs":    auto_prefs,
                        }
                        fb_cache.save_feedback(dh, feedback_data)

                        with st.spinner("根据您的反馈重新生成书评中…"):
                            review_data = reviewer.regenerate_with_feedback(
                                searcher=searcher,
                                doc_name=selected_doc,
                                file_hash=dh,
                                feedback=feedback_data,
                                scoring_prefs=auto_prefs,
                                summary_cache=sum_cache,
                            )
                        st.success("✅ 您的调整已记忆，将用于后续书评参考。")
                        st.rerun()




if __name__ == "__main__":
    main()
