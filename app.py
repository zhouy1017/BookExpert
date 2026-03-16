import os
import streamlit as st
import tempfile
from typing import List

from src.extractors import DocumentProcessor
from src.chunking import ChineseTextSplitter
from src.search import HybridSearcher
from src.summarizer import BookSummarizer

st.set_page_config(page_title="书籍专家 BookExpert", layout="wide")

@st.cache_resource
def get_processor():
    return DocumentProcessor()

@st.cache_resource
def get_chunker():
    return ChineseTextSplitter()

@st.cache_resource
def get_searcher():
    return HybridSearcher()

@st.cache_resource
def get_summarizer():
    return BookSummarizer()

def main():
    st.title("📚 书籍专家 — 中文书籍智能问答系统")
    st.markdown("上传大型 PDF / DOCX 书籍文件，自动建立索引，并通过 DeepSeek AI 进行智能问答与全书摘要。")

    processor = get_processor()
    chunker = get_chunker()
    searcher = get_searcher()
    summarizer = get_summarizer()

    with st.sidebar:
        st.header("📂 上传文档")
        uploaded_file = st.file_uploader("选择 PDF、DOCX 或 DOC 文件", type=['pdf', 'docx', 'doc'])
        if uploaded_file is not None:
            if st.button("⚙️ 处理并建立索引"):
                with st.spinner("正在提取文本内容…"):
                    try:
                        suffix = "." + uploaded_file.name.split('.')[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        text = processor.extract_text(tmp_path)
                        st.success(f"文本提取成功！共 {len(text)} 个字符。")

                        with st.spinner("正在分块…"):
                            chunks = chunker.split_text(text)
                            st.success(f"生成了 {len(chunks)} 个文本块。")

                        with st.spinner("正在向量化和索引（Google API + Qdrant + BM25）…"):
                            searcher.add_documents(chunks, doc_name=uploaded_file.name)
                            st.success("✅ 索引建立完成！")

                    except Exception as e:
                        st.error(f"处理文档时出错：{str(e)}")
                    finally:
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except:
                                pass

    tab1, tab2 = st.tabs(["💬 智能问答", "📝 全书摘要"])

    with tab1:
        st.header("与书籍专家对话")
        query = st.text_input("请输入您对书籍内容的问题：")
        if st.button("🔍 搜索并回答") and query:
            with st.spinner("正在检索相关段落…"):
                results = searcher.search(query, limit=5)

                if not results:
                    st.warning("在已索引的文档中未找到相关信息。")
                else:
                    st.markdown("### 找到的相关段落")
                    for i, r in enumerate(results):
                        with st.expander(f"段落 {i+1} —— 来源：{r['doc_name']}（相关度：{r['score']:.4f}）"):
                            st.markdown(r['text'])

                    with st.spinner("DeepSeek 正在生成回答…"):
                        answer = summarizer.answer_question(query, results)
                        st.markdown("### AI 回答")
                        st.info(answer)

    with tab2:
        st.header("全书内容 Map-Reduce 摘要")
        st.warning("⚠️ 提示：对大型书籍的摘要需要一定时间。系统将逐段请求 DeepSeek 进行摘要，再汇总为完整的全书摘要。")
        if st.button("📖 生成全书摘要"):
            if not searcher.bm25_corpus_texts:
                st.error("尚未索引任何文档，请先上传并处理书籍文件。")
            else:
                with st.spinner("正在运行 Map-Reduce 全书摘要（DeepSeek）…"):
                    summary = summarizer.summarize_large_document(searcher.bm25_corpus_texts)
                    st.markdown("### 全书摘要")
                    st.write(summary)

if __name__ == "__main__":
    main()
