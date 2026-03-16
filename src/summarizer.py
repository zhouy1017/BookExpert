# Map-Reduce Summarization and QA via DeepSeek
import os
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class BookSummarizer:
    def __init__(self):
        try:
            with open("d:/BookExpert/deepseek.apikey", "r", encoding="utf-8") as f:
                self.api_key = f.read().strip()
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
        
    def summarize_large_document(self, texts: List[str]) -> str:
        """Runs a Map-Reduce summarization over chunks."""
        if not texts:
            return "No text to summarize."
            
        docs = [Document(page_content=t) for t in texts]
        
        # 1. Map chain
        map_prompt = PromptTemplate.from_template(
            "Summarize the following text segment from a book. Focus on main events, ideas, or entities:\n\n{text}\n\nSummary:"
        )
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        
        # 2. Reduce chain
        reduce_prompt = PromptTemplate.from_template(
            "The following is a set of summaries of segments of a book:\n\n{text}\n\nWrite a comprehensive, cohesive overall summary of the entire book based on these segment summaries:\n\nFinal Summary:"
        )
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        
        # Combines the intermediate summaries
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="text"
        )
        
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=30000, 
        )
        
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="text",
            return_intermediate_steps=False
        )
        
        logger.info(f"Running Map-Reduce on {len(docs)} documents...")
        result = map_reduce_chain.run(docs)
        return result
        
    def answer_question(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answers a question given retrieved contexts."""
        if not context_chunks:
            return "No relevant context found in the book to answer your question."
            
        context_str = "\n\n".join([f"Excerpt from {c.get('doc_name', 'book')} (Score: {c.get('score', 0):.2f}):\n{c['text']}" for c in context_chunks])
        
        prompt = PromptTemplate.from_template(
            "You are BookExpert, an AI assistant specialized in analyzing a book based on excerpts provided to you.\n"
            "Use the following excerpts from the book to answer the user's question accurately.\n"
            "If the answer is not contained in the excerpts, state that you do not have enough information.\n\n"
            "--- Book Excerpts ---\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info(f"Answering question based on {len(context_chunks)} retrieved chunks.")
        return chain.run(context=context_str, question=query)

