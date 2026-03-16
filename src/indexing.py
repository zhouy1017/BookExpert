# Indexing and Embedding Generation

import os
import logging
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self):
        # Load API key from d:/BookExpert/google.apikey
        try:
            with open("d:/BookExpert/google.apikey", "r", encoding="utf-8") as f:
                self.api_key = f.read().strip()
                os.environ["GOOGLE_API_KEY"] = self.api_key
        except Exception as e:
            logger.error("Could not read google.apikey")
            raise e
            
        logger.info("Initializing Google Generative AI Embeddings")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        return self.embeddings.embed_documents(texts)
        
    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
