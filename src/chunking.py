# Custom Chinese Text Chunking

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ChineseTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        # Optimized separators for Chinese text
        separators = [
            "\n\n",
            "\n",
            "。",  # Chinese full stop
            "！",  # Exclamation
            "？",  # Question mark
            "；",  # Semicolon
            " ",
            ""
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False
        )
        
    def split_text(self, text: str) -> List[str]:
        logger.info(f"Splitting text into chunks (size={self.splitter._chunk_size}, overlap={self.splitter._chunk_overlap})")
        return self.splitter.split_text(text)
