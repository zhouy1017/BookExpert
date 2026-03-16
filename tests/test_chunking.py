# Tests for text chunker
import unittest
from src.chunking import ChineseTextSplitter

class TestChineseTextSplitter(unittest.TestCase):
    def test_init(self):
        splitter = ChineseTextSplitter()
        self.assertIsNotNone(splitter)

if __name__ == '__main__':
    unittest.main()
