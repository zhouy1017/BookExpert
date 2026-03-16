# Tests for document extractors
import unittest
from src.extractors import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def test_init(self):
        processor = DocumentProcessor()
        self.assertIsNotNone(processor)

if __name__ == '__main__':
    unittest.main()
