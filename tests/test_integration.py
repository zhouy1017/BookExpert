"""
Automated end-to-end integration test using tests/xilehui.docx and tests/xilehui.pdf
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractors import DocumentProcessor
from src.chunking import ChineseTextSplitter
from src.search import HybridSearcher
from src.summarizer import BookSummarizer

def run_test(file_path: str, label: str):
    print(f"\n{'='*60}")
    print(f"[TEST] Processing: {label}")

    print("  [1/4] Extracting text...")
    processor = DocumentProcessor()
    text = processor.extract_text(file_path)
    print(f"  ✓ Extracted {len(text):,} characters.")
    assert len(text) > 100, "Text extraction returned too little text!"

    print("  [2/4] Chunking...")
    chunker = ChineseTextSplitter()
    chunks = chunker.split_text(text)
    print(f"  ✓ Generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "No chunks generated!"

    print("  [3/4] Embedding & Indexing (Google API)...")
    # Use a fresh db path for each test to avoid conflicts
    db_path = f"db_test_{label.replace('.','_')}"
    searcher = HybridSearcher(db_path=db_path)
    searcher.add_documents(chunks[:5], doc_name=label)  # test with first 5 chunks to save time
    print(f"  ✓ Indexed {min(5, len(chunks))} chunks.")

    print("  [4/4] Hybrid Search test...")
    results = searcher.search("主角", limit=3)
    print(f"  ✓ Search returned {len(results)} results.")
    for r in results[:2]:
        print(f"    - Score: {r['score']:.4f} | Text[:80]: {r['text'][:80].replace(chr(10),' ')}")

    # Cleanup temp db
    import shutil
    shutil.rmtree(db_path, ignore_errors=True)

    print(f"  [PASS] {label} passed all checks!")
    return True

if __name__ == "__main__":
    failed = []
    tests = [
        ("tests/xilehui.docx", "xilehui.docx"),
        ("tests/xilehui.pdf",  "xilehui.pdf"),
    ]
    for path, label in tests:
        try:
            run_test(path, label)
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            failed.append(label)

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")
