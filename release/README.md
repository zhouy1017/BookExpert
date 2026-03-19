# BookExpert — 中文书籍智能分析系统

**BookExpert** is a Chinese-language AI book analysis platform.  
Features: multi-turn Q&A, RAG-guided full-book summarisation, and professional critic reviews — all powered by Gemini + DeepSeek with local vector indexing.

---

## Quick Start (Windows)

### 1. Prerequisites
- **Python 3.10 or higher** — [download](https://www.python.org/downloads/)
- **Google AI Studio API key** — [get one free](https://aistudio.google.com/)
- **DeepSeek API key** — [get one](https://platform.deepseek.com/)

### 2. Install

```bat
release\install.bat
```

This creates a `bookexpert\` virtual environment and installs all dependencies.

### 3. Add API Keys

Create two plain-text files in the **project root** (`d:\BookExpert\`):

| File | Contents |
|------|----------|
| `google.apikey` | Your Google AI Studio API key |
| `deepseek.apikey` | Your DeepSeek API key |

### 4. Run

```bat
release\run.bat
```

Opens the app at **http://localhost:8501**.

---

## Optional: Build Standalone EXE

```bat
release\build_exe.bat
```

Packages the app into a single `dist\BookExpert.exe` using PyInstaller.  
> Copy your `.apikey` files next to the EXE before running it.

---

## Project Structure

```
BookExpert/
├── app.py               Main Streamlit application
├── src/
│   ├── llm_provider.py  Dual-LLM: Gemini Flash Lite (primary) + DeepSeek (fallback)
│   ├── summarizer.py    RAG Map-Reduce summarisation
│   ├── reviewer.py      Professional book review generation
│   ├── indexing.py      Gemini embedding indexer (dual-model 429 fallback)
│   ├── search.py        Hybrid BM25 + semantic search
│   ├── rate_limiter.py  API quota tracking & enforcement
│   └── cache.py         SQLite caches (embeddings, summaries, feedback)
├── release/
│   ├── install.bat      One-click installer
│   ├── run.bat          One-click launcher
│   └── build_exe.bat    PyInstaller EXE builder
├── db/                  Local vector index + metadata (auto-created)
├── requirements.txt
├── google.apikey        ← you create this
└── deepseek.apikey      ← you create this
```

---

## LLM Models Used

| Role | Primary | Fallback |
|------|---------|---------|
| Chat / QA / Summary / Review | `gemini-3.1-flash-lite-preview` | `deepseek-chat` |
| Embeddings | `gemini-embedding-001` | `gemini-embedding-002` |

Rate limits are tracked locally per session (RPM/TPM/RPD) and displayed in the sidebar quota toolbar. Auto-fallback triggers on quota exhaustion or API errors.
