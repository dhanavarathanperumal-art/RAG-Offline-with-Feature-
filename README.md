# Offline RAG Chatbot

Project scaffold for an offline Retrieval-Augmented Generation (RAG) chatbot using local models (GGUF) and PDF knowledge sources.

## Structure

```
offline_rag_chatbot/
├── .vscode/
├── data/
│   ├── pdfs/                  # PDFs go here
│   ├── database/              # Auto-created vector DB
│   └── models/                # GGUF models
├── src/                       # Python source
├── ui/                        # Minimal UI (Streamlit / other)
└── requirements.txt
```

## Quick start

### Windows one-click setup (recommended)

1. Run: `setup.bat` — this creates a `venv`, installs requirements, and creates `data/` folders.

### macOS / Linux one-click setup (recommended)

1. Run: `./setup.sh` — this will create a `.venv`, install requirements, and create `data/` folders.

### Manual setup (cross-platform)

1. Create a virtual env: `python -m venv .venv` and activate it
2. Install: `pip install -r requirements.txt`
3. Put your PDFs in `data/pdfs/` and configure model path in `.env`
4. Start the API: `uvicorn src.app:app --reload`
5. Start the UI: `streamlit run ui/app.py`

---

Vector store backends
- The project supports **ChromaDB** (default), **FAISS** (if available), and a **NumPy** fallback for small datasets.
- Select the backend in the Streamlit UI: use the sidebar **Vector store backend** control (auto|chroma|faiss|numpy).

Continuous Integration
- A GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs unit tests and smoke tests on push/pull requests.

Next steps you might want:
- Add a local LLM integration (optional; `llama-cpp-python` requires C build tools on Windows) ✅
- Add more unit tests and end-to-end UI tests ✅
- Add an optional FAISS installation path for Linux (if you require high-performance ANN indexing) ✅

Local LLM tips:
- To use a local LLM (llama-cpp / GGUF) on Windows you will need Visual Studio Build Tools (C/C++), CMake and a compatible wheel for `llama-cpp-python` — otherwise use the remote option.

Installing on Windows (high level):
1. Install Visual Studio Build Tools (C++ workload) from https://visualstudio.microsoft.com/downloads/.
2. Install CMake (https://cmake.org/) and add to PATH.
3. Optional: install `msys2` if you need additional Unix tooling.
4. Then: `python -m pip install --upgrade pip` and `python -m pip install llama-cpp-python` (may build from source).

Model files:
- Get a GGUF model (e.g., Llama/LLMs converted to GGUF) and place it in `data/models/` (e.g., `data/models/model.gguf`).

Notes:
- `src/llm.py` will attempt to load the model only if `llama_cpp` is importable and the path you set exists. The app shows availability and a safe fallback message if the model or library is missing.

If you'd like, I can open a PR with the changes above and enable the CI workflow for this repo.

Helper scripts:
- `scripts/check_llama_install.py`: convenience script that checks for CMake, MSVC, and Python dev headers and prints step-by-step installation tips for Windows and Unix.

Run the helper:
- `python scripts/check_llama_install.py`
