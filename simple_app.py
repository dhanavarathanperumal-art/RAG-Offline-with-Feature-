import streamlit as st
from pathlib import Path
from typing import Optional, Any
import numpy as np
import tempfile
import os
import hashlib
from datetime import datetime

# Try to import sentence-transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

# Vector store API
from src.vector_store import VectorStoreConfig, get_vector_store, SimpleChunk

# Safe print
def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

safe_print("Starting simple_app.py — robust offline RAG UI")

# --- simple hash embedding fallback ---

def create_simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in (t for t in (text or "").lower().split() if t):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# --- embedding model factory ---
@st.cache_resource
def get_embedding_model() -> Optional[Any]:
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        return None

# --- create/get vector store ---
@st.cache_resource
def get_store(backend_choice: Optional[str]) -> Any:
    cfg = VectorStoreConfig(persist_directory="./data/database/chroma", collection_name="simple_rag")
    try:
        b = (backend_choice or "auto").lower()
        store = get_vector_store(b, cfg)
        return store
    except Exception as e:
        return e

# --- Streamlit UI ---
st.set_page_config(page_title="Simple Offline RAG", page_icon="🤖", layout="wide")
st.title("Simple Offline RAG")
st.markdown("Upload PDFs, build index, and search — works offline with fallbacks.")

# Sidebar: backend
backend = st.sidebar.selectbox("Vector store backend", options=["auto", "chroma", "faiss", "numpy"], index=0)

# Upload PDFs
uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded:
    target = Path("data/pdfs")
    target.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        save_path = target / f.name
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded)} files to {target}")

# Process PDFs button
if st.button("Process PDFs"):
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        st.warning("No PDFs to process in data/pdfs/")
    else:
        all_texts = []
        chunks = []
        for pdf_path in pdf_files:
            text = ""
            # Try pdfplumber -> PyPDF2
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    text = "\n".join(pages)
            except Exception:
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(pdf_path))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages)
                except Exception as e:
                    st.error(f"Failed to read {pdf_path.name}: {e}")
                    continue

            words = (text or "").split()
            chunk_size = 300
            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i+chunk_size])
                chunk_obj = SimpleChunk(
                    text=chunk_text,
                    pdf_name=pdf_path.name,
                    page_num=0,
                    chunk_id=len(chunks),
                    start_char=i,
                    end_char=i + len(chunk_text),
                    metadata={}
                )
                chunks.append(chunk_obj)
                all_texts.append(chunk_text)

        if not all_texts:
            st.error("No text extracted from PDFs")
        else:
            st.info(f"Processing {len(pdf_files)} files, {len(all_texts)} chunks")
            progress = st.progress(0)
            batch_size = 50
            embeddings_batches = []
            model = get_embedding_model()
            use_model = model is not None

            with st.spinner("Creating embeddings and storing in vector store..."):
                for i in range(0, len(all_texts), batch_size):
                    batch = all_texts[i:i+batch_size]
                    if use_model:
                        be = model.encode(batch, show_progress_bar=False)
                    else:
                        be = np.vstack([create_simple_embedding(t) for t in batch])
                    embeddings_batches.append(be)
                    progress.progress(min(100, int((i + len(batch)) / len(all_texts) * 100)))

                embeddings = np.vstack(embeddings_batches)

                store_candidate = get_store(backend)
                if isinstance(store_candidate, Exception):
                    st.error(str(store_candidate))
                else:
                    try:
                        store_candidate.add_documents(chunks, np.array(embeddings))
                        st.success(f"Indexed {len(chunks)} chunks in {backend} store")
                        st.session_state['active_store'] = store_candidate
                    except Exception as e:
                        st.error(f"Failed to add documents to store: {e}")
            progress.progress(100)

query = st.text_area("Ask a question about your PDFs:")

if st.button("Search") and query:
    store_instance = st.session_state.get('active_store') or get_store(backend)
    if isinstance(store_instance, Exception):
        st.error(str(store_instance))
    elif store_instance is None:
        st.error("Vector store not available; please process PDFs first or check backend selection.")
    else:
        model = get_embedding_model()
        if model is not None:
            q_emb = model.encode([query])[0]
        else:
            q_emb = create_simple_embedding(query)
        q_vector = np.asarray(q_emb, dtype='float32')
        try:
            results = store_instance.search(q_vector, k=5)
        except Exception as e:
            st.error(f"Search error: {e}")
            results = []

        if not results:
            st.info("No results")
        else:
            context = []
            for r in results:
                # r may be a dict with keys 'chunk' or 'document', or a SimpleChunk-like object
                if isinstance(r, dict):
                    chunk = r.get('chunk') or r.get('document') or r
                else:
                    chunk = r

                def _as_str(val: Any) -> str:
                    if val is None:
                        return ""
                    if isinstance(val, bytes):
                        try:
                            return val.decode('utf-8', errors='replace')
                        except Exception:
                            return str(val)
                    return str(val)

                if isinstance(chunk, dict):
                    pdf_name = _as_str(chunk.get('pdf_name'))
                    text = _as_str(chunk.get('text'))
                else:
                    pdf_name = _as_str(getattr(chunk, 'pdf_name', None))
                    text = _as_str(getattr(chunk, 'text', None))

                score = (r.get('score', 0) if isinstance(r, dict) else getattr(r, 'score', 0))
                st.write(f"**From {pdf_name}** (score={score:.3f})")
                preview = text[:500]
                st.write(preview + ("..." if len(text) > 500 else ""))
                st.markdown("---")
                context.append({"text": text, "source": {"pdf_name": pdf_name}})

st.sidebar.markdown("---")
st.sidebar.header("Preview / Search Filters")
num_results = st.sidebar.slider("Max results", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("System info")
try:
    st.sidebar.write(f"SentenceTransformer available: {SentenceTransformer is not None}")
except Exception:
    pass

# end
