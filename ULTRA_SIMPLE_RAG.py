"""
ULTRA SIMPLE RAG CHATBOT - FIXED AND SAFE
- No scikit-learn required
- Uses PyMuPDF (fitz) for PDF extraction (guarded import)
- Uses a deterministic hash-based embedding (no external models)
"""

import streamlit as st
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import tempfile
import re
import hashlib

# Guarded import for PyMuPDF (fitz) to avoid crashes and silence Pylance
try:
    import fitz  # PyMuPDF  # type: ignore[reportMissingImports]
except Exception:
    fitz = None

# Safe print helper to avoid UnicodeEncodeError on Windows consoles
def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback to ASCII-replaced version
        print(msg.encode('ascii', 'replace').decode('ascii'))

safe_print("🚀 ULTRA SIMPLE RAG (fixed & safe) starting...")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "index.pkl"

# -----------------------------
# PDF extraction
# -----------------------------

def extract_text_with_pymupdf(pdf_path):
    if fitz is None:
        st.error("PyMuPDF (fitz) is not available. Install with: pip install pymupdf or select the correct interpreter.")
        return ""

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc, 1):
            pg_text = page.get_text() or ""
            if pg_text.strip():
                text += f"--- Page {page_num} ---\n{pg_text}\n\n"
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {Path(pdf_path).name}: {e}")
        return ""

# -----------------------------
# Chunking
# -----------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

# -----------------------------
# Embedding: deterministic hash-based
# -----------------------------

def create_simple_embedding(text, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def get_simple_embeddings(texts, dim=128):
    return np.vstack([create_simple_embedding(t, dim=dim) for t in texts])

# -----------------------------
# Search
# -----------------------------

def simple_search(query, chunks, embeddings, top_k=5, min_score=0.01):
    if embeddings is None or len(chunks) == 0:
        return []
    q = create_simple_embedding(query, dim=embeddings.shape[1])
    sims = np.dot(embeddings, q)  # dot of normalized vectors = cosine similarity
    idxs = np.argsort(sims)[-top_k:][::-1]
    results = []
    for i in idxs:
        score = float(sims[i])
        if score >= min_score:
            c = chunks[i]
            results.append({
                'text': c['text'] if len(c['text']) <= 1000 else c['text'][:1000] + '...',
                'pdf': c.get('pdf', 'unknown'),
                'similarity': score,
                'chunk_id': i,
                'info': c
            })
    return results

# -----------------------------
# Persistence helpers
# -----------------------------

def save_index(chunks, embeddings):
    import pickle
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)


def load_index():
    import pickle
    if not INDEX_PATH.exists():
        return [], None
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    return data.get('chunks', []), data.get('embeddings', None)

# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(page_title="ULTRA SIMPLE RAG", page_icon="🤖", layout="wide")
    st.markdown("# 🤖 ULTRA SIMPLE RAG (fixed)")
    st.markdown("**Offline • No C++ compilers required • Deterministic hash embeddings**")

    if 'chunks' not in st.session_state:
        st.session_state.chunks = []  # each chunk is dict {'text','pdf','page',...}
        st.session_state.embeddings = None

    # Try load existing index
    if (st.session_state.embeddings is None or not st.session_state.chunks) and INDEX_PATH.exists():
        try:
            chunks, embeddings = load_index()
            if chunks and embeddings is not None:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.info(f"Loaded index: {len(chunks)} chunks")
        except Exception as e:
            st.warning(f"Failed to load index: {e}")

    with st.sidebar:
        st.header("📤 PDF Setup")
        pdf_files = list(Path('.').glob('*.pdf'))
        if pdf_files:
            for p in pdf_files:
                st.write(f"• {p.name} ({p.stat().st_size / 1024/1024:.2f} MB)")
        else:
            st.info("Put PDFs in same folder as this script (or upload below)")

        uploaded = st.file_uploader("Upload PDFs", type='pdf', accept_multiple_files=True)
        if uploaded:
            for uf in uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                try:
                    pages_text = extract_text_with_pymupdf(tmp_path)
                    if pages_text:
                        chs = chunk_text(pages_text)
                        for ch in chs:
                            st.session_state.chunks.append({'text': ch, 'pdf': uf.name})
                        st.success(f"Added {len(chs)} chunks from {uf.name}")
                    else:
                        st.warning(f"No text found in {uf.name}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        st.markdown('---')
        if st.button('🔧 Create/Recreate Index'):
            if not st.session_state.chunks:
                st.warning('No chunks available. Upload PDFs or place PDFs in folder.')
            else:
                texts = [c['text'] for c in st.session_state.chunks]
                with st.spinner('Creating embeddings...'):
                    st.session_state.embeddings = get_simple_embeddings(texts)
                st.success(f"Indexed {len(texts)} chunks")

        if st.button('💾 Save Index'):
            if st.session_state.chunks and st.session_state.embeddings is not None:
                save_index(st.session_state.chunks, st.session_state.embeddings)
                st.success('Index saved')
            else:
                st.warning('Nothing to save')

        if st.button('📂 Load Index'):
            chunks, embeddings = load_index()
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.success(f'Loaded {len(chunks)} chunks')

        if st.button('🗑 Clear'):
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.success('Cleared index and chunks')

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header('Ask your PDFs')
        query = st.text_area('Enter your question', height=120)
        top_k = st.number_input('Results', min_value=1, max_value=20, value=5)
        min_sim = st.slider('Min similarity', 0.0, 1.0, 0.05, 0.01)

        if st.button('Search'):
            if st.session_state.embeddings is None:
                st.warning('Please create the index first')
            else:
                # Normalize and clamp top_k to a safe integer range
                try:
                    k = int(top_k)
                except Exception:
                    k = 5
                k = max(1, min(k, max(1, len(st.session_state.chunks))))

                with st.spinner('Searching...'):
                    results = simple_search(query, st.session_state.chunks, st.session_state.embeddings, top_k=k, min_score=min_sim)
                if results:
                    st.success(f'Found {len(results)} results')
                    for r in results:
                        st.write(f"**{r['pdf']}** (score: {r['similarity']:.3f})")
                        st.write(r['text'])
                else:
                    st.info('No results above threshold')

    with col2:
        st.header('Help & Info')
        st.write('This app uses PyMuPDF for PDF extraction and a simple hash-based embedding. No scikit-learn or sentence-transformers required.')

if __name__ == '__main__':
    main()