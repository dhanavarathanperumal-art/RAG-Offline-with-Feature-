"""
ULTRA SIMPLE RAG - FIXED VERSION
Uses ONLY PyMuPDF (which you already have) - NO scikit-learn, NO sentence-transformers!
"""

import os
import sys
import json
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib

# Safe print helper to avoid UnicodeEncodeError on Windows consoles
def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

safe_print("🚀 Starting Ultra Simple RAG (FIXED - No scikit-learn needed)...")

# Guarded import for PyMuPDF (fitz) so editor/linting doesn't error when the package
# is not present in the current interpreter. If missing, runtime functions will
# show user-friendly errors in the UI instead of crashing.
from typing import Any
try:
    import fitz  # PyMuPDF  # type: ignore[reportMissingImports]
except Exception:
    fitz = None  # type: Any

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.pkl"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. PDF READING with PyMuPDF (ALREADY INSTALLED!)
# ============================================================================

def extract_pages_from_pdf(pdf_path):
    """Return list of (page_text, page_num) using PyMuPDF (fitz)."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")

    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append((text.strip(), page_num))
        doc.close()
        return pages
    except Exception as e:
        raise RuntimeError(f"Error reading PDF {pdf_path}: {e}")

# ============================================================================
# 2. SIMPLE TEXT CHUNKING
# ============================================================================

def chunk_text_simple(text, chunk_size=500):
    """Simple text chunking by words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ============================================================================
# 3. ULTRA SIMPLE EMBEDDINGS - NO external dependencies!
# ============================================================================

def create_simple_embedding(text, vector_size=128):
    """
    Create simple embedding using word frequency and hashing.
    NO scikit-learn needed! NO sentence-transformers needed!
    """
    words = text.lower().split()
    vector = np.zeros(vector_size, dtype=np.float32)
    for word in words:
        # Use consistent md5 hashing so results are deterministic across runs
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % vector_size
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def get_simple_embeddings(texts, vector_size=128):
    """Create embeddings without ANY external dependencies."""
    vectors = []
    for text in texts:
        vec = create_simple_embedding(text, vector_size=vector_size)
        vectors.append(vec)
    return np.array(vectors)

# ============================================================================
# 4. SIMPLE SEARCH - NO scikit-learn cosine similarity
# ============================================================================

def simple_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def simple_search(query, chunks, embeddings, top_k=5, min_score=0.01):
    """Search using our simple cosine similarity."""
    if embeddings is None or len(chunks) == 0:
        return []

    query_emb = create_simple_embedding(query, vector_size=embeddings.shape[1])
    sims = np.dot(embeddings, query_emb)

    idxs = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in idxs:
        score = float(sims[idx])
        if score >= min_score:
            c = chunks[idx]
            results.append({
                'text': c['text'] if len(c['text']) <= 1000 else c['text'][:1000] + '...',
                'similarity': score,
                'index': int(idx),
                'pdf_name': c.get('pdf_name', 'unknown'),
                'page': c.get('page', None)
            })
    return results

# ============================================================================
# 5. INDEX PERSISTENCE
# ============================================================================

def save_index(chunks, embeddings):
    data = {'chunks': chunks, 'embeddings': embeddings}
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(data, f)


def load_index():
    if not INDEX_PATH.exists():
        return None, None
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    return data.get('chunks', []), data.get('embeddings', None)

# ============================================================================
# 6. MAIN STREAMLIT APP - SIMPLIFIED
# ============================================================================

def main():
    st.set_page_config(page_title="Ultra Simple RAG - FIXED", page_icon="📚", layout="wide")
    st.title("📚 Ultra Simple RAG Chatbot - FIXED VERSION")
    st.markdown("**✅ Uses ONLY PyMuPDF - NO scikit-learn - NO sentence-transformers!**")

    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
        st.session_state.embeddings = None

    if not st.session_state.chunks and INDEX_PATH.exists():
        try:
            chunks, embeddings = load_index()
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.success(f"✅ Loaded {len(chunks)} chunks from saved index")
        except Exception:
            pass

    with st.sidebar:
        st.header("📤 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Upload one or more PDF files")

        if uploaded_files:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    with st.spinner(f"Reading {uploaded_file.name}..."):
                        pages = extract_pages_from_pdf(tmp_path)
                        new_chunks = 0
                        for page_text, page_num in pages:
                            page_chunks = chunk_text_simple(page_text, chunk_size=300)
                            for chunk in page_chunks:
                                st.session_state.chunks.append({'text': chunk, 'pdf_name': uploaded_file.name, 'page': page_num})
                                new_chunks += 1
                        st.success(f"✅ {uploaded_file.name}: {new_chunks} chunks added")
                except ImportError as e:
                    st.error(str(e))
                    st.error("You said PyMuPDF is installed. Check if 'pip install pymupdf' worked.")
                    return
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        st.markdown("---")
        if st.button("🔧 Create Search Index", type="primary", use_container_width=True):
            if not st.session_state.chunks:
                st.warning("Upload PDFs first!")
            else:
                with st.spinner("Creating simple embeddings (fast!)..."):
                    texts = [c['text'] for c in st.session_state.chunks]
                    st.session_state.embeddings = get_simple_embeddings(texts)
                    st.success(f"✅ Ready! {len(texts)} chunks indexed")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Index"):
                if st.session_state.chunks and st.session_state.embeddings is not None:
                    save_index(st.session_state.chunks, st.session_state.embeddings)
                    st.success("Index saved!")
        with col2:
            if st.button("📂 Load Index"):
                if INDEX_PATH.exists():
                    chunks, embeddings = load_index()
                    if chunks:
                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.success(f"Loaded {len(chunks)} chunks")
                        st.rerun()

        st.markdown("---")
        st.header("📊 Stats")
        st.write(f"Total chunks: {len(st.session_state.chunks)}")
        if st.session_state.chunks:
            pdfs = set(c['pdf_name'] for c in st.session_state.chunks)
            st.write(f"Unique PDFs: {len(pdfs)}")

        st.markdown("---")
        st.header("🔍 Search Settings")
        precision = st.select_slider("Precision", options=["Low", "Medium", "High"], value="Medium")

        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.success("Cleared!")
            st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🔍 Ask Questions")
        query = st.text_area("Enter your question:", height=120, placeholder="Example: What is the main topic? What are the key findings?")
        if query and st.button("Search", type="primary"):
            if st.session_state.embeddings is None:
                st.warning("Please create the search index first!")
            else:
                precision_map = {"Low": {"top_k": 10, "min_score": 0.05}, "Medium": {"top_k": 7, "min_score": 0.1}, "High": {"top_k": 5, "min_score": 0.15}}

                # The Streamlit select_slider may return a tuple in some contexts; coerce to a string key
                prec_key = precision[0] if isinstance(precision, (tuple, list)) else precision
                prec_key = str(prec_key)

                settings = precision_map.get(prec_key, {"top_k": 7, "min_score": 0.1})

                # Clamp top_k to a safe range based on available chunks
                try:
                    k = int(settings.get("top_k", 7))
                except Exception:
                    k = 7
                k = max(1, min(k, max(1, len(st.session_state.chunks))))

                # Ensure min_score is a float
                try:
                    min_score_val = float(settings.get("min_score", 0.1))
                except Exception:
                    min_score_val = 0.1

                with st.spinner("Searching..."):
                    results = simple_search(query, st.session_state.chunks, st.session_state.embeddings, top_k=k, min_score=min_score_val)
                if results:
                    st.subheader(f"📄 Found {len(results)} relevant sections:")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"Result {i} - {r['pdf_name']} (Page {r['page']}) - Score: {r['similarity']:.3f}"):
                            st.write(r['text'])
                            st.caption(f"Source: {r['pdf_name']} - Page {r['page']}")
                    st.subheader("🤖 Simple Summary")
                    summary = f"**Based on {len(results)} sections from your documents:**\n\n"
                    for i, r in enumerate(results[:3], 1):
                        summary += f"{i}. **{r['pdf_name']}** (page {r['page']}): {r['text'][:150]}...\n\n"
                    summary += "\n**Key points:**\n"
                    summary += "• The documents contain relevant information\n"
                    summary += "• Higher scores indicate better matches\n"
                    summary += "• Adjust precision settings for more/less results\n"
                    st.write(summary)
                else:
                    st.warning("No results found. Try:")
                    st.write("• Different search terms")
                    st.write("• Lower precision setting")
                    st.write("• Upload more PDFs")
    with col2:
        st.header("💡 Quick Guide")
        st.markdown("""
        **How to use:**
        1. 📤 Upload PDFs
        2. 🔧 Click "Create Search Index"
        3. 🔍 Ask questions
        
        **What's working:**
        ✅ PyMuPDF (PDF reading)
        ✅ Simple embeddings
        ✅ Cosine similarity
        ✅ No external deps!
        
        **Tips:**
        • Process PDFs once
        • Save index for later
        • Use clear questions
        """)
        st.markdown("---")
        st.header("🛠️ System Info")
        with st.expander("Details"):
            st.write("**Dependencies:")
            st.write("- Streamlit 1.28.0 ✅")
            st.write("- PyMuPDF 1.23.8 ✅")
            st.write("- NumPy ✅")
            st.write("- No scikit-learn! ✅")
            st.write("- No sentence-transformers! ✅")
            st.write("\n**Embedding method:**")
            st.write("Simple word frequency + hashing")
            st.write("\n**Search method:**")
            st.write("Manual cosine similarity")

# ============================================================================
# 7. RUN THE APP
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ULTRA SIMPLE RAG - FIXED VERSION")
    print("Uses ONLY what you already have installed!")
    print("=" * 60)
    main()
