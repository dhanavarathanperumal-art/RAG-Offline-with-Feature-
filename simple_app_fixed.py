import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from src.vector_store import ChromaVectorStore, VectorStoreConfig, SimpleChunk

st.set_page_config(page_title="Offline RAG Chatbot - Fixed", page_icon="🤖", layout="wide")

st.title("🤖 Offline RAG Chatbot — FAISS-free (ChromaDB)")
st.markdown("**Upload PDFs and ask questions — works on Windows**")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Choose vector store backend (auto prefers chroma -> faiss -> numpy)
backend = st.sidebar.selectbox(
    "Vector store backend",
    options=["auto", "chroma", "faiss", "numpy"],
    index=0,
    help="Choose vector store backend (auto prefers Chroma then FAISS then NumPy)"
)

from typing import Optional, Any

@st.cache_resource
def get_store_resource(backend_choice: Optional[str]) -> Any:
    """Return a vector store instance or an Exception describing the error.
    Accepts None (interpreted as 'auto')."""
    cfg = VectorStoreConfig(persist_directory="./data/database/chroma", collection_name="simple_rag")
    try:
        from src.vector_store import get_vector_store
        b = (backend_choice or "auto").lower()
        store = get_vector_store(b, cfg)
        return store
    except Exception as e:
        # Propagate the message so UI can surface it
        return e

model = get_embedding_model()
store = None  # Will be initialized on demand via get_store_resource(backend)


# Upload
uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        save_path = Path("data/pdfs") / f.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded)} files to data/pdfs/")

if st.button("Process PDFs"):
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        st.warning("No PDFs to process in data/pdfs/")
    else:
        all_texts = []
        chunks = []
        total_pages = 0
        for pdf_path in pdf_files:
            # simple extraction using basic reader (avoid heavy libs here)
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    text = "\n".join(pages)
                    total_pages += len(pages)
            except Exception:
                # fallback
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(pdf_path))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages)
                    total_pages += len(pages)
                except Exception as e:
                    st.error(f"Failed to read {pdf_path.name}: {e}")
                    continue

            # chunking
            words = text.split()
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
            st.info(f"Processing {len(pdf_files)} files, ~{total_pages} pages, {len(all_texts)} chunks")
            progress = st.progress(0)
            batch_size = 50
            embeddings_batches = []
            with st.spinner("Creating embeddings and storing in vector store..."):
                for i in range(0, len(all_texts), batch_size):
                    batch = all_texts[i:i+batch_size]
                    batch_embeddings = model.encode(batch, show_progress_bar=False)
                    embeddings_batches.append(batch_embeddings)
                    progress.progress(min(100, int((i + len(batch)) / len(all_texts) * 100)))

                embeddings = np.vstack(embeddings_batches)

                # Initialize / get store based on backend choice
                store_candidate = get_store_resource(backend)
                if isinstance(store_candidate, Exception):
                    st.error(str(store_candidate))
                elif store_candidate is None:
                    st.error(f"Vector store could not be initialized for backend '{backend}'")
                else:
                    store_candidate.add_documents(chunks, np.array(embeddings))
                    st.success(f"Indexed {len(chunks)} chunks in {backend} store")
                    # Cache the active store in session so searches use the same instance
                    st.session_state['active_store'] = store_candidate
            progress.progress(100)
            st.balloons()

query = st.text_area("Ask a question about your PDFs:")
st.sidebar.markdown("---")
st.sidebar.subheader("LLM (optional)")
llm_model_path = st.sidebar.text_input("Local LLM model path (GGUF)", value="./data/models/model.gguf")
use_llm = st.sidebar.checkbox("Use LLM for answers (if available)", value=False)

# Show current counts / availability
stored_count = None
active_store = st.session_state.get('active_store')
if active_store is not None:
    try:
        stored_count = active_store.count()
    except Exception:
        stored_count = None

if stored_count is not None:
    st.sidebar.metric("Document chunks", stored_count)

# Check LLM model availability indicator
try:
    from src.llm import LocalLLMAdapter
    llama_test = LocalLLMAdapter(model_path=llm_model_path)
    if llama_test.is_available() and llama_test._load_model() is not None:
        st.sidebar.success("Local LLM available and model loaded")
    elif llama_test.is_available():
        st.sidebar.info("llama-cpp installed but model file not found")
    else:
        st.sidebar.info("Local LLM (llama-cpp) not installed")
except Exception:
    st.sidebar.info("LLM adapter error")

if st.button("Search") and query:
    store_instance = st.session_state.get('active_store') or get_store_resource(backend)
    if isinstance(store_instance, Exception):
        st.error(str(store_instance))
    elif store_instance is None:
        st.error("Vector store not available; please process PDFs first or check backend selection.")
    else:
        q_emb = model.encode([query])
        q_vector = np.asarray(q_emb[0], dtype='float32')
        results = store_instance.search(q_vector, k=5)
        if not results:
            st.info("No results")
        else:
            # Format results for display and for LLM context
            context = []
            for r in results:
                chunk = r['chunk']
                st.write(f"**From {chunk.pdf_name}** (score={r['score']:.3f})")
                st.write(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                st.markdown("---")
                context.append({"text": chunk.text, "source": {"pdf_name": chunk.pdf_name, "page": chunk.page_num}})

            # Optionally use an LLM to generate a concise answer
            if use_llm:
                try:
                    from src.llm import LocalLLMAdapter, RemoteLLMAdapter
                    local = LocalLLMAdapter(model_path=llm_model_path)
                    remote = RemoteLLMAdapter()

                    # Show availability indicators
                    if local.is_available() and local._load_model() is not None:
                        with st.spinner("Generating with local LLM..."):
                            resp = local.generate(query, context)
                        st.success("Using local LLM model")
                    elif remote.is_available():
                        with st.spinner("Generating with remote LLM..."):
                            resp = remote.generate(query, context)
                        st.success("Using remote LLM (API key provided)")
                    else:
                        resp = "LLM not available locally or remotely; showing retrieved context."
                except Exception as e:
                    resp = f"LLM error: {e}"

                with st.expander("🤖 Generated Answer (LLM)"):
                    st.write(resp)

# Chunk preview and filters
st.sidebar.markdown("---")
st.sidebar.subheader("Preview / Search Filters")
min_score = st.sidebar.slider("Min similarity score", 0.0, 1.0, 0.0, 0.01)
num_results = st.sidebar.slider("Max results", 1, 20, 5)

# PDF preview
pdf_list = [p.name for p in Path("data/pdfs").glob("*.pdf")]
if pdf_list:
    preview_pdf = st.sidebar.selectbox("Preview document", options=["All"] + pdf_list)
    if preview_pdf != "All":
        # try to show a couple of chunks for that pdf
        st.sidebar.write(f"Showing sample chunks for: {preview_pdf}")
        store_preview = st.session_state.get('active_store')
        if store_preview is not None:
            # perform a no-op query to list items; we will just fetch stored_count and a few chunks by searching a trivial term
            q = "the"
            q_emb = model.encode([q])[0]
            results_preview = store_preview.search(q_emb, k=10)
            shown = 0
            for r in results_preview:
                if r['chunk'].pdf_name == preview_pdf and shown < 3 and r['score'] >= min_score:
                    st.sidebar.write(r['chunk'].text[:200] + ("..." if len(r['chunk'].text) > 200 else ""))
                    shown += 1
            if shown == 0:
                st.sidebar.info("No matching chunks above threshold.")
else:
    st.sidebar.info("No PDFs uploaded yet")
