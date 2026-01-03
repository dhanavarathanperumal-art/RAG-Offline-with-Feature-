import numpy as np
import time
import hashlib
from typing import List, Any, cast

# Guard sentence-transformers import; provide a deterministic fallback if missing
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

print("Starting E2E smoke test: embeddings -> Chroma/FAISS/NumPy -> query")
texts = [
    "Cats are small domesticated carnivorous mammals.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "The quick brown fox jumps over the lazy dog.",
    "Cats and dogs can live together in the same household."
]

# --- fallback embedding (dependency-free) ---
def create_simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in (t for t in (text or "").lower().split() if t):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


def get_embeddings(texts: List[str], model=None) -> np.ndarray:
    if model is not None:
        emb = model.encode(texts, show_progress_bar=False)
        arr = np.asarray(emb)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype('float32')
    else:
        return np.vstack([create_simple_embedding(t) for t in texts]).astype('float32')

# Load model if available (optional)
model = None
if SentenceTransformer is not None:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded embedding model")
    except Exception as e:
        print(f"Failed to load SentenceTransformer: {e}")
        model = None
else:
    print("SentenceTransformer not available; using fallback embeddings")

embeddings = get_embeddings(texts, model)
print(f"Created embeddings shape: {embeddings.shape}")

query = "information about cats"
q_emb = get_embeddings([query], model)

# Try ChromaDB first, then FAISS, otherwise fallback to NumPy brute-force
used_backend = None

# 1) ChromaDB path
try:
    import chromadb
    from chromadb.config import Settings
    print("Trying ChromaDB for nearest neighbor search (local)...")
    client = chromadb.PersistentClient(path="./data/database/chroma_smoke", settings=Settings(anonymized_telemetry=False))
    col_name = f"smoke_{int(time.time())}"
    collection = client.get_or_create_collection(name=col_name)

    ids = [str(i) for i in range(len(texts))]
    metadatas = [{"source": "smoke", "idx": int(i)} for i in range(len(texts))]
    # Cast metadatas to Any to satisfy chromadb typing expectations in editor
    collection.add(ids=ids, documents=list(texts), metadatas=cast(Any, metadatas), embeddings=embeddings.tolist())

    res = collection.query(query_embeddings=q_emb.tolist(), n_results=3)
    print("Search results (ChromaDB):")
    # Safely extract first query result (res typically a dict mapping to lists)
    distances = res.get('distances') or []
    ids_rows = res.get('ids') or []
    docs_rows = res.get('documents') or []

    dist_row = None
    id_row = None
    doc_row = None

    if isinstance(distances, list) and len(distances) > 0:
        dist_row = distances[0]
    if isinstance(ids_rows, list) and len(ids_rows) > 0:
        id_row = ids_rows[0]
    if isinstance(docs_rows, list) and len(docs_rows) > 0:
        doc_row = docs_rows[0]

    if dist_row is not None and id_row is not None and doc_row is not None:
        for dist, idx, doc in zip(dist_row, id_row, doc_row):
            try:
                print(f"- id={str(idx)}, dist={float(dist):.4f}, text={str(doc)}")
            except Exception:
                print(f"- id={idx}, dist={dist}, text={doc}")

    used_backend = 'chroma'
except Exception as e:
    print(f"ChromaDB not available or failed: {e}")

# 2) FAISS path
if used_backend is None:
    use_faiss = False
    try:
        import faiss  # type: ignore[reportMissingImports]
        use_faiss = True
        print("Using FAISS for nearest neighbor search")
    except Exception as e:
        print(f"FAISS not available, falling back to NumPy (reason: {e})")

    if use_faiss:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        D, I = index.search(q_emb.astype('float32'), k=3)
        print("Search results (FAISS):")
        for dist, idx in zip(D[0], I[0]):
            print(f"- idx={int(idx)}, dist={float(dist):.4f}, text={texts[int(idx)]}")

        used_backend = 'faiss'

# 3) NumPy fallback
if used_backend is None:
    # q_emb is (1, dim) so broadcasting works
    dists = np.linalg.norm(embeddings - q_emb, axis=1)
    idxs = np.argsort(dists)[:3]
    print("Search results (NumPy):")
    for idx in idxs:
        print(f"- idx={int(idx)}, dist={float(dists[idx]):.4f}, text={texts[int(idx)]}")

print("E2E smoke test completed")
