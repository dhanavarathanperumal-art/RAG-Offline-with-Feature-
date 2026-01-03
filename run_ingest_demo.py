from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.vector_store import ChromaVectorStore, VectorStoreConfig, SimpleChunk
import numpy as np

pdf_path = Path('data/pdfs/sample.pdf')
if not pdf_path.exists():
    raise SystemExit('Sample PDF not found')

# Extract text
try:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
except Exception:
    from PyPDF2 import PdfReader
    reader = PdfReader(str(pdf_path))
    text = "\n".join([p.extract_text() or "" for p in reader.pages])

words = text.split()
chunks = []
all_texts = []
chunk_size = 300
for i in range(0, len(words), chunk_size):
    chunk_text = ' '.join(words[i:i+chunk_size])
    chunk = SimpleChunk(text=chunk_text, pdf_name=pdf_path.name, page_num=0, chunk_id=len(chunks), start_char=i, end_char=i+len(chunk_text), metadata={})
    chunks.append(chunk)
    all_texts.append(chunk_text)

print(f'Extracted {len(all_texts)} chunks from {pdf_path.name}')

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_texts, show_progress_bar=False)

cfg = VectorStoreConfig(persist_directory='./data/database/chroma_demo', collection_name='demo_col')
store = ChromaVectorStore(cfg)
store.add_documents(chunks, np.array(embeddings))
print('Indexed chunks into Chroma')

# Query
q = 'cats and dogs'
q_emb = model.encode([q])[0]
results = store.search(q_emb, k=3)
print('Query results:')
for r in results:
    chunk = r['chunk']
    print(f"- score={r['score']:.3f} pdf={chunk.pdf_name} text={chunk.text[:120]}")
