from src.vector_store import ChromaVectorStore, VectorStoreConfig, SimpleChunk
import numpy as np

print('Testing ChromaVectorStore...')
cfg = VectorStoreConfig(persist_directory='./data/database/chroma_test', collection_name='test_col')
store = ChromaVectorStore(cfg)
chunks = [SimpleChunk(text='Hello world', pdf_name='a.pdf', page_num=1, chunk_id=0, start_char=0, end_char=11, metadata={})]
emb = np.array([[0.1]*384], dtype='float32')
store.add_documents(chunks, emb)
print('Added document')
q = np.array([0.1]*384)
res = store.search(q, k=1)
print('Search results:', res)