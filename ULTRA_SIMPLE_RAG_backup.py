"""
Minimal offline RAG helper (backup file)

Features:
- Optional PDF extraction using PyMuPDF (if installed)
- Deterministic hash-based embeddings (no heavy deps)
- In-memory index with cosine similarity search
- Save / load index to disk (pickle)

Usage:
  python ULTRA_SIMPLE_RAG_backup.py

This script scans `data/pdfs/` for PDF files, extracts text (if PyMuPDF
is available), splits into chunks, indexes them, and opens a simple
interactive prompt to ask queries. Designed to work offline with only
NumPy as a real dependency.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Optional PDF support
try:
	import fitz  # PyMuPDF  # type: ignore[reportMissingImports]
except Exception:
	fitz = None  # type: ignore


def safe_print(msg: str) -> None:
	try:
		print(msg)
	except UnicodeEncodeError:
		print(msg.encode("ascii", "replace").decode("ascii"))


class SimpleChunk:
	def __init__(self, text: str, pdf_name: str = "", page_num: int = 0, chunk_id: int = 0, metadata: Optional[Dict[str, Any]] = None):
		self.text = text
		self.pdf_name = pdf_name
		self.page_num = page_num
		self.chunk_id = chunk_id
		self.metadata = metadata or {}

	def to_dict(self) -> Dict[str, Any]:
		return {
			"text": self.text,
			"pdf_name": self.pdf_name,
			"page_num": self.page_num,
			"chunk_id": self.chunk_id,
			"metadata": self.metadata,
		}


def create_simple_embedding(text: str, dim: int = 128) -> np.ndarray:
	vec = np.zeros(dim, dtype=np.float32)
	for token in (t for t in (text or "").lower().split() if t):
		idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
		vec[idx] += 1.0
	norm = np.linalg.norm(vec)
	return vec / norm if norm > 0 else vec


def extract_text_from_pdf(path: Path) -> str:
	if fitz is None:
		raise RuntimeError("PyMuPDF not available. Install with: pip install pymupdf")
	doc = fitz.open(str(path))
	texts: List[str] = []
	for page in doc:
		texts.append(page.get_text() or "")
	doc.close()
	return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
	words = (text or "").split()
	chunks: List[str] = []
	for i in range(0, len(words), chunk_size):
		chunks.append(" ".join(words[i : i + chunk_size]))
	return chunks


class SimpleIndex:
	def __init__(self, dim: int = 128):
		self.dim = dim
		self.chunks: List[SimpleChunk] = []
		self.embeddings: Optional[np.ndarray] = None

	def add(self, chunks: List[SimpleChunk], embeddings: np.ndarray) -> None:
		if embeddings.shape[1] != self.dim:
			raise ValueError("Embedding dimension mismatch")
		if self.embeddings is None:
			self.embeddings = embeddings.copy()
		else:
			self.embeddings = np.vstack([self.embeddings, embeddings])
		self.chunks.extend(chunks)

	def save(self, path: Path) -> None:
		data = {
			"dim": self.dim,
			"chunks": [c.to_dict() for c in self.chunks],
		}
		with open(path.with_suffix(".pkl"), "wb") as f:
			pickle.dump(data, f)
		if self.embeddings is not None:
			np.save(path.with_suffix(".npy"), self.embeddings)

	def load(self, path: Path) -> None:
		pkl = path.with_suffix(".pkl")
		npy = path.with_suffix(".npy")
		if not pkl.exists() or not npy.exists():
			raise FileNotFoundError("Index files not found")
		with open(pkl, "rb") as f:
			data = pickle.load(f)
		self.dim = int(data.get("dim", self.dim))
		self.chunks = [SimpleChunk(**c) for c in data.get("chunks", [])]
		self.embeddings = np.load(npy)

	def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
		if self.embeddings is None or len(self.chunks) == 0:
			return []
		q = create_simple_embedding(query, dim=self.dim)
		# embeddings are normalized; dot product is cosine
		scores = np.dot(self.embeddings, q)
		order = np.argsort(-scores)
		results: List[Dict[str, Any]] = []
		for idx in order[:top_k]:
			score = float(scores[idx])
			if score < min_score:
				continue
			ch = self.chunks[idx]
			results.append({"score": score, "chunk": ch.to_dict()})
		return results


def build_index_from_pdfs(pdf_dir: Path, chunk_size: int = 300, dim: int = 128) -> SimpleIndex:
	idx = SimpleIndex(dim=dim)
	pdf_dir.mkdir(parents=True, exist_ok=True)
	pdfs = list(pdf_dir.glob("*.pdf"))
	if not pdfs:
		safe_print("No PDFs found in data/pdfs/")
		return idx

	all_chunks: List[SimpleChunk] = []
	all_embs: List[np.ndarray] = []
	for p in pdfs:
		safe_print(f"Processing {p.name}")
		try:
			text = extract_text_from_pdf(p) if fitz is not None else ""
		except Exception as e:
			safe_print(f"Failed to extract {p.name}: {e}")
			continue
		pieces = chunk_text(text, chunk_size=chunk_size)
		for i, piece in enumerate(pieces):
			sc = SimpleChunk(text=piece, pdf_name=p.name, page_num=0, chunk_id=len(all_chunks))
			all_chunks.append(sc)
			all_embs.append(create_simple_embedding(piece, dim=dim))

	if all_chunks:
		idx.add(all_chunks, np.vstack(all_embs).astype(np.float32))
	return idx


def interactive_loop(index: SimpleIndex) -> None:
	safe_print("Simple offline RAG — interactive query. Type 'exit' to quit.")
	while True:
		try:
			q = input("Query> ")
		except (EOFError, KeyboardInterrupt):
			safe_print("Exiting.")
			break
		if not q or q.strip().lower() in {"exit", "quit"}:
			safe_print("Goodbye")
			break
		res = index.search(q, top_k=5)
		if not res:
			safe_print("No results")
			continue
		for r in res:
			ch = r.get("chunk", {})
			score = r.get("score", 0)
			safe_print(f"Score={score:.4f} Source={ch.get('pdf_name', '')}")
			text = ch.get("text", "")
			safe_print(text[:500] + ("..." if len(text) > 500 else ""))
			safe_print("---")


def main() -> None:
	base = Path("data")
	pdf_dir = base / "pdfs"
	index_path = base / "database" / "ultra_simple_rag_index"
	index_path.parent.mkdir(parents=True, exist_ok=True)

	idx = SimpleIndex(dim=128)
	# Try load existing index
	try:
		idx.load(index_path)
		safe_print("Loaded existing index")
	except Exception:
		safe_print("No existing index — building from PDFs (if any).")
		idx = build_index_from_pdfs(pdf_dir, chunk_size=300, dim=128)
		if idx.embeddings is not None and len(idx.chunks) > 0:
			idx.save(index_path)
			safe_print(f"Saved index to {index_path.with_suffix('.pkl')} and .npy")

	interactive_loop(idx)


if __name__ == "__main__":
	main()
