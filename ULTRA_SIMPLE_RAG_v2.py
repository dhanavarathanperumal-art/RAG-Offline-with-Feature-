"""
ULTRA SIMPLE RAG CHATBOT - NO INTERNET NEEDED
Everything works offline with what you already have!
"""

import streamlit as st
try:
    import fitz  # PyMuPDF  # type: ignore[reportMissingImports]
except Exception:
    fitz = None
import numpy as np
import os
from pathlib import Path
import re
import hashlib
from datetime import datetime

class UltraSimpleRAG:
    def __init__(self):
        self.text_chunks = []
        self.pdf_names = []
        self.chunk_info = []
        self.embeddings = None
        
    def extract_text_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF - Already installed and working!"""
        if fitz is None:
            st.error("PyMuPDF (fitz) is not available in this environment. Install it with: pip install pymupdf or select the correct Python interpreter in VS Code.")
            return ""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"--- Page {page_num} ---\n{page_text}\n\n"
            doc.close()
            return text.strip()
        except Exception as e:
            st.error(f"Error reading {pdf_path.name}: {str(e)}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip()
    
    def chunk_text(self, text, pdf_name, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'pdf': pdf_name,
                    'chunk_id': len(chunks),
                    'start_word': i,
                    'end_word': i + len(chunk_words),
                    'word_count': len(chunk_words)
                })
        
        return chunks
    
    def create_simple_embedding(self, text):
        """Create a simple embedding without external models"""
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Create a simple hash-based embedding (deterministic)
        # This creates a 128-dimensional vector using word hashes
        embedding = np.zeros(128, dtype=np.float32)
        
        for word in words:
            # Hash the word to get a position in the vector
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16) % 128
            embedding[word_hash] += 1.0
        
        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def process_pdfs(self, pdf_folder="."):
        """Process all PDFs in folder"""
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            st.error("❌ No PDF files found! Please put PDFs in the same folder as this script.")
            return False
        
        st.info(f"📚 Found {len(pdf_files)} PDF file(s)")
        
        all_chunks = []
        all_pdf_names = []
        all_chunk_info = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name} ({idx+1}/{len(pdf_files)})...")
            
            # Extract text
            text = self.extract_text_with_pymupdf(pdf_file)
            
            if text.strip():
                # Clean text
                clean_text = self.clean_text(text)
                
                # Split into chunks
                chunks = self.chunk_text(clean_text, pdf_file.name)
                
                # Add to collections
                for chunk in chunks:
                    all_chunks.append(chunk['text'])
                    all_pdf_names.append(chunk['pdf'])
                    all_chunk_info.append(chunk)
                
                st.success(f"✅ {pdf_file.name}: Extracted {len(chunks)} text chunks")
            else:
                st.warning(f"⚠️ {pdf_file.name}: No text found (might be scanned or empty)")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(pdf_files))
        
        if not all_chunks:
            st.error("❌ No text extracted from any PDF!")
            return False
        
        # Create embeddings
        status_text.text("Creating embeddings...")
        with st.spinner("🔢 Creating embeddings (this might take a minute for large PDFs)..."):
            embeddings_list = []
            for i, chunk in enumerate(all_chunks):
                emb = self.create_simple_embedding(chunk)
                embeddings_list.append(emb)
                
                # Update progress occasionally
                if i % 100 == 0:
                    progress_bar.progress(0.5 + (i / len(all_chunks)) * 0.5)
            
            self.embeddings = np.array(embeddings_list)
        
        # Store data
        self.text_chunks = all_chunks
        self.pdf_names = all_pdf_names
        self.chunk_info = all_chunk_info
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        
        st.success(f"🎉 Ready! Processed {len(all_chunks)} text chunks from {len(pdf_files)} PDF(s)")
        return True
    
    def search(self, query, top_k=5, min_similarity=0.1):
        """Search for similar text chunks"""
        if self.embeddings is None or len(self.text_chunks) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.create_simple_embedding(query)
        
        # Calculate similarities
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            similarities.append((sim, idx))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Collect results above threshold
        results = []
        for sim, idx in similarities:
            if sim >= min_similarity and len(results) < top_k:
                results.append({
                    'text': self.text_chunks[idx],
                    'pdf': self.pdf_names[idx],
                    'similarity': float(sim),
                    'chunk_id': idx,
                    'info': self.chunk_info[idx]
                })
        
        return results
    
    def create_response(self, query, results):
        """Create a simple AI response based on results"""
        if not results:
            return "I couldn't find any relevant information in the documents."
        
        response = f"**Based on your query: \"{query}\"**\n\n"
        response += f"I found {len(results)} relevant section(s):\n\n"
        
        for i, result in enumerate(results[:3], 1):
            response += f"{i}. **From '{result['pdf']}'** (relevance: {result['similarity']:.2f}):\n"
            response += f"   {result['text'][:200]}...\n\n"
        
        response += "\n**Key insights:**\n"
        response += "• The documents contain information about your query\n"
        response += "• Higher similarity scores indicate more relevant sections\n"
        response += "• You can adjust search settings for more precise results\n"
        
        return response

def main():
    st.set_page_config(
        page_title="ULTRA SIMPLE RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 ULTRA SIMPLE RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">100% Offline • No C++ Compilers • No ChromaDB • Just Works!</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = UltraSimpleRAG()
        st.session_state.processed = False
        st.session_state.search_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📁 PDF Setup")
        
        # Current directory info
        current_dir = Path(".")
        pdf_files = list(current_dir.glob("*.pdf"))
        
        if pdf_files:
            with st.expander(f"📚 Found {len(pdf_files)} PDF(s)", expanded=True):
                for pdf in pdf_files:
                    size_mb = pdf.stat().st_size / (1024 * 1024)
                    st.write(f"• **{pdf.name}** ({size_mb:.2f} MB)")
        else:
            st.warning("No PDF files found in current directory")
            st.info("Put your PDF files in the same folder as this script")
        
        # Process button
        st.markdown("---")
        if st.button("🔄 Process All PDFs", type="primary", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                st.session_state.processed = st.session_state.rag.process_pdfs(".")
        
        # Search settings
        st.markdown("---")
        st.header("🔍 Search Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Results", 1, 10, 5)
        with col2:
            min_sim = st.slider("Min Similarity", 0.0, 1.0, 0.1, 0.05)
        
        # System info
        st.markdown("---")
        st.header("📊 System Status")
        
        if st.session_state.processed:
            st.markdown('<div class="success-box">✅ PDFs Processed</div>', unsafe_allow_html=True)
            if hasattr(st.session_state.rag, 'text_chunks'):
                st.metric("Text Chunks", len(st.session_state.rag.text_chunks))
                st.metric("PDF Files", len(set(st.session_state.rag.pdf_names)))
        else:
            st.markdown('<div class="info-box">⏳ Ready to Process</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("---")
        st.header("⚡ Quick Actions")
        
        if st.button("🧹 Clear Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.button("📁 Show Folder", use_container_width=True):
            folder_path = Path(".").absolute()
            st.info(f"Current folder: {folder_path}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        st.header("💬 Ask Your PDFs")
        
        query = st.text_area(
            "Enter your question:",
            height=120,
            placeholder="Example: What are the main findings in the documents?\nOr: Explain the methodology used...",
            help="Type your question about the PDF content here"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            search_clicked = st.button("🔍 Search & Analyze", type="primary", use_container_width=True)
        with col_btn2:
            example_clicked = st.button("💡 Example", use_container_width=True)
        
        if example_clicked:
            query = "What are the key points or main topics discussed?"
            st.rerun()
        
        if search_clicked and query:
            if not st.session_state.processed:
                st.warning("⚠️ Please process PDFs first! Click the 'Process All PDFs' button in the sidebar.")
            else:
                with st.spinner("Searching through documents..."):
                    results = st.session_state.rag.search(
                        query, 
                        top_k=top_k, 
                        min_similarity=min_sim
                    )
                
                # Store in history
                st.session_state.search_history.append({
                    'query': query,
                    'results': results,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant section(s)")
                    
                    # Display results
                    st.markdown("---")
                    st.header("📄 Search Results")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"Result {i}: {result['pdf']} (Score: {result['similarity']:.3f})",
                            expanded=(i == 1)
                        ):
                            st.write(f"**Source:** {result['pdf']}")
                            st.write(f"**Chunk ID:** {result['chunk_id'] + 1}")
                            st.write(f"**Word Count:** {result['info']['word_count']}")
                            st.write(f"**Relevance Score:** {result['similarity']:.3f}")
                            st.markdown("---")
                            st.write("**Content:**")
                            st.write(result['text'])
                    
                    # AI Response
                    st.markdown("---")
                    st.header("🤖 AI Analysis")
                    
                    ai_response = st.session_state.rag.create_response(query, results)
                    st.write(ai_response)
                    
                    # Export results
                    st.markdown("---")
                    st.header("📥 Export Results")
                    
                    export_text = f"Query: {query}\n"
                    export_text += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    export_text += f"Found: {len(results)} results\n\n"
                    
                    for i, result in enumerate(results, 1):
                        export_text += f"--- Result {i} ---\n"
                        export_text += f"PDF: {result['pdf']}\n"
                        export_text += f"Similarity: {result['similarity']:.3f}\n"
                        export_text += f"Content:\n{result['text']}\n\n"
                    
                    st.download_button(
                        label="💾 Download Results as Text",
                        data=export_text,
                        file_name=f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.warning("No relevant results found. Try:")
                    st.write("1. Lowering the 'Min Similarity' threshold")
                    st.write("2. Using different search terms")
                    st.write("3. Checking if your PDFs contain text (not scanned images)")
    
    with col2:
        st.header("💡 How to Use")
        
        st.markdown("""
        **Step-by-Step:**
        1. 📁 Put PDFs in same folder as this script
        2. 🔄 Click "Process All PDFs"
        3. 💬 Ask questions in the text box
        4. 🔍 Click "Search & Analyze"
        
        **Tips for Better Results:**
        • Use specific, clear questions
        • Start with default settings
        • Process PDFs only once
        
        **What's Working:**
        • ✅ PDF text extraction
        • ✅ Offline processing
        • ✅ Simple similarity search
        • ✅ No C++ compilers needed
        """)
        
        # Recent searches
        if st.session_state.search_history:
            st.markdown("---")
            st.header("🕐 Recent Searches")
            
            for i, search in enumerate(st.session_state.search_history[-3:], 1):
                st.caption(f"{search['timestamp']}: {search['query'][:50]}...")
        
        # System info
        st.markdown("---")
        st.header("🛠️ System Info")
        
        with st.expander("Technical Details"):
            st.write("**Python Packages:**")
            st.write("- Streamlit 1.28.0")
            st.write("- PyMuPDF 1.23.8")
            st.write("- NumPy 1.26.4")
            st.write("**Embedding Method:**")
            st.write("Hash-based word frequency")
            st.write("**Similarity:** Cosine similarity")

if __name__ == "__main__":
    main()
