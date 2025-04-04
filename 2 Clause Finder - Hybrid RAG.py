# Legal teams often need to search across **multiple contracts** to answer precise, clause-based questions like *“What does the termination clause say?”* or *“Is there a non-compete?”* Traditional search methods fail to combine the benefits of both **semantic understanding** and **exact keyword matching**, especially when working across many documents.

# **Objective:**  
# To develop a **Streamlit-based Hybrid RAG Clause Finder** that allows users to:

# - Upload multiple legal contract PDFs  
# - Ask **clause-specific legal questions**  
# - Use **both semantic (FAISS)** and **keyword-based (Whoosh BM25)** retrieval  
# - Show **retrieved clauses along with their source document**  
# - Display answers using **FLAN-T5**, a local instruction-following LLM

# All within a clean, interactive web interface — fully **offline and enterprise-friendly**.

# What's new?
# Full Streamlit web UI
# Interactive text input box
# Upload from browser UI (via file_uploader)
# Explicitly shows document source for each chunk
# Shows retrieved context chunks with file names
# End-user focused: titles, loading spinners, expandable chunks
# Production-ready clause assistant for law/compliance teams

import streamlit as st
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser
import tempfile

st.set_page_config(page_title="Clause Finder - Hybrid RAG", layout="centered")
st.title("⚖️ Hybrid RAG Clause Finder")
st.markdown("Upload legal contracts and ask clause-based questions. Hybrid = Semantic + Keyword search.")

# ------------------ Upload PDFs ------------------

uploaded_files = st.file_uploader("📎 Upload one or more legal contracts (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save PDFs
    contracts_dir = "contracts"
    os.makedirs(contracts_dir, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(contracts_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    # ------------------ Load & Chunk ------------------

    def load_chunks(folder_path, chunk_size=300):
        chunks, sources = [], []
        for fname in os.listdir(folder_path):
            if fname.endswith(".pdf"):
                path = os.path.join(folder_path, fname)
                reader = PyPDF2.PdfReader(path)
                text = "".join([page.extract_text() for page in reader.pages])
                text = text.replace("\n", " ")
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    chunks.append(chunk)
                    sources.append(fname)
        return chunks, sources

    with st.spinner("🔍 Indexing documents..."): # Shows a spinner while chunking and loading the data.
        chunks, sources = load_chunks(contracts_dir)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks, convert_to_tensor=False)
        dim = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(np.array(embeddings))

        # Whoosh BM25 Index
        schema = Schema(content=TEXT(stored=True), path=ID(stored=True))
        index_dir = tempfile.mkdtemp()
        ix = create_in(index_dir, schema)
        writer = ix.writer()
        for chunk, src in zip(chunks, sources):
            writer.add_document(content=chunk, path=src)
        writer.commit()

        qa = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

    # ------------------ Retrieval + Generation ------------------

    def hybrid_retrieve(query, top_k=3):
        query_vec = embedder.encode([query])
        _, indices = faiss_index.search(np.array(query_vec), top_k)
        semantic_results = [(chunks[i], sources[i]) for i in indices[0]]

        with ix.searcher() as searcher:
            parser = QueryParser("content", schema=ix.schema)
            parsed = parser.parse(query)
            results = searcher.search(parsed, limit=top_k)
            keyword_results = [(r['content'], r['path']) for r in results]

        merged = list(dict.fromkeys(semantic_results + keyword_results))
        return merged[:top_k]

    def generate_answer(query, hybrid_results):
        context = "\n".join([r[0] for r in hybrid_results])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = qa(prompt)[0]['generated_text']
        return response.strip()

    # ------------------ UI Interaction ------------------

    query = st.text_input("💬 Ask a legal clause-related question:", placeholder="e.g., What does the termination clause say?")
    if st.button("🔎 Answer"):
        with st.spinner("Retrieving and answering..."):
            results = hybrid_retrieve(query)
            answer = generate_answer(query, results)

            st.subheader("📌 Answer:")
            st.write(answer)

            # Shows the actual text chunks used for answering, along with filenames — very useful for transparency and source citation.
            with st.expander("📚 Retrieved Chunks (Top Matches)"):
                for chunk, src in results:
                    st.markdown(f"**Source:** `{src}`")
                    st.write(chunk)
                    st.markdown("---")

else:
    st.info("⬆️ Upload PDF files to begin.")
