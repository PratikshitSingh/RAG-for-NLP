# Objective:
# To build a modular, production-grade Hybrid RAG Clause Finder using LangChain, which:

# Parses legal contract PDFs
# Supports hybrid retrieval using FAISS (semantic) and BM25 (keyword) with weight tuning
# Leverages LangChain’s RetrievalQA chain to abstract away RAG complexity
# Uses FLAN-T5 locally for generation
# Returns both the answer and the source clauses
# All this in a streamlined, reusable LangChain-based architecture, enabling rapid experimentation, modular upgrades, and easy deployment.

# Whats new?
# LangChain abstraction of RAG components
# langchain.vectorstores.FAISS
# Built-in LangChain BM25Retriever
# EnsembleRetriever with tunable weights
# Wrapped with HuggingFacePipeline and integrated into LangChain
# End-to-end handled by RetrievalQA.from_chain_type()
# Structured source docs with metadata
# Modular, swappable components for scaling and testing

# What this version demonstrates?
# Production-ready LangChain RAG system for legal workflows
# Modular components for retriever, embedder, LLM, and retrieval+generation chains
# Easy customization for future upgrades:
# - Swap FLAN-T5 with Llama 2 or Mixtral
# - Use other retrievers like ParentDocumentRetriever, TimeWeightedRetriever, etc.
# - Add more vector stores like Weaviate or Pinecone
# - Integrate with LangChain Agents for complex workflows
# - Use LangChain Memory for stateful interactions
# - Add LangChain Callbacks for monitoring and logging
# Clean UI with extracted source chunks clearly shown
# Ready for multi-file analysis with attribution

# langchain_app.py

import os
import PyPDF2
import tempfile
import streamlit as st
from typing import List # List - Helps annotate that a function returns a list of documents.

import numpy as np
from sentence_transformers import SentenceTransformer

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings # Wraps SentenceTransformer for embedding support in LangChain.
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import EnsembleRetriever # Combines multiple retrievers (here: FAISS + BM25).

from transformers import pipeline
from langchain.chains import RetrievalQA #  A LangChain chain that handles retrieval + answer generation.

# ---------------------- PDF LOADER ---------------------- #
def load_pdf_chunks(files, chunk_size=300) -> List[Document]:
    documents = []
    for file in files:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        text = text.replace("\n", " ")
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            documents.append(Document(page_content=chunk, metadata={"source": file.name}))
    return documents

# ---------------------- STREAMLIT UI ---------------------- #
st.set_page_config(page_title="LangChain Hybrid RAG - Clause Finder", layout="centered")
st.title("⚖️ LangChain Hybrid RAG: Legal Clause Finder")
uploaded_files = st.file_uploader("📎 Upload legal contracts (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Parsing and chunking documents..."):
        docs = load_pdf_chunks(uploaded_files)

    # ---------------------- RETRIEVERS ---------------------- #
    with st.spinner("🔗 Building FAISS + BM25 Hybrid retrievers..."):
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Wraps MiniLM in LangChain’s HuggingFaceEmbeddings
        vectorstore = FAISS.from_documents(docs, embedder) # Builds a FAISS index from documents
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Creates a Retriever to return top-3 similar chunks

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3

        # Combines semantic and keyword retrievers with equal weight (hybrid power )
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

    # ---------------------- LLM Setup ---------------------- #
    with st.spinner("⚙️ Loading FLAN-T5 model..."):
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever,
            return_source_documents=True,
            chain_type="stuff"
        )

    # ---------------------- Ask Questions ---------------------- #
    user_query = st.text_input("💬 Ask a legal clause question", placeholder="e.g., What does the indemnity clause say?")

    if st.button("🧠 Get Answer"):
        with st.spinner("Running hybrid RAG..."):
            result = qa_chain(user_query)
            st.subheader("📌 Answer")
            st.write(result["result"])

            st.subheader("📄 Sources")
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** `{doc.metadata['source']}`")
                st.write(doc.page_content)
                st.markdown("---")
else:
    st.info("⬆️ Upload your legal contract PDFs to get started.")
