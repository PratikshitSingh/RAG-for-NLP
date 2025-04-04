# Use Case Title:
# 🔗 “LangChain-powered Research Paper Selector with CrossEncoder Reranking”

# Problem Statement:
# Researchers often face the challenge of sifting through hundreds of academic papers to find the most relevant ones for a specific research question. Traditional keyword-based or even semantic-only search can return low-quality results due to the lack of deep contextual understanding.

# This solution introduces a modular and powerful RAG system using LangChain with:

# Semantic retrieval using HuggingFaceEmbeddings + FAISS
# CrossEncoder reranking for fine-grained, query-aware ranking
# User-friendly UI via Streamlit for exploring, selecting, and exporting top papers
# It transforms manual paper filtering into an intelligent, interactive assistant that retrieves, reranks, and curates research literature.

# What's new?
# Uses LangChain retrievers
# Wrapped in langchain.schema.Document objects with metadata
# LangChain Retriever API abstraction
# Encapsulated in a CrossEncoderReranker class
# LangChain Embeddings, Vectorstore, and Retriever API
# CSV export (retained)
# Chain-compatible and plug-and-play with other LangChain tools

# pip install streamlit langchain faiss-cpu pandas sentence-transformers
# streamlit run langchain_rag_selector.py


# langchain_rag_selector.py

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever # compresses retrieved documents for smaller LLM inputs.

# ---------------------- Streamlit Setup ---------------------- #
st.set_page_config(page_title="LangChain RAG Selector", layout="wide")
st.title("🔗 LangChain + CrossEncoder: Academic Paper RAG")

uploaded_file = st.file_uploader("📎 Upload your papers.csv (with `title` and `abstract`)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['content'] = df['title'] + ". " + df['abstract']
    docs = [
        Document(page_content=row['content'], metadata={"title": row['title'], "abstract": row['abstract']})
        for _, row in df.iterrows()
    ]

    # ---------------------- Embed & Store ---------------------- #
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embed_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

    # ---------------------- CrossEncoder for Reranking ---------------------- #
    st.write("🧠 Loading reranker...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6")

    # Initializes a custom class that will use:
    # a bi-encoder retriever to quickly find top 30 documents.
    # a cross-encoder to rerank them more accurately.
    class CrossEncoderReranker:
        def __init__(self, retriever, encoder):
            self.retriever = retriever
            self.encoder = encoder

        def get_relevant_documents(self, query: str) -> List[Document]:
            docs = self.retriever.get_relevant_documents(query)
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.encoder.predict(pairs)
            reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in reranked[:10]]

    reranked_retriever = CrossEncoderReranker(retriever, cross_encoder)

    # ---------------------- Query Input ---------------------- #
    user_query = st.text_input("💬 Ask a research question", placeholder="e.g., What are the latest approaches in few-shot learning?")

    if st.button("🔍 Retrieve & Rerank") and user_query:
        top_docs = reranked_retriever.get_relevant_documents(user_query)

        st.subheader("📚 Top Retrieved Papers")
        selected = []

        for i, doc in enumerate(top_docs):
            title = doc.metadata['title']
            abstract = doc.metadata['abstract']
            with st.expander(f"📄 {title}"):
                st.write(abstract)
                if st.checkbox("✅ Select this paper", key=f"{title}_{i}"):
                    selected.append({
                        "title": title,
                        "abstract": abstract
                    })

        if selected:
            sel_df = pd.DataFrame(selected)
            st.download_button("📥 Download Selected as CSV", data=sel_df.to_csv(index=False),
                               file_name="selected_papers.csv", mime="text/csv")
else:
    st.info("⬆️ Please upload your CSV to begin.")
