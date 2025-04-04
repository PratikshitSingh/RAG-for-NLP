# Knowledge-Based QA System
# Problem Statement

# In many organizations, valuable knowledge resides in static documents like company policies, HR manuals, or IT protocols. Employees often struggle to extract relevant information quickly due to the unstructured nature of these documents.

# To solve this, we aim to build a Knowledge-Based QA System that leverages Retrieval-Augmented Generation (RAG) powered by OpenAI’s GPT models, allowing users to ask natural language questions and get accurate answers directly from the uploaded knowledge base.

# Objective
# Build a customizable RAG pipeline using LangChain and OpenAI to:
# Accept user-uploaded .txt documents as knowledge base.
# Automatically split, index, and embed the text using OpenAI embeddings.
# Retrieve relevant chunks based on a user’s question.
# Generate precise answers using OpenAI’s GPT-3.5-Turbo.
# Provide a Streamlit interface to make the system user-friendly and interactive.
# Ensure the system is easily reusable for different document types and NLP tasks (e.g., HR FAQ, IT troubleshooting, policy search).

# Expected Outcomes
# A real-time, interactive, knowledge-based Q&A application powered by OpenAI’s API.
# Accurate and context-aware answers for user queries sourced directly from the uploaded knowledge base.
# Seamless integration with .env for secure API key usage.
# A generic, reusable RAG pipeline that can be plugged into multiple industries.

# rag_openai_knowledge_qa_app.py

import os
import streamlit as st
from dotenv import load_dotenv

# FAISS is used to store and retrieve embeddings of the text chunks from your .txt file. It allows fast similarity search for relevant context during RAG.
from langchain.vectorstores import FAISS

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# Imports a smart text splitter that splits long documents into chunks (with some overlap).
# LLMs have input limits. So, instead of feeding the entire doc, you split it into chunks (e.g., 500 characters each with 100 overlap) for granular search and context retrieval.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imports the wrapper for OpenAI’s GPT models (like GPT-3.5-Turbo).
from langchain.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.schema import Document

# ------------------ Load OpenAI Key ------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="RAG OpenAI QA", layout="wide")
st.title("💡 Knowledge-Based QA using RAG + OpenAI")

uploaded_file = st.file_uploader("📄 Upload a knowledge base (.txt file)", type=["txt"])

if uploaded_file:
    with st.spinner("📚 Loading and chunking document..."):
        temp_path = f"./temp_uploaded.txt" # Saves the uploaded file to disk temporarily. Reads it in binary mode and writes the file contents.
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and chunk document
        loader = TextLoader(temp_path)
        raw_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(raw_docs)

        # Embed with OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # OpenAI LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3, model="gpt-3.5-turbo")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Document loaded and indexed. You can now ask questions!")

    query = st.text_input("🔎 Ask a question based on the uploaded document")

    if query:
        with st.spinner("🧠 Generating answer..."):
            result = qa_chain(query)
            st.subheader("📌 Answer")
            st.write(result["result"])

            st.subheader("📚 Source Chunks")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)

else:
    st.info("⬆️ Please upload a `.txt` file containing your knowledge base.")

