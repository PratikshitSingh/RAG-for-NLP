# Objective:
# To enhance the Hybrid RAG-based Clause Finder by adding a PDF export feature, allowing users to:

# Upload and search across multiple legal contracts
# Ask clause-specific legal questions using a Hybrid RAG pipeline (semantic + keyword)
# View the AI-generated response along with retrieved source text
# Download the full result as a structured PDF report with question, answer, and citations
# All implemented using LangChain + Streamlit + FPDF for a seamless and offline-ready user experience.

# What’s new?
# Downloadable PDF report with formatted Q/A and source citations
# Bidirectional: Upload + Export
# Ready for reporting, sharing, compliance submission
# Tech Stack Addition  - Added fpdf + clean text formatting for export
# Results can now be saved, stored, or emailed as PDFs

# What this version demonstrates:
# End-to-end enterprise-ready document QA system
# Hybrid retrieval (semantic + keyword) using LangChain
# Fully local pipeline (no APIs required) with FLAN-T5
# Beautifully formatted PDF export with wrapped text, special character handling, and document source attribution
# Perfect for:
# - Legal reviews
# - Client contract analysis
# - Audit/compliance documentation

# hybrid_clause_finder_app.py

import os
import io
import re # for regex operations
import PyPDF2
import tempfile
import textwrap # for text wrapping
import streamlit as st
import numpy as np
from typing import List
from fpdf import FPDF #  to generate a downloadable PDF report

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="LangChain Hybrid RAG - Clause Finder", layout="centered")

# ---------------------- PDF Loading ----------------------
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

# ---------------------- PDF Export Function ----------------------
# Creates a downloadable PDF from user query, generated answer, and retrieved document chunks.
# def create_pdf(question, answer, sources):
#     pdf = FPDF() # Create a PDF object
#     pdf.add_page() # Add a new page
#     pdf.set_auto_page_break(auto=True, margin=15) # Set auto page break
#     pdf.set_font("Helvetica", size=12) # Set font for the PDF

#     # clean_text() - fixes curly quotes, dashes, spacing issues
#     def clean_text(text, width=80):
#         text = text.replace("\u2013", "-").replace("\u2014", "-")
#         # replace("\u2013", "-"): replaces en-dash with hyphen
#         # \u2013 - It means "en dash" and is used to represent a range of values (e.g., 2010–2020)
#         text = text.replace("\u2018", "'").replace("\u2019", "'")
#         # replace("\u2018", "'"): replaces left single quotation mark with apostrophe
#         # \u2018 - It means "left single quotation mark" and is used to represent a quote
#         text = text.replace("\u201c", '"').replace("\u201d", '"')
#         # replace("\u201c", '"'): replaces left double quotation mark with double quote
#         # \u201c - It means "left double quotation mark" and is used to represent a quote
#         text = re.sub(r"\s+", " ", text)
#         words = text.split(" ")
#         cleaned = []
#         for word in words:
#             if len(word) > width:
#                 word = "\n".join(textwrap.wrap(word, width=width))
#             cleaned.append(word)
#         return " ".join(cleaned)

#     def write_section(title, text):
#         pdf.set_font("Helvetica", "B", 12)
#         pdf.multi_cell(0, 10, title) # Write title
#         pdf.set_font("Helvetica", "", 12) # Set font for text
#         for line in textwrap.wrap(clean_text(text), width=100): # Wrap text to fit the page width
#             try:
#                 pdf.multi_cell(0, 10, line) # Write text
#             except:
#                 line = line.encode("latin-1", "replace").decode("latin-1") # Handle encoding issues
#                 pdf.multi_cell(0, 10, line) # Write text
#         pdf.ln() # Add a line break

#     write_section("Question:", question)
#     write_section("Answer:", answer)
#     pdf.set_font("Helvetica", "B", 12)
#     pdf.multi_cell(0, 10, "Sources:")
#     pdf.set_font("Helvetica", "", 12)
#     for i, doc in enumerate(sources):
#         write_section(f"Source {i+1}: {doc.metadata['source']}", doc.page_content)

#     buffer = io.BytesIO() # Create a buffer to hold the PDF
#     pdf_output = pdf.output(dest='S').encode('latin-1')  # PDF bytes
#     buffer.write(pdf_output) # Write PDF bytes to buffer
#     buffer.seek(0) # Move the buffer cursor to the beginning
#     return buffer # Return the buffer containing the PDF bytes

# ...existing code...

def create_pdf(question, answer, sources):
    pdf = FPDF()  # Create a PDF object
    pdf.add_page()  # Add a new page
    pdf.set_auto_page_break(auto=True, margin=15)  # Set auto page break
    pdf.set_font("Helvetica", size=12)  # Set font for the PDF

    # Helper function to clean and encode text for PDF
    def clean_text(text, width=80):
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = re.sub(r"\s+", " ", text)
        words = text.split(" ")
        cleaned = []
        for word in words:
            if len(word) > width:
                word = "\n".join(textwrap.wrap(word, width=width))
            cleaned.append(word)
        return " ".join(cleaned)

    # Helper function to encode text in latin-1
    def encode_text_latin1(text):
        try:
            return text.encode("latin-1", "replace").decode("latin-1")
        except Exception:
            return "Encoding Error"

    def write_section(title, text):
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 10, encode_text_latin1(title))  # Write title
        pdf.set_font("Helvetica", "", 12)  # Set font for text
        for line in textwrap.wrap(clean_text(text), width=100):  # Wrap text to fit the page width
            pdf.multi_cell(0, 10, encode_text_latin1(line))  # Write text
        pdf.ln()  # Add a line break

    write_section("Question:", question)
    write_section("Answer:", answer)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 10, encode_text_latin1("Sources:"))
    pdf.set_font("Helvetica", "", 12)
    for i, doc in enumerate(sources):
        write_section(f"Source {i+1}: {doc.metadata['source']}", doc.page_content)

    buffer = io.BytesIO()  # Create a buffer to hold the PDF
    pdf_output = pdf.output(dest='S').encode('latin-1')  # PDF bytes
    buffer.write(pdf_output)  # Write PDF bytes to buffer
    buffer.seek(0)  # Move the buffer cursor to the beginning
    return buffer  # Return the buffer containing the PDF bytes

# ...existing code...

# ---------------------- Streamlit UI ----------------------
st.title("⚖️ Hybrid RAG: Clause Finder (LangChain + PDF Export)")
uploaded_files = st.file_uploader("📎 Upload legal contracts (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🧠 Chunking and Indexing PDFs..."):
        docs = load_pdf_chunks(uploaded_files)

        # 🔍 Build Hybrid Retrievers
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedder)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        # 🧠 LLM with FLAN-T5
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever,
            return_source_documents=True,
            chain_type="stuff"
        )

    # 🧾 User query + result
    user_query = st.text_input("💬 Ask a clause-based legal question:", placeholder="e.g., What does the termination clause say?")
    if st.button("🔎 Get Answer"):
        with st.spinner("Running Hybrid RAG..."):
            result = qa_chain(user_query)
            answer = result["result"]
            sources = result["source_documents"]

            st.subheader("📌 Answer")
            st.write(answer)

            st.subheader("📄 Sources")
            for doc in sources:
                st.markdown(f"**Source:** `{doc.metadata['source']}`")
                st.write(doc.page_content)
                st.markdown("---")

            # 📥 PDF Export
            pdf_bytes = create_pdf(user_query, answer, sources)
            st.download_button(
                label="📥 Download Response as PDF",
                data=pdf_bytes,
                file_name="clause_answer.pdf",
                mime="application/pdf"
            )
else:
    st.info("⬆️ Please upload one or more legal PDFs to begin.")
