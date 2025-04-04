# pip install -U streamlit langchain langchain-openai langchain-community faiss-cpu python-dotenv

# rag_chatbot_with_memory_final.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot with Memory", layout="wide")
st.title("🤖 RAG Chatbot with Chat Memory (LangChain v0.2+)")

uploaded_file = st.file_uploader("📄 Upload a knowledge base (.txt file)", type=["txt"])

if uploaded_file:
    with st.spinner("📚 Processing document..."):
        temp_path = "./kb.txt"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = TextLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",  # 🔐 Required for proper tracking
            output_key="answer"         # ✅ FIX: Store only 'answer' in memory
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
        )

    st.success("✅ Document indexed. Start chatting!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("💬 Ask a question")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain.invoke({
                "question": user_query,
                "chat_history": st.session_state.chat_history
            })

            st.session_state.chat_history.append(("human", user_query))
            st.session_state.chat_history.append(("ai", result["answer"]))

            st.subheader("🧠 Chat History")
            for speaker, msg in st.session_state.chat_history[-6:]:
                st.markdown(f"**{speaker.title()}:** {msg}")

            with st.expander("📚 Sources Used"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)

else:
    st.info("⬆️ Upload a `.txt` file to get started.")

# Questions:
# How many leave days do I get?
# Can I carry forward my leaves?
# What’s the reimbursement deadline?
# Is remote work allowed all week?
# What’s the notice period before leaving?
# Which platform should we use for communication?