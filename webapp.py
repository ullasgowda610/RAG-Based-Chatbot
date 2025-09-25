import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf

# Sidebar - Upload PDF
st.sidebar.title(":orange[Upload your Document (PDF Only)]")
file_uploader = st.sidebar.file_uploader("Upload file", type=["pdf"])

# Main Page
st.title(":green[RAG Based Chatbot]")
st.write("""
Follow the steps to use this application:
1. Upload your PDF document in the sidebar  
2. Write your query below and start chatting with the bot  
""")

# Only run RAG if PDF is uploaded
if file_uploader:
    file_text = text_extractor_pdf(file_uploader)

    # Configure LLM
    key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Configure embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # safer for most setups
    )

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Create FAISS Vector Store
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    # Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # RAG pipeline
    def generate_response(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        You are a helpful assistant using RAG.
        Here is the context: {context}
        The query asked by the user is as follows: {query}
        """

        content = llm_model.generate_content(prompt)
        return content.text

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Append user message
        st.session_state.history.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate bot response
        response = generate_response(prompt)
        st.session_state.history.append({"role": "assistant", "text": response})

        with st.chat_message("assistant"):
            st.markdown(response)
