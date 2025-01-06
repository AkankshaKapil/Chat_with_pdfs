import os
import requests
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Function to extract text from a PDF file
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Function to chunk the extracted PDF text using LangChain
def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Function to call Azure OpenAI API and get a response
def get_response_from_openai(context, question):
    endpoint = os.getenv("ENDPOINT")
    deployment_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    api_key = os.getenv("API_KEY")

    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    
    # Enhanced system message for multi-document context
    system_message = """You are a helpful assistant analyzing multiple documents. 
    When providing answers, cite the specific document or context where you found the information."""
    
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context from documents: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Initialize session state for storing processed documents
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {}
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

# Streamlit app
st.title("Multi-Document PDF Chat")

# Step 1: File upload - Now supports multiple files
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Process new uploads
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_docs:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Extract and chunk text
                pdf_text = extract_pdf_text(uploaded_file)
                chunks = chunk_text(pdf_text)
                
                # Add source information to chunks
                chunks_with_source = [
                    f"[Source: {uploaded_file.name}] {chunk}"
                    for chunk in chunks
                ]
                
                # Store processed chunks
                st.session_state.processed_docs[uploaded_file.name] = chunks_with_source
                st.success(f"Processed {uploaded_file.name}")

    # Display processed documents
    st.write("### Processed Documents:")
    for doc_name in st.session_state.processed_docs:
        st.write(f"- {doc_name} ({len(st.session_state.processed_docs[doc_name])} chunks)")

    # Combine all chunks from all documents
    all_chunks = []
    for doc_chunks in st.session_state.processed_docs.values():
        all_chunks.extend(doc_chunks)

    # Initialize or update FAISS index
    if len(all_chunks) > 0:
        with st.spinner("Creating/updating FAISS index..."):
            huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.faiss_index = FAISS.from_texts(all_chunks, embedding=huggingface_embeddings)
            st.success("FAISS index updated!")

    # Query interface
    st.write("### Ask Questions")
    query = st.text_input("Enter your question about the documents")

    # Search settings
    k_results = st.slider("Number of relevant chunks to consider", min_value=1, max_value=5, value=3)

    if query and st.session_state.faiss_index:
        with st.spinner("Searching across all documents..."):
            search_results = st.session_state.faiss_index.similarity_search(query, k=k_results)
            
            # Combine relevant chunks for context
            combined_context = "\n\n".join([result.page_content for result in search_results])
            
            # Get and display answer
            with st.spinner("Generating comprehensive answer..."):
                final_answer = get_response_from_openai(combined_context, query)
                st.write("### Answer:")
                st.success(final_answer)
                
            # Option to view source chunks
            if st.checkbox("Show source chunks"):
                st.write("### Relevant chunks used for the answer:")
                for i, result in enumerate(search_results, 1):
                    st.info(f"Chunk {i}:\n{result.page_content}")

    # Add option to clear processed documents
    if st.button("Clear all processed documents"):
        st.session_state.processed_docs = {}
        st.session_state.faiss_index = None
        st.success("All documents cleared! You can upload new files now.")