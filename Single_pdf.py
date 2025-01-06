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
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context} {question}"}
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

# Streamlit app
st.title("Chat with Your PDF")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
        st.success("PDF text extracted successfully!")

    # Step 2: Chunk text
    with st.spinner("Chunking text into manageable parts..."):
        chunks = chunk_text(pdf_text)
        st.success(f"Text chunked into {len(chunks)} parts!")

    # Step 3: Initialize HuggingFace Embeddings and FAISS
    huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    with st.spinner("Creating FAISS index..."):
        faiss_index = FAISS.from_texts(chunks, embedding=huggingface_embeddings)
        st.success("FAISS index created!")

    # Step 4: Query
    query = st.text_input("Enter your question about the document")

    if query:
        with st.spinner("Searching for relevant information..."):
            search_results = faiss_index.similarity_search(query, k=1)
            result = search_results[0].page_content if search_results else None
            if result:
                # st.write("Relevant Chunk:", result)
                
                # Step 5: Get final answer
                with st.spinner("Generating answer using Azure OpenAI..."):
                    final_answer = get_response_from_openai(result, query)
                    st.write("Answer from Azure OpenAI:")
                    st.success(final_answer)
            else:
                st.error("No relevant chunk found!")
