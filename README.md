# Multi-Document PDF Chat Application
This is a Streamlit-based application that enables you to upload multiple PDF files, extract their content, and ask questions across all the documents using embeddings and an Azure OpenAI API.

# Features
- Multi-file Upload: Upload multiple PDF files for processing.
- Text Extraction: Extract and chunk text from uploaded PDF files.
- Embeddings with FAISS: Create a FAISS index for efficient similarity searches using HuggingFace embeddings.
- Azure OpenAI Integration: Generate answers by querying the combined context from relevant document chunks.
- Interactive UI: Streamlit-powered interface for easy usage.


# Requirements
- Clone the git repository  
- Activate the environment  
- Install the dependencies: `pip install -r requirements.txt`  
- Add a `.env` file in the root directory with your Azure OpenAI API credentials
- Run command : `streamlit run main.py`











 
