import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def loading_documents(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {file_path}")
    return pages

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks,index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    print(f"Vector store created and saved to {index_path}")


if __name__ == "__main__":
    file_path = "data/notes.pdf"  # Update with your PDF file path
    index_path = "faiss_index"  # Directory to save the FAISS index

    pages = loading_documents(file_path)
    chunks = split_documents(pages)
    create_vector_store(chunks,index_path)