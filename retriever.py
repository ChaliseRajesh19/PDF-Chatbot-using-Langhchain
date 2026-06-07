import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_retriever(index_path,k):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
    print(f"FAISS index loaded from {index_path}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print(f"Retriever created with k={k}")
    return retriever

if __name__ == "__main__":
    index_path = "faiss_index"  # Directory where the FAISS index is saved
    k = 5  # Number of similar documents to retrieve

    retriever = load_retriever(index_path,k)


    query = "What are the main topics covered in the notes?"  # Example query
    docs = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} documents:")
    
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...")  # Print the first 200 characters of the content
        print()