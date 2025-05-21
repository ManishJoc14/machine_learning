from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path


def create_vector_store(text_chunks: List[str]) -> FAISS:
    # Initialize the HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings()

    # Create a FAISS vector store from the provided text chunks using the embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Return the vector store
    return vector_store
