from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    # Initialize the text splitter with specified chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split the text and return the list of chunks
    return splitter.split_text(text)
