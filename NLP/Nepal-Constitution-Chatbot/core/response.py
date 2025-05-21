from pathlib import Path
from utils.loader import extract_text_from_pdf
from utils.splitter import split_text
from utils.vector_store import create_vector_store
from utils.qa import ask_question
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


# Load PDF and build vector store once
# text = extract_text_from_pdf()
# chunks = split_text(text)
# # create vector store if not created.
# vector_store = create_vector_store(chunks)
# # save the vector store if not saved.
# vector_store.save_local(VECTORSTORE_PATH)

# If vector store is saved
# Load vector store 
embedding = HuggingFaceEmbeddings()
vector_store = FAISS.load_local(
    VECTORSTORE_PATH, embeddings=embedding, allow_dangerous_deserialization=True
)


def get_response(user_input: str) -> str:
    return ask_question(vector_store, user_input)
