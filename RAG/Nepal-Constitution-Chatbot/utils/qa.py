import os
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def ask_question(vector_store, question: str) -> str:
    # Initialize the language model with the specified API key and model name
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192")

    # Create a RetrievalQA chain using the language model and retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vector_store.as_retriever()
    )

    # Invoke the chain with the user's question and get the answer
    answer = qa_chain.invoke(question)

    return answer["result"]
