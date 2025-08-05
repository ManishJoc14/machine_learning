# 🇳🇵 Nepal Constitution Chatbot

A simple yet powerful chatbot to answer questions related to the **Constitution of Nepal**. Built using:

- 📚 LangChain
- 🦙 LLaMA 3 via Groq API
- 💾 FAISS for vector search
- 📄 PDF text extraction (PyMuPDF)
- 🧠 HuggingFace Embeddings
- 🌐 Streamlit for UI

---

##  Features

- Ask natural questions related to the Constitution of Nepal.
- Fast and intelligent answers powered by LLaMA-3-70B via Groq.
- Local document processing with vector similarity search using FAISS.
- Fully interactive Streamlit chat interface.

---

##  How It Works (RAG Pipeline)

This chatbot uses the **RAG (Retrieval-Augmented Generation)** technique.

### What is RAG?

> **RAG** = *Retrieval* of relevant document chunks → *Augmented* into prompt → *Generated* answer from LLM.

Instead of relying only on the model's memory, we fetch real document context and inject it into the model’s input. This results in far more accurate, grounded answers.

---

##  Pipeline Steps

### 1. Text Extraction
We use `PyMuPDF` to extract the entire text from `Constitution-of-Nepal.pdf`.

### 2. Chunking & Embeddings
The text is split into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter`, then embedded using `HuggingFaceEmbeddings`.

### 3. Vector Store (FAISS)
The embedded chunks are stored in a local FAISS index to support fast similarity-based retrieval.

### 4. LLM via Groq API
For answering, we use **LLaMA 3-70B (8192 context)** via `langchain_groq.ChatGroq`.

### 5. Streamlit Chat UI
The interface is built using Streamlit’s `st.chat_message` with persistent chat history.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ManishJoc14/machine_learning
cd .\RAG\Nepal-Constitution-Chatbot\
````

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# Or: source venv/bin/activate (on Unix/macOS)
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file:

```
LANGCHAIN_TRACING_V2=true
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_ENDPOINT=your_langchain_endpoint
```

> 🔑 You can get a Groq API key from [https://console.groq.com](https://console.groq.com)
> 🔑 You can get a LangChain API key from [https://smith.langchain.com/](https://smith.langchain.com/)

### 5. Run the App

```bash
streamlit run app.py
```

---

## Example Usage

Ask questions like:

* "What is the role of the President according to the Constitution?"
* "How is the Prime Minister elected?"
* "What are the fundamental rights mentioned?"

---

## Model Info

We are using:

* Model: `llama3-70b-8192`
* Provider: [Groq](https://groq.com/)
* Embeddings: `HuggingFaceEmbeddings`

---

## How it works ?

User types question
    ↓
Chat UI passes input to get_response()
    ↓
get_response() runs:
    → FAISS retrieves top chunks from Constitution
    → LLaMA-3 gets query + chunks
    → Responds with an answer
    ↓
Chat UI displays answer and updates history

---

## 📄 Acknowledgements

* [LangChain](https://python.langchain.com/)
* [Groq](https://groq.com/)
* [HuggingFace](https://huggingface.co/)
* [Streamlit](https://streamlit.io/)
* [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)

---

## 📬 License

This project is for educational purposes only. Constitution content belongs to the Government of Nepal.
