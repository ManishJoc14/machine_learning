import streamlit as st
from utils.ui_helpers import show_training_ui, handle_training, show_chat_ui
from model.train import train_model
from pathlib import Path

st.set_page_config(page_title="NLP Chatbot", page_icon="🧠")

# Session state initialization
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Training
if not st.session_state.model_trained:
    uploaded_file, use_default, epochs = show_training_ui()
    handle_training(uploaded_file, use_default, epochs, train_model)

# Step 2: Chat
if st.session_state.model_trained:
    model_path = Path(__file__).resolve().parent / "saved_model" / "model.pth"
    if model_path.exists():
        show_chat_ui()
    else:
        st.warning("⚠️ Model not found. Please train the model first.")
