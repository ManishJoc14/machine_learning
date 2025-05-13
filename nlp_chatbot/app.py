import streamlit as st
import os
from utils.file_manager import save_uploaded_file
from model.train import train_model
from utils.ui_helpers import (
    show_training_ui,
    handle_training,
    show_chat_ui,
    create_uploaded_file_from_path,
)

# Streamlit page config
st.set_page_config(page_title="NLP Chatbot", page_icon="🧠")

# Session state
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Training
if not st.session_state.model_trained:
    uploaded_file, use_default, epochs = show_training_ui()

    if uploaded_file:
        # Use uploaded file directly
        handle_training(uploaded_file, epochs, save_uploaded_file, train_model)

    elif use_default:
        default_path = "data/default_intent.json"
        if not os.path.exists(default_path):
            st.error("❌ Default intent file not found at `data/default_intent.json`.")
            st.stop()

        fake_uploaded_file = create_uploaded_file_from_path(
            default_path, "default_intent.json"
        )
        handle_training(fake_uploaded_file, epochs, save_uploaded_file, train_model)

# Step 2: Chat
if st.session_state.model_trained:
    if os.path.exists("saved_model/model.pth"):
        show_chat_ui()
    else:
        st.warning("⚠️ Model not found. Please train the model first.")
