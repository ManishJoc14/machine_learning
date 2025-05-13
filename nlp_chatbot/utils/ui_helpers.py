import streamlit as st
import os
from chatbot.predict import get_response
from chatbot.chat_manager import add_to_history


def show_training_ui():
    st.header("NLP Chatbot Trainer")

    uploaded_file = st.file_uploader("Upload your intents.json", type=["json"])
    epochs = st.number_input(
        "Number of epochs", min_value=1000, max_value=20000, value=10000
    )

    return uploaded_file, epochs


def handle_training(uploaded_file, epochs, save_uploaded_file_fn, train_model_fn):
    save_uploaded_file_fn(uploaded_file, "data/intents.json")
    st.success("✅ File uploaded successfully!")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            train_model_fn("data/intents.json", epochs)
        st.success("✅ Model trained and saved!")
        st.session_state.model_trained = True
        st.rerun()


def show_chat_ui():
    st.header("Chat with Your Bot")

    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(msg)

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append(("You", user_input))

        response = get_response(user_input)
        st.session_state.chat_history.append(("Bot", response))

        st.rerun()
