import io
import streamlit as st
from utils.file_manager import save_uploaded_file
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
DEFAULT_INTENT_PATH = DATA_PATH / "default_intent.json"
INTENT_PATH = DATA_PATH / "intents.json"


def show_training_ui():
    st.subheader("Train Your Chatbot")
    uploaded_file = st.file_uploader("Upload your intent.json", type="json")
    use_default = st.checkbox("Use default intent.json")
    epochs = st.number_input(
        "Training Epochs", min_value=1000, max_value=20000, value=1000, step=100
    )
    return uploaded_file, use_default, epochs


def handle_training(uploaded_file, use_default, epochs, train_model_fn):
    if use_default:
        if not DEFAULT_INTENT_PATH.exists():
            st.error("❌ Default intent file not found.")
            return
        with open(DEFAULT_INTENT_PATH, "rb") as f:
            content = f.read()
        with open(INTENT_PATH, "wb") as f:
            f.write(content)
        st.success("✅ Using default intent.json for training.")
    elif uploaded_file:
        save_uploaded_file(uploaded_file, INTENT_PATH)
        st.success("✅ File uploaded successfully!")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            train_model_fn(INTENT_PATH, epochs)
        st.success("✅ Model trained and saved!")
        st.session_state.model_trained = True
        st.rerun()


def show_chat_ui():
    from chatbot.predict import get_response

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
