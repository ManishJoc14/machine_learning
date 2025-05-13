import io
import os
import streamlit as st
from chatbot.predict import get_response
from utils.file_manager import save_uploaded_file

# Function to display the training UI
def show_training_ui():
    st.subheader("Train Your Chatbot")

    # File uploader for custom intent JSON
    uploaded_file = st.file_uploader("Upload your intent.json", type="json")

    # Checkbox for using the default intent JSON
    use_default = st.checkbox("Use default intent.json")

    # Number of epochs for training
    epochs = st.number_input(
        "Training Epochs", min_value=1000, max_value=20000, value=1000, step=100
    )

    return uploaded_file, use_default, epochs


# Function to handle the training process
def handle_training(uploaded_file, use_default, epochs, train_model_fn):
    if use_default:
        # Directly read the default file into memory
        default_path = "data/default_intent.json"
        if not os.path.exists(default_path):
            st.error("❌ Default intent file not found at `data/default_intent.json`.")
            return

        with open(default_path, "rb") as f:
            file_content = f.read()

        # Train the model using the default file content
        with open("data/intents.json", "wb") as f:
            f.write(file_content)  # Save it to a temporary location for training
        st.success("✅ Using default intent.json for training.")
    elif uploaded_file:
        # Save the uploaded file directly
        save_uploaded_file(uploaded_file, "data/intents.json")
        st.success("✅ File uploaded successfully!")

    # Proceed to train the model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            train_model_fn("data/intents.json", epochs)
        st.success("✅ Model trained and saved!")
        st.session_state.model_trained = True
        st.rerun()


# Function to display the chatbot UI
def show_chat_ui():
    st.header("Chat with Your Bot")

    # Display chat history
    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(msg)

    # Input for user to type a message
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append(("You", user_input))

        # Get the bot's response and append it to chat history
        response = get_response(user_input)
        st.session_state.chat_history.append(("Bot", response))

        st.rerun()


def create_uploaded_file_from_path(path, name="default.json"):
    """Simulate a Streamlit uploaded file from a local file path."""
    with open(path, "rb") as f:
        file_content = f.read()
        uploaded_file = io.BytesIO(file_content)
        uploaded_file.name = name
    return uploaded_file
