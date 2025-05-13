import streamlit as st
from chatbot.predict import get_response


# Function to display the training UI
def show_training_ui():
    st.header("NLP Chatbot Trainer")

    # File uploader for intents.json
    uploaded_file = st.file_uploader("Upload your intents.json", type=["json"])
    # Input for specifying the number of training epochs
    epochs = st.number_input(
        "Number of epochs", min_value=1000, max_value=20000, value=10000
    )

    return uploaded_file, epochs


# Function to handle the training process
def handle_training(uploaded_file, epochs, save_uploaded_file_fn, train_model_fn):
    # Save the uploaded file to a specific location
    save_uploaded_file_fn(uploaded_file, "data/intents.json")
    st.success("✅ File uploaded successfully!")

    # Train the model when the button is clicked
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Call the training function with the uploaded file and epochs
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
