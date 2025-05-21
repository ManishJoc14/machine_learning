from core.response import get_response
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def main():
    st.set_page_config(page_title="Nepal Constitution Chatbot")
    st.title("📜 Nepal Constitution Chatbot")
    st.markdown("Ask anything about the Constitution of Nepal.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for sender, msg in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(msg)

    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        response = get_response(user_input)
        st.session_state.chat_history.append(("Bot", response))
        st.rerun()


if __name__ == "__main__":
    main()
