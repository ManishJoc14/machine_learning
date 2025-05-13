import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.preprocess import preprocess_data
from model.model_arch import ChatbotModel
import streamlit as st


def train_model(json_path, epochs):
    X, Y, input_size, output_size = preprocess_data(json_path)
    model = ChatbotModel(input_size, 512, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float("inf")
    patience, counter = 500, 0

    # Streamlit progress bar and text
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            os.makedirs("saved_model", exist_ok=True)
            torch.save(model.state_dict(), "saved_model/model.pth")
        else:
            counter += 1

        # Update progress bar and status text
        percentage = int((epoch + 1) / epochs * 100)
        progress_bar.progress(percentage)
        status_text.text(
            f"Training... Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}"
        )

        if counter >= patience:
            st.warning("⏹️ Early stopping triggered!")
            break

    status_text.text("✅ Training complete.")

    # Provide download link for the model
    with open("saved_model/model.pth", "rb") as f:
        model_data = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_data,
        file_name="model.pth",
        mime="application/octet-stream",
    )

    # Set model trained flag
    st.session_state.model_trained = True
    # No need for rerun, state is updated
