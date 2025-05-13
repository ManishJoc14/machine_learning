import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.preprocess import preprocess_data
from model.model_arch import ChatbotModel
import streamlit as st


def train_model(json_path, epochs):
    # Preprocess the data and retrieve input/output sizes
    X, Y, input_size, output_size = preprocess_data(json_path)

    # Initialize the chatbot model with input size, hidden layer size, and output size
    model = ChatbotModel(input_size, 512, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Initialize variables for early stopping
    best_loss = float("inf")
    patience, counter = 500, 0

    # Streamlit progress bar and status text for UI updates
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)  # Compute loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the model if the loss improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            os.makedirs("saved_model", exist_ok=True)  # Ensure directory exists
            torch.save(model.state_dict(), "saved_model/model.pth")  # Save model state
        else:
            counter += 1  # Increment counter if no improvement

        # Update progress bar and status text in Streamlit
        percentage = int((epoch + 1) / epochs * 100)
        progress_bar.progress(percentage)
        status_text.text(
            f"Training... Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}"
        )

        # Trigger early stopping if patience is exceeded
        if counter >= patience:
            st.warning("⏹️ Early stopping triggered!")
            break

    # Update status text when training is complete
    status_text.text("✅ Training complete.")

    # Set a flag in Streamlit session state to indicate the model is trained
    st.session_state.model_trained = True
