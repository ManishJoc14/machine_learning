import torch
import pickle
import json
import numpy as np
import spacy
from model.model_arch import ChatbotModel

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize global variables
model = None  # Placeholder for the chatbot model
words = []  # List of words used for training
classes = []  # List of intent classes
intents = {}  # Dictionary to store intents and responses


def load_model():
    """
    Load the trained model, words, classes, and intents from saved files.
    """
    global model, words, classes, intents

    # Load the list of words used for training
    with open("saved_model/words.pkl", "rb") as f:
        words = pickle.load(f)
    
    # Load the list of intent classes
    with open("saved_model/classes.pkl", "rb") as f:
        classes = pickle.load(f)
    
    # Load the intents and responses from a JSON file
    with open("data/intents.json", "r") as f:
        intents = json.load(f)

    # Initialize the chatbot model with the appropriate input and output sizes
    model = ChatbotModel(len(words), 512, len(classes))
    
    # Load the trained model weights
    model.load_state_dict(torch.load("saved_model/model.pth"))
    model.eval()  # Set the model to evaluation mode


def get_response(text):
    """
    Generate a response for the given input text.

    Args:
        text (str): The input text from the user.

    Returns:
        str: The chatbot's response.
    """
    # Load the model and data if not already loaded
    if model is None:
        load_model()

    # Tokenize and lemmatize the input text using spaCy
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc]
    
    # Create a bag-of-words representation for the input text
    bag = [1 if word in tokens else 0 for word in words]
    X = torch.tensor([bag], dtype=torch.float32)
    
    # Predict the intent using the trained model
    with torch.no_grad():
        output = model(X)
    
    # Get the predicted intent tag
    tag = classes[torch.argmax(output, dim=1).item()]
    
    # Find the corresponding response for the predicted intent
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    
    # Default response if no intent matches
    return "Sorry, I didn't understand that."
