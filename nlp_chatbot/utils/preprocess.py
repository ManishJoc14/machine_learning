import json
import pickle
import numpy as np
import torch
import spacy
import random

# Load the small English language model from spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_data(json_path):
    """
    Preprocesses the data from a JSON file containing intents and patterns.

    Args:
        json_path (str): Path to the JSON file containing intents.

    Returns:
        tuple: Processed input features (X), target labels (Y), 
               number of unique words, and number of unique classes.
    """
    # Load intents from the JSON file
    with open(json_path, "r") as file:
        intents = json.load(file)

    words = []  # List to store all unique words (lemmas)
    classes = []  # List to store all unique intent tags
    documents = []  # List to store tokenized patterns and their corresponding tags
    ignore = ["?", "!", ".", ","]  # Characters to ignore during tokenization

    # Iterate through each intent in the JSON file
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize and lemmatize the pattern
            doc = nlp(pattern)
            tokens = [t.lemma_.lower() for t in doc if t.text not in ignore]
            words.extend(tokens)  # Add tokens to the words list
            documents.append((tokens, intent["tag"]))  # Store tokens and tag as a tuple
            if intent["tag"] not in classes:
                classes.append(intent["tag"])  # Add unique tags to the classes list

    # Sort and remove duplicates from words and classes
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save the words and classes to pickle files for later use
    with open("saved_model/words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("saved_model/classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    training = []  # List to store training data
    output_empty = [0] * len(classes)  # Template for one-hot encoding of classes

    # Create the training data
    for tokens, tag in documents:
        # Create a bag-of-words representation for the tokens
        bag = [1 if w in tokens else 0 for w in words]
        # Create a one-hot encoded output row for the tag
        output_row = output_empty[:]
        output_row[classes.index(tag)] = 1
        # Combine the bag-of-words and one-hot encoded output
        training.append(bag + output_row)

    # Shuffle the training data to ensure randomness
    random.shuffle(training)
    training = np.array(training)

    # Split the training data into input features (X) and target labels (Y)
    X = torch.tensor(training[:, : len(words)], dtype=torch.float32)  # Features
    Y = torch.tensor(training[:, len(words) :], dtype=torch.float32)  # Labels
    Y = torch.argmax(Y, axis=1)  # Convert one-hot encoded labels to class indices

    return X, Y, len(words), len(classes)
