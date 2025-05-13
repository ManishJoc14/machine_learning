import json
import pickle
import numpy as np
import torch
import spacy
import random
from pathlib import Path

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

    words = []
    classes = []
    documents = []
    ignore = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            doc = nlp(pattern)
            tokens = [t.lemma_.lower() for t in doc if t.text not in ignore]
            words.extend(tokens)
            documents.append((tokens, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # Sort and deduplicate
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Create directory if not exists using absolute path
    save_path = Path(__file__).resolve().parent / "saved_model"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save words and classes
    with open(save_path / "words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open(save_path / "classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    training = []
    output_empty = [0] * len(classes)

    for tokens, tag in documents:
        bag = [1 if w in tokens else 0 for w in words]
        output_row = output_empty[:]
        output_row[classes.index(tag)] = 1
        training.append(bag + output_row)

    random.shuffle(training)
    training = np.array(training)

    X = torch.tensor(training[:, : len(words)], dtype=torch.float32)
    Y = torch.tensor(training[:, len(words) :], dtype=torch.float32)
    Y = torch.argmax(Y, axis=1)

    return X, Y, len(words), len(classes)