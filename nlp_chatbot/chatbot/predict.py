import torch
import pickle
import json
import numpy as np
import spacy
from model.model_arch import ChatbotModel

nlp = spacy.load("en_core_web_sm")

model = None
words = []
classes = []
intents = {}


def load_model():
    global model, words, classes, intents

    with open("saved_model/words.pkl", "rb") as f:
        words = pickle.load(f)
    with open("saved_model/classes.pkl", "rb") as f:
        classes = pickle.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)

    model = ChatbotModel(len(words), 512, len(classes))
    model.load_state_dict(torch.load("saved_model/model.pth"))
    model.eval()


def get_response(text):
    if model is None:
        load_model()

    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc]
    bag = [1 if word in tokens else 0 for word in words]
    X = torch.tensor([bag], dtype=torch.float32)
    with torch.no_grad():
        output = model(X)
    tag = classes[torch.argmax(output, dim=1).item()]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "Sorry, I didn't understand that."
