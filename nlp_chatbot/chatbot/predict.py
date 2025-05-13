import torch
import pickle
import json
import numpy as np
import spacy
import en_core_web_sm
from pathlib import Path
from model.model_arch import ChatbotModel

nlp = en_core_web_sm.load()
BASE_DIR = Path(__file__).resolve().parent.parent

model = None
words = []
classes = []
intents = {}


def load_model():
    global model, words, classes, intents

    try:
        words_path = BASE_DIR / "saved_model" / "words.pkl"
        classes_path = BASE_DIR / "saved_model" / "classes.pkl"
        model_path = BASE_DIR / "saved_model" / "model.pth"
        intents_path = BASE_DIR / "data" / "intents.json"

        for path in [words_path, classes_path, model_path, intents_path]:
            if not path.exists():
                raise FileNotFoundError(f"❌ Missing file: {path}")

        with open(words_path, "rb") as f:
            words = pickle.load(f)
        with open(classes_path, "rb") as f:
            classes = pickle.load(f)
        with open(intents_path, "r") as f:
            intents = json.load(f)

        model = ChatbotModel(len(words), 512, len(classes))
        model.load_state_dict(torch.load(model_path))
        model.eval()

    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        raise


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
