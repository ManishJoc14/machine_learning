import os

def save_uploaded_file(uploaded_file, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
