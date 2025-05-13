import os

def save_uploaded_file(uploaded_file, path):
    # Create the directory for the given path if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Open the file in binary write mode and save the uploaded file's content
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
