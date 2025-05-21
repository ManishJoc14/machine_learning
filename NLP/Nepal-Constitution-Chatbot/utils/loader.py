import fitz  # PyMuPDF library for PDF processing
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Constitution-of-Nepal.pdf"

def extract_text_from_pdf() -> str:
    # Open the PDF file
    doc = fitz.open(PDF_PATH)

    text = ""
    for page in doc:  # Iterate through each page in the PDF
        text += page.get_text()  # Extract text from the current page and append

    # Return the combined text
    return text
