import os
from pypdf import PdfReader
from docx import Document


def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_docx(file_path: str) ->str:
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def load_document(file_path: str) -> str:
    """
    Automatically detect the file type
    and extract text.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

        