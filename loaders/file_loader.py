import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join("data", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    print("🔹 Loading document...")
    if suffix == ".txt":
        loader = TextLoader(temp_path, encoding='utf-8')
    elif suffix == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif suffix in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(temp_path)
    else:
        return None

    return loader.load()
