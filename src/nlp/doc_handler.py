import os
import tempfile
from pathlib import Path 
from typing import List
from src.utils.logging import doc_handler_logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# --- Class to handle document loading and processing ---
class DocumentHandler:
    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt"]
    
    def load_standard_documents(self, standard_docs_dir: str) -> List[Document]:
        """
        Loads ESG standard documents (SASB, GRI, IFRS) from a directory.
        """
        if not os.path.exists(standard_docs_dir):
            doc_handler_logger.error(f"Directory does not exist: {standard_docs_dir}")
            return []

        all_standard_docs = []
        for root, _, files in os.walk(standard_docs_dir):
            subject = Path(root)
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                try:
                    if file_ext == ".pdf":
                        loader = PyPDFLoader(file_path=file_path)
                    elif file_ext == ".docx":
                        loader = Docx2txtLoader(file_path=file_path)
                    elif file_ext == ".txt":
                        loader = TextLoader(file_path=file_path, encoding="utf-8")
                    else:
                        doc_handler_logger.warning(f"Unsupported file type: {file_ext}")
                        continue

                    docs = loader.load()
                    for d in docs:
                        d.metadata["subject"] = subject.name
                        all_standard_docs.append(d)

                except Exception as e:
                    doc_handler_logger.error(f"Error loading {file_path}: {e}")
                    continue

        return all_standard_docs

    def load_uploaded_documents(self, uploaded_files) -> List[Document]:
        """
        Loads ESG reports uploaded by the user and converts them into Langchain Document object. 
        """
        # initialize an empty list to store the loaded documents
        ESG_docs = []

        doc_handler_logger.info(f"Starting to process {len(uploaded_files)} uploaded files")
        # create a temporary directory to store the uploaded files 
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                doc_handler_logger.info(f"Processing file: {file.filename}")
                # save the uploaded file to the temporary directory 
                file_path = Path(temp_dir) / file.filename
                try:
                    with open(file_path, "wb") as f:
                        content = file.file.read()
                        f.write(content)
                        doc_handler_logger.info(f"Saved {len(content)} bytes for file: {file.filename}")
                except Exception as e:
                    doc_handler_logger.error(f"Error saving file {file.filename}: {e}")
                    continue
                
                docs = []
                # determine the file type and use the appropriate loader
                suffix = os.path.splitext(file.filename)[-1].lower()
                doc_handler_logger.info(f"Detected file extension: {suffix} for {file.filename}")
                try:
                    if suffix == ".pdf":
                        loader = PyPDFLoader(file_path=str(file_path))
                        docs = loader.load()
                    
                    elif suffix == ".docx":
                        loader = Docx2txtLoader(file_path=str(file_path))
                        docs = loader.load()
                    
                    elif suffix == ".txt":
                        loader = TextLoader(file_path=str(file_path), encoding="utf-8")
                        docs = loader.load()
                    else:
                        doc_handler_logger.warning(f"Unsupported file type: {suffix}. Skipping file: {file.filename}")
                        print(f"Unsupported file type: {suffix}. Skipping file: {file.filename}")
                        continue

                    doc_handler_logger.info(f"Loaded {len(docs)} documents from {file.filename}")
                    # add metadata to each document and append to the list
                    for d in docs:
                        d.metadata["source"] = file.filename
                        d.metadata["subject"] = "user_upload"
                        ESG_docs.append(d)

                except Exception as e:
                    doc_handler_logger.error(f"Error loading file {file.filename}, Error: {e}")
                    print(f"Error loading file {file.filename}, Error: {e}")
                    continue

                finally:
                    # remove the temporary file
                    if file_path.exists():
                        os.remove(file_path)
        
        doc_handler_logger.info(f"Successfully loaded {len(ESG_docs)} total documents from all uploaded files")
        return ESG_docs

