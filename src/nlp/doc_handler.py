import os
import tempfile
from pathlib import Path 
from typing import List
from src.utils.logging import doc_handler_logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_standard_documents(standard_docs_dir: str) -> List[Document]:
    """
    Loads standard ESG reports from a specified directory and converts them into Langchain Document objects.

    Args:
        standard_docs_dir (str): The directory containing the standard ESG reports.
    Returns:
        List[Document]: A list of Langchain Document objects representing the loaded ESG reports.
    """
    #check if the directory exists
    if not os.path.exists(standard_docs_dir):
        doc_handler_logger.error(f"Directory does not exist: {standard_docs_dir}")
        return []
    all_standard_docs = []

    #iterate through all files in the directory
    for root, _, files in os.walk(standard_docs_dir):
        subject = Path(root)
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            try:
                if file_ext == ".pdf":
                    print(f"[PDF] Processing PDF: {file}")
                    doc_handler_logger.info(f"[PDF] Processing PDF: {file}")
                    loader = PyPDFLoader(file_path=file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["subject"] = subject.name
                        all_standard_docs.append(d)
                elif file_ext == ".docx":
                    print(f"[DOCX] Processing DOCX: {file}")
                    doc_handler_logger.info(f"[DOCX] Processing DOCX: {file}")
                    loader = Docx2txtLoader(file_path=file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["subject"] = subject.name
                        all_standard_docs.append(d)
                elif file_ext == ".txt":
                    print(f"[TXT] Processing file in text format:{file}")
                    doc_handler_logger.info(f"[TXT] Processing file in text format:{file}")
                    loader = TextLoader(file_path=file_path, encoding = "utf-8")
                    docs = loader.load()
                    for d in docs:
                        d.metadata["subject"] = subject.name
                        all_standard_docs.append(d)
                else:
                    doc_handler_logger.warning(f"Unsupported file type: {file_ext}. Skipping file: {file}")
                    print(f"Unsupported file type: {file_ext}. Skipping file: {file}")
                    continue
            except Exception as e:
                doc_handler_logger.error(f"Error loading file {file_path}, Error: {e}")
                print(f"Error loading file {file_path}, Error: {e}")
                continue

def load_uploaded_documents(uploaded_files) -> List[Document]:
    """
    Loads ESG reports uploaded by the user and converts them into Langchain Document object. 
    """
    # initialize an empty list to store the loaded documents
    ESG_docs = []

    # create a temporary directory to store the uploaded files 
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            # save the uploaded file to the temporary directory 
            file_path = Path(temp_dir) / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            docs = []
            # determine the file type and use the appropriate loader
            suffix = os.path.splitext(file.name)[-1].lower()
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
                    doc_handler_logger.warning(f"Unsupported file type: {suffix}. Skipping file: {file.name}")
                    print(f"Unsupported file type: {suffix}. Skipping file: {file.name}")
                    continue

                # add metadata to each document and append to the list
                for d in docs:
                    d.metadata["source"] = file.name
                    d.metadata["subject"] = "user_upload"
                    ESG_docs.append(d)

            except Exception as e:
                doc_handler_logger.error(f"Error loading file {file.name}, Error: {e}")
                print(f"Error loading file {file.name}, Error: {e}")
                continue

            finally:
                # remove the temporary file
                os.remove(file_path)
    return ESG_docs

