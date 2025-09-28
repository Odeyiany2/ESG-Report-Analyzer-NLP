import os
from pathlib import Path 
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_uploaded_documents(upload_dir: str) -> List[Document]:
    pass