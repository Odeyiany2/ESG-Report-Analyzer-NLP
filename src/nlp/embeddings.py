from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def embed_standards(standards: List[str], persist_directory: str) -> Chroma:
    pass