from fastapi import FastAPI
from typing import List, Dict
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler
from src.nlp.retriever import Retriever
from src.utils.logging import api_logger
from langchain_core.documents import Document

app = FastAPI()
