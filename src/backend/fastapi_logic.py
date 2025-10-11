from fastapi import FastAPI
from typing import List, Dict
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler
from src.nlp.retriever import Retriever
from src.utils.logging import api_logger
from langchain_core.documents import Document

app = FastAPI()


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the API is running smoothly.
    """
    return {"status": "ok", "message": "API is running smoothly."}

@app.post("/upload")
async def upload_documents():
    pass

@app.post("/query")
async def query_assistant():
    pass

@app.post("/regression_analysis")
async def regression_analysis():
    pass

@app.post("/forecast_esg_trends")
async def forecast_esg_trends():
    pass