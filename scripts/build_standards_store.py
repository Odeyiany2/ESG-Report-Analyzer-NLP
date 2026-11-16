"""Build the standards vector store once, locally, and commit it.

Run from the repo root with the SAME embedding model the server uses:
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 python scripts/build_standards_store.py
"""
import os
import shutil
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler

DOCS = os.getenv("STANDARD_DOCS_DIR", "data/raw/standards")
OUT = os.getenv("STANDARD_VECTORSTORE_PATH", "esg_vectorstores/standards")
MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)
docs = DocumentHandler().load_standard_documents(DOCS)
print(f"Loaded {len(docs)} pages from {DOCS}")
EmbeddingHandler(model_name=MODEL).embed_documents(docs, persist_directory=OUT)
print(f"Standards store written to {OUT} using {MODEL}")
