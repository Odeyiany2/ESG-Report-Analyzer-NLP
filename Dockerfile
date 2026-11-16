FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# download the embedding model at build time so the first request isn't slow
RUN python -c "from src.nlp.embeddings import EmbeddingHandler; EmbeddingHandler(model_name='sentence-transformers/all-MiniLM-L6-v2')"
RUN mkdir -p esg_vectorstores/uploaded_reports

EXPOSE 7860
CMD ["uvicorn", "src.backend.fastapi_logic:app", "--host", "0.0.0.0", "--port", "7860"]