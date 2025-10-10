import os 
from dotenv import load_dotenv
from openai import OpenAI
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import retriever_logger

load_dotenv() #load environment variables from .env file

#set up the llm client 
gpt_client = OpenAI(
    base_url= "https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model = "openai/gpt-oss-120b:fireworks-ai"
)
#retrieval and comparison class
class Retriever:
    def __init__(self, embedding_handler: EmbeddingHandler, llm_client = gpt_client):
        self.embed = embedding_handler
        self.llm = llm_client

    