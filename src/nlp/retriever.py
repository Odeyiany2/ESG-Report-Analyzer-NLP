import os 
from dotenv import load_dotenv
from openai import OpenAI
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import retriever_logger
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate


#set up the llm client 
gpt_client = OpenAI(
    base_url= "https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model = "openai/gpt-oss-120b:fireworks-ai"
)
class Retriever:
    def __init__(self, embedding_handler: EmbeddingHandler, llm_client = gpt_client):
        self.embedding_handler = embedding_handler
        self.llm_client = llm_client
    
    def create_retrieval_chain():
        pass