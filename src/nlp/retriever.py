import os 
import yaml
from typing import List, Dict
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
        self.standards_vectorstore = None
        self.reports_vectorstore = None
    
    #load the vector store for standards and the user uploaded reports
    def load_vector_store(self, standards_path:str, reports_path:str):
        retriever_logger.info("Loading vector stores for standards and reports...")
        try:
            self.standards_vectorstore = self.embed.load_vectorstore(standards_path)
            self.reports_vectorstore = self.embed.load_vectorstore(reports_path)
            retriever_logger.info("Vector stores loaded successfully.")
        except Exception as e:
            retriever_logger.error(f"Error loading vector stores: {e}")
            print(f"Error loading vector stores: {e}")
    
    def load_prompts(self, prompts_file: str) -> Dict[str, str]:
        """
        Load prompts from a YAML file.
        """
        try:
            with open(prompts_file, 'r') as file:
                prompts = yaml.safe_load(file)
            retriever_logger.info(f"Prompts loaded from {prompts_file}")
            return prompts
        except Exception as e:
            retriever_logger.error(f"Error loading prompts from {prompts_file}: {e}")
            print(f"Error loading prompts from {prompts_file}: {e}")
            return {}
    


        