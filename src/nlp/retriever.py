import os 
import yaml
from typing import List, Dict
from dotenv import load_dotenv
from textwrap import dedent
from openai import OpenAI
from transformers import BertTokenizer, pipeline, BertForSequenceClassification
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import retriever_logger

load_dotenv() #load environment variables from .env file

#set up the llm client 
gpt_client = OpenAI(
    base_url= "https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
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
    
    #load the prompts from a yaml file
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
    #build the prompt by filling in the template with the query and context
    def build_prompt(self, query: str, context: Dict[str, List[str]], prompt_data:dict) -> str:
        """
        Build the prompt by filling in the template with the query and context.
        """
        # merge all context sections into a single string
        try:
            prompt = dedent(f"""
                            {prompt_data.get("persona", "")}
                            Context Sections:
                            {prompt_data.get("context_instructions", "")}

                            Standards Context:
                            {" \n".join(context.get("standards", []))}

                            Reports Context:
                            {" \n".join(context.get("reports", []))}

                            Task Instructions:
                            {prompt_data.get("task_instructions", "")}
                            
                            Response Format:
                            {prompt_data.get("response_format", "")}
                          
                            Stule Guidelines:
                            {prompt_data.get("style_guidelines", "")}

                            Examples:
                            {prompt_data.get("examples", "")}

                            User Query:
                            {query}
                            """)
            retriever_logger.info("Prompt built successfully.")
            return prompt
        except Exception as e:
            retriever_logger.error(f"Error building prompt: {e}")
            print(f"Error building prompt: {e}")
            return ""
    
    #retrieve relevant context from the vector stores based on the user query
    def retrieve_context(self, query:str, k:int = 3) -> Dict[str, List[str]]:
        """
        Retrieve relevant context from the vector stores based on the user query."""

        if not self.standards_vectorstore or not self.reports_vectorstore:
            retriever_logger.error("Vector stores not loaded. Please load them before retrieving context.")
            print("Vector stores not loaded. Please load them before retrieving context.")
            raise ValueError("Vector stores not loaded. Please load them before retrieving context.")
        
        retriever_logger.info("Retrieving relevant ESG sections from vector stores...")
        standards_docs = self.standards_vectorstore.similarity_search(query, k=k)
        reports_docs = self.reports_vectorstore.similarity_search(query, k=k)
        return {
            "standards": [doc.page_content for doc in standards_docs],
            "reports": [doc.page_content for doc in reports_docs]
        }
    
    #compare report content against standards using the fully constructed prompt and the llm 
    def compare_content(self, query: str, context: dict, prompt_path: str) -> str:
        """
        Compare report content against standards using the fully constructed prompt and the llm. 
        """
        try:
            prompts = self.load_prompts(prompt_path)
            full_prompt = self.build_prompt(query, context, prompts)
            retriever_logger.info("Sending prompt to LLM for ESG analysis...")

            response = self.llm.chat.completions.create(
                model = "openai/gpt-oss-120b:fireworks-ai",
                messages= [
                    {"role": "user", "content": full_prompt},
                    {"role": "system", "content": "You are a helpful assistant that helps users analyze ESG reports based on relevant standards."}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            retriever_logger.info("Received response from LLM.")
            return response.choices[0].message.content
        except Exception as e:
            retriever_logger.error(f"Error during content comparison: {e}")
            print(f"Error during content comparison: {e}")
            return "An error occurred when trying to analyze the ESG report. Please try again."
        