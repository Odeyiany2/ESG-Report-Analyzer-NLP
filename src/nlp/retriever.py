import os 
import yaml
from typing import List, Dict
from dotenv import load_dotenv
from textwrap import dedent
from openai import OpenAI
from langchain_core.documents import Document
from transformers import BertTokenizer, pipeline, BertForSequenceClassification
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import retriever_logger

load_dotenv() #load environment variables from .env file

#set up the llm client 
gpt_client = OpenAI(
    base_url= "https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)

#set up the tokenizer and model for esg classification
finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg", num_labels=4)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-esg")
nlp_classifier = pipeline("text-classification", model = finbert, tokenizer = tokenizer)

#retrieval and comparison class
class Retriever:
    def __init__(self, embedding_handler: EmbeddingHandler, llm_client = gpt_client):
        self.embed = embedding_handler
        self.llm = llm_client
        self.standards_vectorstore = None
        self.reports_vectorstore = None
    
    #load the vector store for standards and the user uploaded reports
    def create_vector_store(self, standards_vector_path:str, 
                            standard_docs: List[Document],
                            uploaded_report: List[Document], reports_vector_path:str):
        retriever_logger.info("Creating vector stores for standards and reports uploaded...")
        try:
            #check if the vector stores already exist, if not create them
            if not os.path.exists(standards_vector_path) or not os.path.exists(reports_vector_path):
                retriever_logger.info("Vector stores not found, creating new ones...")
                os.makedirs(standards_vector_path, exist_ok=True)
                os.makedirs(reports_vector_path, exist_ok=True)
                self.reports_vectorstore = self.embed.embed_documents(uploaded_report, persist_directory=reports_vector_path)
                self.standards_vectorstore = self.embed.embed_documents(standard_docs, persist_directory=standards_vector_path)
            else:
                retriever_logger.info("Loading existing vector stores...")
                self.reports_vectorstore = self.embed.load_vectorstore(persist_directory=reports_vector_path)
                self.standards_vectorstore = self.embed.load_vectorstore(persist_directory=standards_vector_path)
            retriever_logger.info("Vector stores are ready for use.")
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

                            Reports Context (with ESG classifications):
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
    
    #classify the ESG aspects of the report content using the finbert model
    def classify_esg(self, text:str)-> List[Dict]:
        """
        Classify the ESG aspects of the report content using the finbert model.
        """
        try:
            results = nlp_classifier(text)
            retriever_logger.info("ESG classification completed.")
            return results
        except Exception as e:
            retriever_logger.error(f"Error during ESG classification: {e}")
            print(f"Error during ESG classification: {e}")
            return [{"label": "ERROR", "score": 0.0}]
    
    #explainability layer
    def explain_classification(self, text:str, classification:Dict) -> str:
        """Provide explanations for the ESG classification results by showing the most influential tokens."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = finbert(**inputs)

    
    #enrich context with ESG classifications
    def enrich_context_with_esg(self, report_sections: List[str]) -> List[Dict]:
        """
        Enrich context with ESG classifications. 
        """
        enriched_sections = []
        for text in report_sections:
            classification = self.classify_esg(text)
            label = classification[0]["label"] if classification else "UNKNOWN"
            score = round(classification[0]["score"], 3) if classification else 0.0
            enriched_sections.append({
                "text": text,
                "esg_label": label,
                "confidence": score
            })
        return enriched_sections
    
    #compare report content against standards using the fully constructed prompt and the llm 
    def compare_content(self, query: str, prompt_path: str) -> str:
        """
        Compare report content against standards using the fully constructed prompt and the llm. 
        """
        try:
            contexts = self.retrieve_context(query)
            enriched_reports = self.enrich_context_with_esg(contexts["reports"])
            contexts["reports"] = enriched_reports

            prompts = self.load_prompts(prompt_path)
            full_prompt = self.build_prompt(query, contexts, prompts)
            retriever_logger.info("Sending enriched prompt to LLM for ESG analysis...")

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
        