import os 
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from textwrap import dedent
import requests
from openai import OpenAI
from langchain_core.documents import Document
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import retriever_logger

load_dotenv() #load environment variables from .env file

#set up the llm client 
gpt_client = OpenAI(
    base_url= "https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)

TYPESAFE_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_JEV_MODEL = "jev-latest"

#retrieval and comparison class
class Retriever:
    def __init__(self, embedding_handler: EmbeddingHandler, llm_client = gpt_client):
        self.embed = embedding_handler
        self.llm = llm_client
        self.standards_vectorstore = None
        self.reports_vectorstore = None
    
    # vector store management - create or load vector stores for standards and reports 
    def create_vector_store(self, standards_vector_path:str, 
                            standard_docs: List[Document],
                            uploaded_report: List[Document], reports_vector_path:str):
        """
        Creates or loads vector stores for ESG standards and the uploaded report.
        Each store is handled independently so a new user upload always gets
        a fresh reports store even when the standards store already exists.
        """
        retriever_logger.info("setting up vector stores...")
        try:
            # Convert paths to Path objects for safe handling
            standards_path = Path(standards_vector_path)
            reports_path = Path(reports_vector_path)
            
            #standards vector store 
            if not standards_path.exists() or not list(standards_path.iterdir()):
                retriever_logger.info(f"Standard vector store not found at {standards_vector_path}, creating...")
                standards_path.mkdir(parents=True, exist_ok=True)
                self.standards_vectorstore = self.embed.embed_documents(standard_docs, 
                                                                        persist_directory=str(standards_path))
            else:
                retriever_logger.info(f"Loading existing standard vector stores from {standards_vector_path}...")
                self.standards_vectorstore = self.embed.load_vectorstore(
                    persist_directory=str(standards_path))
            retriever_logger.info("Vector stores are ready for use.")
            
            #reports vector store - always create a new one for each user upload
            retriever_logger.info(f"creating reports vector store for this session at {reports_vector_path}...")
            reports_path.mkdir(parents=True, exist_ok=True)
            self.reports_vectorstore = self.embed.embed_documents(uploaded_report, 
                                                                  persist_directory=str(reports_path))
            retriever_logger.info("Reports vector store created successfully.")
        except Exception as e:
            retriever_logger.error(f"Error loading vector stores: {e}")
            raise
    
    #prompt management 
    def load_prompts(self, prompts_file: str) -> Dict[str, str]:
        """
        Load prompts from a YAML file.
        """
        try:
            with open(prompts_file, 'r', encoding='utf-8') as file:
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
            standards_context = " \n".join(context.get("standards", []))
            reports_context = " \n".join(context.get("reports", []))

            prompt = dedent(f"""
                            {prompt_data.get("persona", "")}

                            {prompt_data.get("prior_analysis", "")}
                            Context Sections:
                            {prompt_data.get("context_instructions", "")}

                            Standards Context:
                            {standards_context}

                            Reports Context (with ESG classifications):
                            {reports_context}

                            Task Instructions:
                            {prompt_data.get("task_instructions", "")}
                            
                            Response Format:
                            {prompt_data.get("response_format", "")}
                          
                            Style Guidelines:
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
    
    #classify the ESG aspects of the report content using TypeSafe Jev
    def classify_esg(self, text:str)-> List[Dict]:
        """
        Classifies a text chunk into one of Jev's four ESG categories:
        Environmental, Social, Governance, or None.
        """
        try:
            api_key = os.getenv("TYPESAFE_API_KEY")
            if not api_key:
                raise ValueError("TYPESAFE_API_KEY is not set.")

            payload = {
                "state": text,
                "model": TYPESAFE_JEV_MODEL,
                "questions": {
                    "esg_category": {
                        "type": "choice",
                        "instructions": "Which ESG category does this text chunk primarily belong to?",
                        "criteria": {
                            "Environmental": "Environmental topics such as emissions, energy, climate, water, waste, biodiversity, pollution, or environmental management.",
                            "Social": "Social topics such as labor, human rights, health and safety, community impact, diversity, customers, or supply-chain social issues.",
                            "Governance": "Governance topics such as board oversight, ethics, compliance, risk management, transparency, anti-corruption, controls, or executive accountability.",
                            "None": "The text does not contain meaningful ESG disclosure or cannot be assigned to Environmental, Social, or Governance."
                        }
                    }
                }
            }

            response = requests.post(
                TYPESAFE_JEV_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answers", {}).get("esg_category", {})
            label = answer.get("choice", "None")
            probabilities = answer.get("probabilities", {})
            score = answer.get("confidence", probabilities.get(label, 0.0))

            retriever_logger.info("ESG classification completed via TypeSafe Jev.")
            return [{
                "label": label,
                "score": float(score or 0.0),
                "probabilities": probabilities
            }]
        except Exception as e:
            retriever_logger.error(f"Error during Jev ESG classification: {e}")
            print(f"Error during Jev ESG classification: {e}")
            return [{"label": "ERROR", "score": 0.0}]
    
    #explainability layer
    def explain_classification(self, text: str, classification: Dict) -> str:
        return (
            f"Classification '{classification['label']}' "
            f"with calibrated confidence {classification['score']:.3f} "
            f"via TypeSafe Jev"
        )
    
    #enrich context with ESG classifications
    def enrich_context_with_esg(self, report_sections: List[str]) -> Optional[List[Dict]]:
        """
        Runs each retrieved report chunk through TypeSafe Jev classification.
 
        Returns a list of dicts with keys: text, esg_label, confidence, important_tokens.
        The caller (orchestrator summarize_node) is responsible for serializing
        these dicts to strings before passing them to build_prompt().
        """
        try:
            retriever_logger.info("Enriching report sections with ESG classifications...")
            enriched = []
            for text in report_sections:
                classification = self.classify_esg(text)
                if not classification or classification[0]["label"] == "ERROR":
                    enriched.append({
                        "text": text,
                        "esg_label": "UNKNOWN",
                        "confidence": 0.0,
                        "important_tokens": "Classification failed."
                    })
                    continue
 
                label = classification[0]["label"]
                score = round(classification[0]["score"], 3)
                explanation = self.explain_classification(text, classification[0])
                enriched.append({
                    "text": text,
                    "esg_label": label,
                    "confidence": score,
                    "important_tokens": explanation
                })
            retriever_logger.info(f"Enriched {len(enriched)} report sections.")
            return enriched
        except Exception as e:
            retriever_logger.error(f"Error during context enrichment: {e}")
            return []

    #LLM call to run the analysis based on the constructed prompt
    def run_analysis(self, prompt:str) -> str:
        """
        Run the ESG analysis by sending the constructed prompt to the LLM.
        """
        try:
            retriever_logger.info("Sending prompt to LLM for ESG analysis...")
            response = self.llm.chat.completions.create(
                model = "openai/gpt-oss-120b:fireworks-ai",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that helps users analyze ESG reports based on relevant standards."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=0,
                max_tokens=2000,
            )
            retriever_logger.info("Received response from LLM.")
            return response.choices[0].message.content
        except Exception as e:
            retriever_logger.error(f"Error during ESG analysis: {e}")
            return "An error occurred when trying to analyze the ESG report. Please try again."
        
    #compare report content against standards using the fully constructed prompt and the llm 
    #orchestration shortcut (used by FastAPI directly, bypasses LangGraph)
    def compare_content(self, query: str, prompt_file: str, prior_analysis: str = "") -> str:
        """
        Compare report content against standards using the fully constructed prompt and the llm. 
        """
        try:
            #Retrieve relevant chunks
            contexts = self.retrieve_context(query)
 
            #Classify and enrich report sections
            enriched_reports = self.enrich_context_with_esg(contexts["reports"])
 
            #Serialize enriched dicts → strings for the prompt
            contexts["reports"] = [
                f"[{s['esg_label']} | confidence: {s['confidence']}]\n"
                f"{s['text']}\n"
                f"Key tokens: {s.get('important_tokens', 'N/A')}"
                for s in enriched_reports
            ]
 
            #Load prompts and build the full prompt string
            prompt_data = self.load_prompts(prompt_file)
            if prior_analysis:
                prompt_data["prior_analysis"] = (
            f"Prior analysis of this report (do not contradict it unless asked):\n{prior_analysis}\n"
        )
            full_prompt = self.build_prompt(query, contexts, prompt_data)
 
            #Send to LLM
            retriever_logger.info("Running full compare_content pipeline...")
            return self.run_analysis(full_prompt)
 
        except Exception as e:
            retriever_logger.error(f"Error during content comparison: {e}")
            return "An error occurred when trying to analyze the ESG report. Please try again."