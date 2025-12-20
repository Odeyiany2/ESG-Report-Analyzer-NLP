import streamlit as st
from src.agentic_ai.orchestrator import ESGstate, build_esg_agent_graph
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler
from src.nlp.retriever import Retriever
from src.utils.logging import streamlit_app_logger


st.title("ESG Report Analyzer")

st.markdown("""
This application allows you to analyze ESG reports using an AI-powered agent.""")


  