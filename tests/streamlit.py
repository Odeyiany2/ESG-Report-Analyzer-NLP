import streamlit as st
from src.agentic_ai.orchestrator import ESGstate, build_esg_agent_graph
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler
from src.nlp.retriever import Retriever
from src.utils.logging import streamlit_app_logger


st.title("ESG Report Analyzer NLP")

st.markdown("""
This application allows you to analyze ESG reports using an AI-powered agent.""")

@st.cache_resource
def initialize_esg_agent():
    embedding_handler = EmbeddingHandler()
    document_handler = DocumentHandler(embedding_handler=embedding_handler)
    esg_state = ESGstate(document_handler=document_handler)
    esg_agent = build_esg_agent_graph(esg_state)
    return esg_agent

esg_agent = initialize_esg_agent()

uploaded_files = st.file_uploader("Upload ESG Reports", accept_multiple_files=True, type=["pdf", "docx", "txt"])
if uploaded_files:
    try:
        docs = esg_agent.document_handler.load_uploaded_documents(uploaded_files)
        if not docs:
            streamlit_app_logger.error("No valid documents were uploaded.")
            st.error("No valid documents were uploaded.")
        else:
            streamlit_app_logger.info(f"Successfully processed {len(docs)} documents from upload.")
            st.success(f"Successfully uploaded and processed {len(docs)} documents.")
    except Exception as e:
        streamlit_app_logger.error(f"Error processing uploaded documents: {e}")
        st.error("Error processing uploaded documents.")

query = st.text_input("Enter your query about the uploaded ESG reports:")
if query:
    try:
        #using the agent to process and retrieve structured information from the report based on the query
        state = esg_agent.run(initial_state=esg_agent.state_class(query=query))
    

        

    except Exception as e:
        streamlit_app_logger.error(f"Error processing query: {e}")
        st.error("Error processing query.")