import streamlit as st
from src.agentic_ai.orchestrator import ESGstate, build_esg_agent_graph
from src.nlp.doc_handler import DocumentHandler
from src.nlp.retriever import Retriever
from src.nlp.embeddings import EmbeddingHandler
from src.utils.logging import streamlit_app_logger


st.title("ESG Report Analyzer NLP")

st.markdown("""
This interface is for testing the ESG Report Analyzer NLP functionalities
""")

state = ESGstate(
    query="",
    messages=[]
)

embed = EmbeddingHandler()

retriever = Retriever(embedding_handler=embed)


#load the standards document 
standard_path = r"C:\Projects_ML\ESG-Report-Analyzer\ESG-Report-Analyzer-NLP\data\raw\standards"

standard_documents = DocumentHandler().load_standard_documents(standard_docs_dir=standard_path)

## Input section 
files = st.file_uploader("Upload ESG Reports", accept_multiple_files=True)

if files:
    docs = DocumentHandler().load_uploaded_documents(files)
    st.success(f"Loaded {len(docs)} documents from upload.")
 
    retriever.create_vector_store(
        standards_vector_path=r"C:\Projects_ML\ESG-Report-Analyzer\ESG-Report-Analyzer-NLP\vectorstores\standards",
        standard_docs=standard_documents,
        uploaded_report=docs,
        reports_vector_path=r"C:\Projects_ML\ESG-Report-Analyzer\ESG-Report-Analyzer-NLP\vectorstores\test_uploaded_reports"
    )

    state.reports_ready = True


if state.reports_ready:
    query = st.text_input("Enter your ESG query:", value="What are the environmental policies of the company?")
    if query:
        state.query = query
        graph = build_esg_agent_graph()
    
    if st.button("Analyze"):
        with st.spinner("Analyzing ESG reports..."):
            result = graph.invoke(state)
        

    st.subheader("### ESG Analysis Result")
    st.write(state.response)