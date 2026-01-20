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

@st.cache_resource
def init_retriever() -> Retriever:
   embed = EmbeddingHandler()
   retriever = Retriever(embedding_handler=embed)

retriever = init_retriever()
graph = build_esg_agent_graph()



@st.cache_resource
def load_standard_doc():
    #load the standards document 
    path = r"C:\Projects_ML\ESG-Report-Analyzer\ESG-Report-Analyzer-NLP\data\raw\standards"

    return DocumentHandler().load_standard_documents(standard_docs_dir=path)

standard_docs = load_standard_doc()

## Input section 
files = st.file_uploader("Upload ESG Reports", accept_multiple_files=True)

reports_ready = False

if files:
    docs = DocumentHandler().load_uploaded_documents(files)
    st.success(f"{len(docs)} ESG report(s) loaded")

    if st.button("📥 Index Documents"):
        with st.spinner("Creating vector stores..."):
            retriever.create_vector_store(
                standards_vector_path=r"...\vectorstores\standards",
                standard_docs=standard_docs,
                uploaded_report=docs,
                reports_vector_path=r"...\vectorstores\uploaded_reports"
            )

        reports_ready = True
        st.success("Vector stores created successfully")

st.subheader("2️⃣ Ask an ESG Question")

query = st.text_input(
    "Enter your ESG query",
    value="What environmental policies has the company implemented?"
)


if st.button("🚀 Run ESG Agent"):

    if not reports_ready:
        st.error("Please upload and index reports first.")
        st.stop()

    state = ESGstate(
        query=query,
        reports_ready=True
    )

    with st.spinner("Running ESG analysis pipeline..."):
        final_state = graph.invoke(state)

    st.success("Analysis complete!")
