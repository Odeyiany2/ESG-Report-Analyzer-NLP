from src.utils.logging import esg_agents_logger
from src.nlp.retriever import Retriever
from src.nlp.embeddings import EmbeddingHandler
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Any, Optional, Dict, List
from pydantic import BaseModel, Field

retriever = Retriever(embedding_handler= EmbeddingHandler())

#define the agent state
class ESGstate(BaseModel):
    query: str
    reports_ready: bool = False
    retrieved_docs: Optional[Dict[str, List]] = None
    classification: Optional[Dict[str, Any]] = None
    #scores: Optional[Dict[str, float]] = None
    response: Optional[str] = None
    messages: Annotated[list[dict], add_messages]



#retrieve ESG report node
def retrieve_node(state: ESGstate) -> ESGstate:
    if not state.reports_ready:
        raise ValueError("Reports not uploaded or indexed.")
    query = state.query
    esg_agents_logger.info(f"Retrieving documents for query: {query}")
    response = retriever.retrieve_context(query)
    if not response:
        esg_agents_logger.error("No relevant documents found for the given query.")
        raise ValueError("No relevant documents found for the given query.")
    state.retrieved_docs = response
    return state


    #classify ESG aspects node
def classify_node(state: ESGstate) -> ESGstate:
    if not state.retrieved_docs:
        esg_agents_logger.error("No documents retrieved to classify.")
        raise ValueError("No documents retrieved to classify.")
    report_sections = state.retrieved_docs.get("reports", [])
    enriched_reports = retriever.enrich_context_with_esg(report_sections)
    if enriched_reports is None:
        esg_agents_logger.error("Failed to enrich report sections with ESG classifications.")
        raise ValueError("Failed to enrich report sections with ESG classifications.")
    state.classification = enriched_reports
    return state

    # #ESG scores node 
    # def predict_node(state):
    #     pass

#summarize ESG findings node
def summarize_node(state: ESGstate) -> ESGstate:
    if not state.retrieved_docs or not state.classification:
        esg_agents_logger.error("Insufficient data to summarize ESG findings.")
        raise ValueError("Insufficient data to summarize ESG findings.")
    esg_agents_logger.info("Summarizing ESG findings...")

    context = {
            "standards": state.retrieved_docs.get("standards", []),
            "reports": state.classification,
        }

    prompt = retriever.load_prompts(r"C:\Projects_ML\ESG-Report-Analyzer\ESG-Report-Analyzer-NLP\src\nlp\prompt_esg.yml")

    full_prompt = retriever.build_prompt(
            query=state.query,
            context=context,
            prompt_data = prompt 
        )

    response = retriever.run_analysis(full_prompt)
    state.response = response

    return state

def build_esg_agent_graph() -> StateGraph:
    graph = StateGraph(ESGstate)

    # Define nodes
    retrieve_tool = ToolNode(
        name="Retrieve ESG Reports",
        func=ESGstate.retrieve_node,
        description="Retrieves relevant ESG reports based on the user's query."
    )

    classify_tool = ToolNode(
        name="Classify ESG Aspects",
        func=ESGstate.classify_node,
        description="Classifies sections of the retrieved ESG reports into E, S, and G categories."
    )

    # predict_tool = ToolNode(
    #     name="Predict ESG Scores",
    #     func=ESGstate.predict_node,
    #     description="Predicts ESG scores based on classified report sections."
    # )

    summarize_tool = ToolNode(
        name="Summarize ESG Findings",
        func=ESGstate.summarize_node,
        description="Summarizes the ESG findings from the classified report sections."
    )

    # Add nodes to graph
    graph.add_node(retrieve_tool, START)
    graph.add_node(classify_tool, tools_condition(START))
    # graph.add_node(predict_tool, tools_condition(classify_tool))
    graph.add_node(summarize_tool, tools_condition(classify_tool))
    graph.add_node(END, tools_condition(summarize_tool))

    # Add memory saver checkpoint
    memory_saver = MemorySaver(ESGstate)
    graph.add_checkpoint(memory_saver)

    return graph.compile()