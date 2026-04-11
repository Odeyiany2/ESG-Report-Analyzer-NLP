from src.utils.logging import esg_agents_logger
from src.nlp.retriever import Retriever
from src.nlp.embeddings import EmbeddingHandler
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Any, Optional, Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field



#define the agent state
class ESGstate(TypedDict):
    query: str
    reports_ready: bool = False
    retrieved_docs: Optional[Dict[str, List[str]]] = None
    enriched_reports: Optional[List[Dict[str, Any]]] = None
    #scores: Optional[Dict[str, float]] = None
    response: Optional[str] = None
    



#retrieve ESG report node
def retrieve_node(state: ESGstate, retriever: Retriever) -> ESGstate:
    """Retrieves relevant ESG report sections and standard references."""
    if not state.reports_ready:
        esg_agents_logger.error("ESG reports not uploaded or indexed before retrieval.")
        raise ValueError("Reports not uploaded or indexed.")
    query = state["query"]
    esg_agents_logger.info(f"Retrieving documents for query: {query}")
    retrieved = retriever.retrieve_context(query)
    if not retrieved:
        esg_agents_logger.error("No relevant documents found for the given query.")
        raise ValueError("No relevant documents found for the given query.")
    return {"retrieved_docs": retrieved}

#classify ESG aspects node
def classify_node(state: ESGstate, retriever: Retriever) -> dict:
    """classify retrieved report sections into E, S, G categories using FinBert"""
    retrieved_docs = state.get("retrieved_docs")
    if not retrieved_docs:
        esg_agents_logger.error("No documents retrieved to classify.")
        raise ValueError("No documents retrieved to classify.")
    report_sections = retrieved_docs.get("reports", [])
    enriched = retriever.enrich_context_with_esg(report_sections)
    if enriched is None:
        esg_agents_logger.error("Failed to enrich report sections with ESG classifications.")
        raise ValueError("Failed to enrich report sections with ESG classifications.")
    return {"enriched_reports": enriched}

#summarize ESG findings node
def summarize_node(state: ESGstate, retriever: Retriever, prompt_file: str) -> ESGstate:
    """Builds the full prompt and runs LLM analysis to produce the final response."""
    retrieved_docs = state.get("retrieved_docs")
    enriched_reports = state.get("enriched_reports")
    if not retrieved_docs or not enriched_reports:
        esg_agents_logger.error("Insufficient data to summarize.")
        raise ValueError("Both retrieved_docs and enriched_reports must be present before summarizing.")
 
    esg_agents_logger.info("Building prompt and running LLM analysis...")
    serialized_reports = [
        f"[{s['esg_label']} | confidence: {s['confidence']}]\n"
        f"{s['text']}\n"
        f"Key tokens: {s.get('important_tokens', 'N/A')}"
        for s in enriched_reports
    ]
    context = {
            "standards": retrieved_docs.get("standards", []),
            "reports": serialized_reports,
        }

    prompt = retriever.load_prompts(prompt_file)
    full_prompt = retriever.build_prompt(
            query=state["query"],
            context=context,
            prompt_data = prompt 
        )

    response = retriever.run_analysis(full_prompt)

    return {"response": response}

def build_esg_agent_graph(retriever: Retriever, prompt_file: str) -> any:
    """
    Builds and compiles the LangGraph ESG analysis pipeline.
 
    Args:
        retriever: An initialized Retriever instance with vector stores loaded.
        prompts_file: Path to the prompt_esg.yml file.
 
    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    graph = StateGraph(ESGstate)

    graph.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    graph.add_node("classify", lambda state: classify_node(state, retriever))
    graph.add_node("summarize", lambda state: summarize_node(state, retriever, prompt_file))

    #define the pipeline flow
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "summarize")
    graph.add_edge("summarize", END)


    #memory saver to keep track of the agent's reasoning process
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)