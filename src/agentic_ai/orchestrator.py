from src.utils.logging import esg_agents_logger
from src.nlp.retriever import Retriever
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Any, Optional, Dict, List
from pydantic import BaseModel, Field

retriever = Retriever()

#define the agent state
class ESGstate(BaseModel):
    query: str
    retrieved_docs: Optional[Dict[str, List]] = None
    classification: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
    response: Optional[str] = None
    messages: Annotated[list[dict], add_messages]

#retrieve ESG report node
def retrieve_node(state):
    query = state.query
    esg_agents_logger.info(f"Retrieving documents for query: {query}")


#classify ESG aspects node
def classify_node(state):
    pass

#ESG scores node 
def predict_node(state):
    pass

#summarize ESG findings node
def summarize_node(state):
    pass
