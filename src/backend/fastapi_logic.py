import uuid
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Dict
from src.nlp.doc_handler import DocumentHandler
from src.nlp.embeddings import EmbeddingHandler
from src.nlp.retriever import Retriever
from src.agentic_ai.orchestrator import run_esg_analysis
from src.utils.logging import api_logger


#load environment variables from .env file
load_dotenv()

#initialize FastAPI app
app = FastAPI(
    title="ESG Report Analyzer",
    description="AI-powered ESG report analysis against GRI, SASB, and IFRS standards.",
    version="1.0.0"

)

#load the standard ESG documents from the specified directory
standard_docs_dir = os.path.normpath(os.getenv("STANDARD_DOCS_DIR", "data/standards"))
doc_handler = DocumentHandler()
standard_documents = doc_handler.load_standard_documents(standard_docs_dir)

#get the file paths for vector stores from environment variables
standards_vector_path = os.path.normpath(os.getenv("STANDARD_VECTORSTORE_PATH", "vectorstores/standards"))
reports_vector_path = os.path.normpath(os.getenv("REPORTS_VECTORSTORE_PATH", "vectorstores/uploaded_reports"))

#get the file path for the prompts yaml file
prompts_file_path = os.path.normpath(os.getenv("PROMPTS_FILE_PATH")) if os.getenv("PROMPTS_FILE_PATH") else None

#initialize a dictionary to store user uploaded reports in-memory (keyed by session ID)
user_uploaded_report: Dict[str, List] = {}

#initialize embedding handler once at startup (expensive operation)
try:
    api_logger.info("Initializing embedding handler at app startup...")
    embedding_handler = EmbeddingHandler()
    api_logger.info("Embedding handler initialized successfully.")
except Exception as e:
    api_logger.error(f"Failed to initialize embedding handler: {e}")
    embedding_handler = None

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the API is running smoothly.
    """
    return {
        "status": "ok", 
        "message": "API is running smoothly.",
        "standard_docs_loaded": len(standard_documents),}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint to handle user uploaded ESG reports for analysis
    """
    try:
        doc_handler = DocumentHandler()
        docs = doc_handler.load_uploaded_documents(files)
        if not docs:
            api_logger.error("No valid documents were uploaded.")
            raise HTTPException(status_code=400, detail="No valid documents were uploaded.")
        session_id = str(uuid.uuid4())
        user_uploaded_report[session_id] = docs
        api_logger.info(f"Successfully processed {len(docs)} documents from upload with session ID {session_id}.")
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully uploaded and processed {len(docs)} documents.",
                "session_id": session_id
            },
            status_code=200
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error processing uploaded documents: {e}")
        raise HTTPException(status_code=500, detail="Error processing uploaded documents.")



@app.post("/query")
async def query_assistant(request:Request):
    """
    Accepts a user query and session_id, runs the full ESG analysis pipeline
    (retrieve → classify → summarize), and returns the LLM-generated analysis.
    """
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()
        
        if not query:
            api_logger.error("No query provided in the request.")
            raise HTTPException(status_code=400, detail="No query provided.")
        
        #retrieve the uploaded reports for the given session ID
        uploaded_reports = user_uploaded_report.get(session_id, [])
        if not uploaded_reports:
            api_logger.error(f"No uploaded reports found for session ID {session_id}.")
            raise HTTPException(status_code=400, detail="No uploaded reports found for the provided session ID.")
        
        
        #check if embedding handler is initialized
        if embedding_handler is None:
            api_logger.error("Embedding handler not initialized.")
            raise HTTPException(status_code=500, detail="System not ready. Embedding handler failed to initialize.")
        
        #initialize the retriever with the pre-initialized embedding handler
        retriever = Retriever(embedding_handler=embedding_handler)

        session_reports_path = os.path.normpath(os.path.join(reports_vector_path, session_id))
        retriever.create_vector_store(
            standards_vector_path=standards_vector_path,
            standard_docs=standard_documents,
            uploaded_report=uploaded_reports,
            reports_vector_path= session_reports_path
        )

        api_logger.info(f"Session {session_id}: running ESG analysis for query: '{query}'")
        response = run_esg_analysis(
            query=query,
            retriever=retriever,
            prompt_file=prompts_file_path,
            thread_id=session_id
        )
        if not response:
            api_logger.error(f"Session {session_id}: LLM returned empty response.")
            raise HTTPException(status_code=500, detail="No response generated. Please try again.")
 
        api_logger.info(f"Session {session_id}: analysis complete.")
        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "response": response
            },
            status_code=200
        )
 
    except HTTPException:
        raise  # Pass our explicit HTTP errors through cleanly
    except Exception as e:
        api_logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query.")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Removes a session's uploaded documents from memory.
    Call this when the user is done to free up memory.
    Note: this does not delete the vector store files from disk.
    """
    if session_id not in user_uploaded_report:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
 
    del user_uploaded_report[session_id]
    api_logger.info(f"Session {session_id} cleared from memory.")
    return JSONResponse(
        content={"status": "success", "message": f"Session '{session_id}' cleared."},
        status_code=200
    )