from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def embed_standards(standards: List[str], persist_directory: str) -> Chroma:
    """
    Embeds a list of ESG standards -> for the banks and consumer finance -> using HuggingFaceEmbeddings 
    and stores them in a Chroma vector store.
    """

    # initialize the embedding model
    embedding_model  = HuggingFaceEmbeddings(
        model_name = "intfloat/e5-base-v2", 
        model_kwargs = {'device': 'cpu'}
    )

    # create documents from the standards
    documents = [Document(page_content=standard) for standard in standards]

    # split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        separators = ["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    # create the Chroma vector store
    vector_store = Chroma.from_documents(
        documents = texts,
        embedding = embedding_model,
        persist_directory = persist_directory
    )

    return vector_store
