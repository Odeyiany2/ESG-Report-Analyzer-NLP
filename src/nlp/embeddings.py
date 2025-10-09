from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils.logging import embedding_logger

class EmbeddingHandler:
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "cpu"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
    def text_split(self, documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
        """
        Splits documents into smaller chunks for better embedding performance.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def embed_documents(self, documents: List[Document], persist_directory: str) -> Chroma:
        """
        Embeds documents and stores them in a Chroma vector store.
        """
        try:
            docs = self.text_split(documents)
            vectordb = Chroma.from_documents(
                docs,
                embedding=self.embedding_model,
                persist_directory=persist_directory
            )
            vectordb.persist()
            embedding_logger.info(f"Documents embedded and stored at {persist_directory}")
            return vectordb
        except Exception as e:
            embedding_logger.error(f"Error during embedding or storing documents: {e}")
            print(f"Error during embedding or storing documents: {e}")
    
    def load_vectorstore(self, persist_directory: str) -> Chroma:
        """
        Loads an existing Chroma vector store from the specified directory.
        """
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )