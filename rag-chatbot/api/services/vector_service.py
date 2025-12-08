"""
Vector store service
Handles FAISS vector store operations
"""

import os
import logging
import shutil
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

import config
from .api_client import get_api_client

logger = logging.getLogger(__name__)

class CustomEmbeddings(Embeddings):
    """
    Embeddings implementation backed by the configured gateway
    """

    def __init__(self) -> None:
        self.api_client = get_api_client()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.api_client.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.api_client.embed_text(text)


def get_embeddings() -> Embeddings:
    """
    Create embeddings instance

    Returns:
        Embeddings instance
    """
    return CustomEmbeddings()


def store_in_vector_storage(chunks: list) -> FAISS:
    """
    Create embeddings and store in FAISS vector store

    Args:
        chunks: List of document chunks

    Returns:
        FAISS vectorstore instance

    Raises:
        Exception: If storage operation fails
    """
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(config.VECTOR_STORE_PATH) or ".",
            exist_ok=True,
        )
        vectorstore.save_local(config.VECTOR_STORE_PATH)
        logger.info(f"Saved vector store to {config.VECTOR_STORE_PATH}")

        return vectorstore
    except Exception as e:
        logger.error(f"Error storing vectors: {str(e)}")
        raise


def load_vector_store() -> Optional[FAISS]:
    """
    Load existing FAISS vector store

    Returns:
        FAISS vectorstore instance or None if not found
    """
    try:
        if not os.path.exists(config.VECTOR_STORE_PATH):
            logger.warning(f"No vector store found at {config.VECTOR_STORE_PATH}")
            return None

        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            config.VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded existing FAISS vector store")
        return vectorstore
    except Exception as e:
        logger.warning(f"Could not load vector store: {str(e)}")
        return None


def delete_vector_store() -> bool:
    """
    Delete the vector store from disk

    Returns:
        True if deleted successfully, False otherwise

    Raises:
        Exception: If deletion fails
    """
    try:
        if os.path.exists(config.VECTOR_STORE_PATH):
            shutil.rmtree(config.VECTOR_STORE_PATH)
            logger.info("Deleted vector store")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting vector store: {str(e)}")
        raise
