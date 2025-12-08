"""
Retrieval service
Handles query processing and retrieval chain operations
"""

import logging
from typing import List, Optional, Any, Dict

from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from .api_client import get_api_client

logger = logging.getLogger(__name__)


class CustomLLM(LLM):
    """
    LLM wrapper backed by the configured gateway
    """

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        api_client = get_api_client()
        return api_client.complete(
            prompt,
            max_tokens=kwargs.get("max_tokens", 150),
            temperature=kwargs.get("temperature", 0),
        )


class CustomChatModel(BaseChatModel):
    """
    Chat model wrapper backed by the configured gateway
    """

    @property
    def _llm_type(self) -> str:
        return "custom_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        api_client = get_api_client()

        prompt_parts: List[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")

        full_prompt = "\n\n".join(prompt_parts)
        if not full_prompt.endswith("Assistant:"):
            full_prompt += "\n\nAssistant:"

        logger.info("Sending prompt to gateway (length: %s chars)", len(full_prompt))

        response_text = api_client.complete(
            full_prompt,
            max_tokens=kwargs.get("max_tokens", 150),
            temperature=kwargs.get("temperature", 0),
        )

        generations = [Generation(text=response_text)]
        return LLMResult(generations=[generations])


def get_llm() -> BaseChatModel:
    """
    Get chat model instance backed by the gateway
    """
    return CustomChatModel()


def build_retrieval_chain(vectorstore: FAISS):
    """
    Build retrieval chain with the custom chat model

    Args:
        vectorstore: FAISS vectorstore instance

    Returns:
        Configured retrieval chain
    """
    try:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        llm = get_llm()
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            retrieval_qa_chat_prompt,
        )
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain,
        )
        return retrieval_chain
    except Exception as e:
        logger.error("Error building retrieval chain: %s", e)
        raise


def query_documents(query: str, vectorstore: FAISS) -> Dict[str, str]:
    """
    Query the documents using RAG with custom embedding and inference

    Args:
        query: User question
        vectorstore: FAISS vectorstore instance

    Returns:
        Dictionary with answer and query
    """
    try:
        logger.info("Processing query: %s", query)

        api_client = get_api_client()

        logger.info("Creating query embedding")
        query_embedding = api_client.embed_text(query)
        logger.info("Query embedding dimension: %s", len(query_embedding))

        logger.info("Searching for similar documents")
        similar_docs = vectorstore.similarity_search_by_vector(query_embedding, k=4)
        logger.info("Found %s similar documents", len(similar_docs))

        if not similar_docs:
            return {
                "answer": "I could not find any relevant documents to answer your question.",
                "query": query,
            }

        context_parts: List[str] = []
        for i, doc in enumerate(similar_docs):
            context_parts.append(f"Document {i + 1}:\n{doc.page_content}")

        context = "\n\n".join(context_parts)
        logger.info("Context length: %s characters", len(context))

        prompt = f"""Based on the following documents, provide a clear answer that addresses the question.

Documents:
{context}

Question: {query}

Answer:"""

        logger.info("Calling gateway completion with prompt length: %s", len(prompt))

        answer = api_client.complete(
            prompt=prompt,
            max_tokens=200,
            temperature=0,
        ).strip()

        if not answer:
            answer = "I could not find a relevant answer in the documents."

        logger.info("Query completed successfully")

        return {
            "answer": answer,
            "query": query,
        }
    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        raise
