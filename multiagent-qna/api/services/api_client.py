"""
API Client for authentication and API calls
Similar to rag-chatbot implementation
"""

import logging
import requests
import httpx
from typing import Optional
import config

logger = logging.getLogger(__name__)


import logging
import requests
import httpx
from typing import Optional
import config

logger = logging.getLogger(__name__)


class APIClient:
    """
    Client for handling authentication and API calls
    """

    def __init__(self):
        # Make BASE_URL safe and consistent
        self.base_url = getattr(config, "BASE_URL", "").rstrip("/")
        self.token: Optional[str] = None
        self.api_key: Optional[str] = getattr(config, "INFERENCE_API_KEY", None)

        # Single httpx client reused for all calls
        self.http_client = httpx.Client(verify=False)

        # Auth priority:
        # 1) Keycloak, if fully configured
        # 2) INFERENCE_API_KEY
        # 3) No auth
        if (
            getattr(config, "KEYCLOAK_CLIENT_ID", None)
            and getattr(config, "KEYCLOAK_CLIENT_SECRET", None)
        ):
            logger.info("Trying Keycloak authentication")
            self._authenticate()
        elif self.api_key:
            self.token = self.api_key
            logger.info("Using INFERENCE_API_KEY for authentication")
        else:
            logger.info("No auth configured, running without auth header")

    def _authenticate(self) -> None:
        """
        Authenticate and obtain access token from Keycloak
        Only called when KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET are set.
        """
        if not self.base_url:
            logger.error("BASE_URL is not set, cannot perform Keycloak auth")
            return

        token_url = f"{self.base_url}/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": config.KEYCLOAK_CLIENT_ID,
            "client_secret": config.KEYCLOAK_CLIENT_SECRET,
        }

        try:
            response = requests.post(token_url, data=payload, verify=False)

            if response.status_code == 200:
                self.token = response.json().get("access_token")
                if self.token:
                    logger.info("Access token obtained via Keycloak")
                else:
                    logger.error("Keycloak response did not include access_token")
            else:
                logger.error(
                    "Error obtaining token: %s - %s",
                    response.status_code,
                    response.text,
                )
        except Exception as e:
            logger.error("Error during Keycloak authentication: %s", str(e))

    def get_embedding_client(self):
        """
        Get OpenAI-style client for embeddings
        Uses bge-base-en-v1.5 endpoint
        """
        from openai import OpenAI

        return OpenAI(
            api_key=self.token or "",
            base_url=f"{self.base_url}/{config.EMBEDDING_MODEL_ENDPOINT}/v1",
            http_client=self.http_client,
        )

    def get_inference_client(self):
        """
        Get OpenAI-style client for inference/completions
        Uses Llama-3.1-8B-Instruct endpoint
        """
        from openai import OpenAI

        return OpenAI(
            api_key=self.token or "",
            base_url=f"{self.base_url}/{config.INFERENCE_MODEL_ENDPOINT}/v1",
            http_client=self.http_client,
        )

    def embed_text(self, text: str) -> list:
        """
        Get embedding for text
        Uses the bge-base-en-v1.5 embedding model
        """
        try:
            client = self.get_embedding_client()
            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL_NAME,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_texts(self, texts: list) -> list:
        """
        Get embeddings for multiple texts
        Batches requests to avoid exceeding API limits (max batch size: 32)
        """
        try:
            BATCH_SIZE = 32
            all_embeddings = []
            client = self.get_embedding_client()

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                logger.info(
                    "Processing embedding batch %d/%d (%d texts)",
                    (i // BATCH_SIZE) + 1,
                    (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE,
                    len(batch),
                )

                response = client.embeddings.create(
                    model=config.EMBEDDING_MODEL_NAME,
                    input=batch,
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def chat_complete(
        self,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Get chat completion from the inference model
        """
        try:
            client = self.get_inference_client()

            # Same prompt construction as before
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant:"

            logger.info("Calling inference with prompt length: %d", len(prompt))

            response = client.completions.create(
                model=config.INFERENCE_MODEL_NAME,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "text"):
                    return choice.text
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
                logger.error("Unexpected response choice structure: %r", choice)
                return str(choice)
            else:
                logger.error("Unexpected response object: %r", response)
                return ""
        except Exception as e:
            logger.error("Error generating chat completion: %s", str(e), exc_info=True)
            raise

    def __del__(self):
        if self.http_client:
            self.http_client.close()


# Global API client instance
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """
    Get or create the global API client instance
    
    Returns:
        APIClient instance
    """
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client

