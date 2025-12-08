import logging
from typing import Optional

import httpx
import requests

import config

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self) -> None:
        self.base_url = config.BASE_URL.rstrip("/")
        self.embeddings_base_url = config.EMBEDDINGS_BASE_URL.rstrip("/")
        self.http_client: Optional[httpx.Client] = None
        self.api_key: Optional[str] = None
        self.auth_mode: str = "unconfigured"

        self._configure_http_client()
        self._configure_auth()

    def _configure_http_client(self) -> None:
        self.http_client = httpx.Client(verify=False)
        logger.info("HTTP client initialized for gateway")

    def _try_keycloak_token(self) -> Optional[str]:
        if not (config.KEYCLOAK_CLIENT_ID and config.KEYCLOAK_CLIENT_SECRET):
            return None

        token_url = f"{self.base_url}/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": config.KEYCLOAK_CLIENT_ID,
            "client_secret": config.KEYCLOAK_CLIENT_SECRET,
        }

        try:
            logger.info("Requesting Keycloak token")
            resp = requests.post(token_url, data=payload, verify=False)
            if resp.status_code != 200:
                logger.error(
                    "Keycloak token request failed: %s %s",
                    resp.status_code,
                    resp.text,
                )
                return None

            token = resp.json().get("access_token")
            if not token:
                logger.error("Keycloak response missing access_token")
                return None

            return token
        except Exception as e:
            logger.error("Keycloak auth error: %s", e)
            return None

    def _configure_auth(self) -> None:
        token = self._try_keycloak_token()
        if token:
            self.api_key = token
            self.auth_mode = "keycloak"
            logger.info("Gateway auth configured with Keycloak token")
            return

        if config.INFERENCE_API_KEY:
            self.api_key = config.INFERENCE_API_KEY
            self.auth_mode = "inference_api_key"
            logger.info("Gateway auth configured with static INFERENCE_API_KEY")
            return

        raise ValueError(
            "No gateway auth. Set KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET "
            "or INFERENCE_API_KEY."
        )

    def _normalize_base(self, url: str) -> str:
        if url.endswith("/v1"):
            return url
        return f"{url}/v1"

    def get_embedding_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self._normalize_base(self.embeddings_base_url),
            http_client=self.http_client,
        )

    def get_inference_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self._normalize_base(self.base_url),
            http_client=self.http_client,
        )

    def embed_text(self, text: str) -> list:
        try:
            client = self.get_embedding_client()
            resp = client.embeddings.create(
                model=config.EMBEDDING_MODEL_NAME,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error("Embedding error: %s", e)
            raise

    def embed_texts(self, texts: list) -> list:
        try:
            client = self.get_embedding_client()
            batch_size = 32
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(
                    model=config.EMBEDDING_MODEL_NAME,
                    input=batch,
                )
                all_embeddings.extend([item.embedding for item in resp.data])

            return all_embeddings
        except Exception as e:
            logger.error("Batch embedding error: %s", e)
            raise

    def complete(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str:
        try:
            client = self.get_inference_client()
            resp = client.chat.completions.create(
                model=config.INFERENCE_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if resp.choices:
                msg = resp.choices[0].message
                return msg.content if getattr(msg, "content", None) else ""
            logger.error("Unexpected completion response: %r", resp)
            return ""
        except Exception as e:
            logger.error("Completion error: %s", e, exc_info=True)
            raise

    def chat_complete(
        self,
        messages: list,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str:
        try:
            client = self.get_inference_client()
            resp = client.chat.completions.create(
                model=config.INFERENCE_MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if resp.choices:
                msg = resp.choices[0].message
                return msg.content if getattr(msg, "content", None) else ""
            logger.error("Unexpected chat response: %r", resp)
            return ""
        except Exception as e:
            logger.error("Chat completion error: %s", e, exc_info=True)
            raise

    def __del__(self) -> None:
        if self.http_client:
            self.http_client.close()


_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
