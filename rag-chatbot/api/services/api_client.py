import logging
from typing import Optional, List

import httpx
import requests

import config

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self) -> None:
        # Raw values from config
        self.base_url = (getattr(config, "BASE_URL", "") or "").rstrip("/")
        self.embeddings_base_url = (
            getattr(config, "EMBEDDINGS_BASE_URL", "") or self.base_url
        ).rstrip("/")

        self.http_client: Optional[httpx.Client] = None
        self.api_key: Optional[str] = None
        self.auth_mode: str = "unconfigured"

        self._configure_http_client()
        self._configure_auth()

    # ---------------------------
    # HTTP client and auth setup
    # ---------------------------

    def _configure_http_client(self) -> None:
        # You can tighten this if you want TLS verification
        self.http_client = httpx.Client(verify=False)
        logger.info(
            "HTTP client initialized. "
            "Inference base_url=%r, embeddings_base_url=%r",
            self.base_url,
            self.embeddings_base_url,
        )

    def _try_keycloak_token(self) -> Optional[str]:
        token_url = getattr(config, "KEYCLOAK_TOKEN_URL", "") or ""
        client_id = getattr(config, "KEYCLOAK_CLIENT_ID", "") or ""
        client_secret = getattr(config, "KEYCLOAK_CLIENT_SECRET", "") or ""

        if not (token_url and client_id and client_secret):
            return None

        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        try:
            logger.info("Requesting Keycloak token from %s", token_url)
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
        # Try Keycloak first if configured
        token = self._try_keycloak_token()
        if token:
            self.api_key = token
            self.auth_mode = "keycloak"
            logger.info("Gateway auth configured with Keycloak token")
            return

        # Fallback to static API key
        api_key = getattr(config, "INFERENCE_API_KEY", "") or ""
        if api_key:
            self.api_key = api_key
            self.auth_mode = "inference_api_key"
            logger.info("Gateway auth configured with static INFERENCE_API_KEY")
            return

        raise ValueError(
            "No gateway auth configured. "
            "Set KEYCLOAK_TOKEN_URL + KEYCLOAK_CLIENT_ID + KEYCLOAK_CLIENT_SECRET "
            "or INFERENCE_API_KEY."
        )

    # ---------------------------
    # Base URL helpers
    # ---------------------------

    def _normalize_base(self, url: str) -> str:
        """
        Ensure the base URL ends with /v1 exactly once.

        Accepts:
          http://host:8080
          http://host:8080/
          http://host:8080/v1
          http://host:8080/v1/
        And always returns:
          http://host:8080/v1
        """
        if not url:
            raise ValueError("Base URL is empty for OpenAI compatible client")

        cleaned = url.rstrip("/")
        if cleaned.endswith("/v1"):
            return cleaned
        return f"{cleaned}/v1"

    def get_embedding_client(self):
        from openai import OpenAI

        base = self._normalize_base(self.embeddings_base_url)
        logger.info("Using embeddings endpoint base_url=%s", base)
        return OpenAI(
            api_key=self.api_key,
            base_url=base,
            http_client=self.http_client,
        )

    def get_inference_client(self):
        from openai import OpenAI

        base = self._normalize_base(self.base_url)
        logger.info("Using inference endpoint base_url=%s", base)
        return OpenAI(
            api_key=self.api_key,
            base_url=base,
            http_client=self.http_client,
        )

    # ---------------------------
    # Embeddings helpers
    # ---------------------------

    def embed_text(self, text: str) -> List[float]:
        try:
            client = self.get_embedding_client()
            resp = client.embeddings.create(
                model=getattr(config, "EMBEDDINGS_MODEL_NAME", ""),
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error("Embedding error: %s", e, exc_info=True)
            raise

    def embed_texts(self, texts: list) -> list:
        try:
            client = self.get_embedding_client()
            batch_size = 32
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                logger.info(
                    "Requesting embeddings batch %d:%d from %s",
                    i,
                    i + len(batch),
                    self.embeddings_base_url,
                )
                resp = client.embeddings.create(
                    model=getattr(config, "EMBEDDINGS_MODEL_NAME", ""),
                    input=batch,
                )
                all_embeddings.extend([item.embedding for item in resp.data])

            return all_embeddings
        except Exception as e:
            logger.error("Batch embedding error: %s", e, exc_info=True)
            raise

    # ---------------------------
    # Completion helpers
    # ---------------------------

    def complete(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str:
        try:
            client = self.get_inference_client()
            resp = client.chat.completions.create(
                model=getattr(config, "INFERENCE_MODEL_NAME", ""),
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
                model=getattr(config, "INFERENCE_MODEL_NAME", ""),
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
