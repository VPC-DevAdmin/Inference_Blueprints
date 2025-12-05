from openai import OpenAI
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with a local OpenAI-compatible API.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,  # kept for signature compatibility, ignored
        default_model: str = "gpt-4o-mini",
    ):
        """
        Initialize LLM client.

        Only uses a local OpenAI-compatible endpoint behind BASE_URL.

        Auth priority:
          1) Keycloak (if BASE_URL + Keycloak creds)
          2) INFERENCE_API_KEY
        """

        # Base URL for your local gateway (no trailing slash)
        self.base_url = settings.BASE_URL.rstrip("/") if settings.BASE_URL else None
        self.client: Optional[OpenAI] = None
        self.api_key: Optional[str] = None
        self.auth_mode: str = "none"

        if not self.base_url:
            logger.error(
                "LLMClient: BASE_URL is not set. Local inference endpoint is required."
            )
            return

        # 1) Try Keycloak first (if creds present)
        if settings.KEYCLOAK_CLIENT_ID and settings.KEYCLOAK_CLIENT_SECRET:
            token = self._try_keycloak_token()
            if token:
                self.api_key = token
                self.auth_mode = "keycloak"
                logger.info("LLMClient: using Keycloak access token for local gateway")

        # 2) Fall back to INFERENCE_API_KEY
        if not self.api_key and getattr(settings, "INFERENCE_API_KEY", None):
            self.api_key = settings.INFERENCE_API_KEY
            self.auth_mode = "inference_api_key"
            logger.info("LLMClient: using INFERENCE_API_KEY for local gateway")

        if not self.api_key:
            logger.error(
                "LLMClient: no auth configured. "
                "Set either KEYCLOAK_CLIENT_SECRET or INFERENCE_API_KEY."
            )
            return

        # Model name exposed by your local gateway
        self.default_model = getattr(settings, "INFERENCE_MODEL_NAME", None) or default_model

        # Create a single OpenAI-style client pointed at your gateway /v1
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1",
        )
        logger.info(
            "LLMClient: configured local OpenAI-compatible client at %s/v1 (auth_mode=%s)",
            self.base_url,
            self.auth_mode,
        )

    def _try_keycloak_token(self) -> Optional[str]:
        """
        Try to obtain an access token from Keycloak via the local gateway.

        Assumes a /token endpoint exposed behind BASE_URL.
        """
        if not self.base_url:
            return None

        token_url = f"{self.base_url}/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": settings.KEYCLOAK_CLIENT_ID,
            "client_secret": settings.KEYCLOAK_CLIENT_SECRET,
        }

        try:
            resp = requests.post(token_url, data=payload, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token")
                if token:
                    return token
                logger.error("LLMClient: Keycloak response missing access_token")
            else:
                logger.error(
                    "LLMClient: Keycloak token request failed %s: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as e:
            logger.error(f"LLMClient: Keycloak token request error: {str(e)}")

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def generate_with_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """
        Generate text using the local OpenAI-compatible endpoint.

        Always uses /v1/chat/completions behind BASE_URL.
        """
        if not self.client:
            raise ValueError(
                "LLM client is not configured. "
                "Check BASE_URL and either KEYCLOAK_CLIENT_SECRET or INFERENCE_API_KEY."
            )

        use_model = model or self.default_model

        logger.info(
            "LLMClient: generating with model=%s (auth_mode=%s)",
            use_model,
            self.auth_mode,
        )

        try:
            response = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.info("LLMClient: generated %d characters", len(content))
            return content

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str = "openai",
        **kwargs,
    ) -> str:
        """
        Public wrapper used by the rest of the app.

        provider is ignored and kept only for compatibility.
        """
        return await self.generate_with_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count for text.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            return len(text) // 4  # rough estimate

    def is_available(self, provider: str = "openai") -> bool:
        """
        Check if LLM client is available (local gateway only).
        """
        return self.client is not None
