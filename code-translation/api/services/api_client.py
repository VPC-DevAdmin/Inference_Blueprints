"""
API Client for authentication and code translation
"""

import logging
from typing import Optional

import requests
import httpx
from openai import OpenAI
import config

logger = logging.getLogger(__name__)


class APIClient:
    """
    Auth priority:
      1) Keycloak (if BASE_URL, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET set)
      2) INFERENCE_API_KEY from config
      3) Open / no-auth mode
    """

    def __init__(self) -> None:
        self.base_url: Optional[str] = getattr(config, "BASE_URL", None)
        self.token: Optional[str] = None
        self.api_key: Optional[str] = getattr(config, "INFERENCE_API_KEY", None)
        self.http_client = httpx.Client(verify=False)

        self.auth_mode = "open"
        self._init_auth()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _init_auth(self) -> None:
        """Choose auth method in priority order."""
        # 1) Keycloak
        if (
            self.base_url
            and getattr(config, "KEYCLOAK_CLIENT_ID", None)
            and getattr(config, "KEYCLOAK_CLIENT_SECRET", None)
        ):
            token_url = f"{self.base_url.rstrip('/')}/token"
            payload = {
                "grant_type": "client_credentials",
                "client_id": config.KEYCLOAK_CLIENT_ID,
                "client_secret": config.KEYCLOAK_CLIENT_SECRET,
            }

            try:
                logger.info("Authenticating with Keycloak at %s", token_url)
                response = requests.post(token_url, data=payload, verify=False)

                if response.status_code == 200:
                    self.token = response.json().get("access_token")
                    if self.token:
                        self.auth_mode = "keycloak"
                        logger.info("Keycloak authentication successful")
                        return
                else:
                    logger.warning(
                        "Keycloak auth failed: %s %s",
                        response.status_code,
                        response.text,
                    )
            except Exception as exc:
                logger.warning("Keycloak authentication error: %s", exc)

        # 2) API Key
        if self.api_key:
            self.auth_mode = "api_key"
            logger.info("Using INFERENCE_API_KEY for authentication")
            return

        # 3) Open mode
        logger.info("No auth configured, running in open mode")
        self.auth_mode = "open"

    # ------------------------------------------------------------------
    # OpenAI Client Builder
    # ------------------------------------------------------------------
    def _get_openai_client(self) -> OpenAI:
        """Construct an OpenAI-compatible client pointed at BASE_URL/v1/chat/completions"""
        if not self.base_url:
            raise ValueError("BASE_URL must be configured")

        # Choose which key the OpenAI client sends
        if self.auth_mode == "keycloak" and self.token:
            key = self.token
        elif self.auth_mode == "api_key" and self.api_key:
            key = self.api_key
        else:
            key = "no-auth"

        full_base = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        return OpenAI(
            api_key=key,
            base_url=full_base,
            http_client=self.http_client,
        )

    def get_inference_client(self) -> OpenAI:
        return self._get_openai_client()

    # ------------------------------------------------------------------
    # Code Translation
    # ------------------------------------------------------------------
    def translate_code(self, source_code: str, source_lang: str, target_lang: str) -> str:
        """Translate code using OpenAI-style chat completions."""
        client = self._get_openai_client()

        system_prompt = (
            "You are a senior software engineer. Translate code between languages. "
            "Output only the translated code. Do not add markdown or explanations."
        )

        user_prompt = f"""
Translate the following {source_lang} code into {target_lang}.
Return only the translated {target_lang} code. No markdown fences.

Source ({source_lang}):
{source_code}
"""

        try:
            response = client.chat.completions.create(
                model=config.INFERENCE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )

            content = response.choices[0].message.content or ""
            return self._strip_code_fences(content)

        except Exception as exc:
            logger.error("Code translation failed: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```python fences and language tags."""
        t = text.strip()

        if t.startswith("```") and t.endswith("```"):
            t = t[3:-3].strip()

        # Remove leading language tag line if present
        first_line = t.split("\n", 1)[0].lower().strip()
        if first_line in {"python", "java", "c", "cpp", "go", "rust"}:
            t = "\n".join(t.split("\n")[1:])

        return t.strip()

    def is_authenticated(self) -> bool:
        return (
            (self.auth_mode == "keycloak" and self.token)
            or (self.auth_mode == "api_key" and self.api_key)
        )


# Global singleton
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
