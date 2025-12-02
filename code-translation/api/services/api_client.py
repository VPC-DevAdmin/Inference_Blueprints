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
      3) Open mode (dummy key, backend is expected to ignore)
    """

    def __init__(self) -> None:
        self.base_url: Optional[str] = getattr(config, "BASE_URL", None)
        self.token: Optional[str] = None              # Keycloak access token
        self.api_key: Optional[str] = getattr(config, "INFERENCE_API_KEY", None)
        self.http_client: httpx.Client = httpx.Client(verify=False)
        # "keycloak", "api_key", or "open"
        self.auth_mode: str = "open"

        self._init_auth()

    def _init_auth(self) -> None:
        """Decide how to authenticate: Keycloak, API key, or open."""
        # Tier 1: Keycloak
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
                    logger.warning("Keycloak response did not contain an access_token")
                else:
                    logger.warning(
                        "Keycloak auth failed: %s - %s",
                        response.status_code,
                        response.text,
                    )
            except Exception as exc:
                logger.warning("Keycloak auth error: %s", exc)

        # Tier 2: direct API key
        if self.api_key:
            self.auth_mode = "api_key"
            logger.info("Using INFERENCE_API_KEY for authentication")
            return

        # Tier 3: open mode
        self.auth_mode = "open"
        logger.info("No authentication configured, running in open mode")

    def _get_openai_client(self) -> OpenAI:
        """
        Build an OpenAI style client pointed at your inference endpoint.

        Example:
          BASE_URL = "https://inference-on-ibm.edgecollaborate.com"
          INFERENCE_MODEL_ENDPOINT = "v1/chat/completions"
          -> full base: "https://inference-on-ibm.edgecollaborate.com/v1/chat/completions"
        """
        if not self.base_url:
            raise ValueError("BASE_URL is not configured in config.py")

        if self.auth_mode == "keycloak" and self.token:
            key = self.token
        elif self.auth_mode == "api_key" and self.api_key:
            key = self.api_key
        else:
            # Open mode, backend is expected to ignore or not enforce the key
            key = "no-auth"

        endpoint_path = getattr(config, "INFERENCE_MODEL_ENDPOINT", "").strip("/")
        full_base = (
            f"{self.base_url.rstrip('/')}/{endpoint_path}"
            if endpoint_path
            else self.base_url.rstrip("/")
        )

        return OpenAI(
            api_key=key,
            base_url=full_base,
            http_client=self.http_client,
        )

    def get_inference_client(self) -> OpenAI:
        """Public accessor for the underlying OpenAI style client."""
        return self._get_openai_client()

    def translate_code(self, source_code: str, source_lang: str, target_lang: str) -> str:
        """
        Translate code from one language to another using the configured model.

        Args:
            source_code: Code to translate
            source_lang: Source programming language
            target_lang: Target programming language

        Returns:
            Translated code as plain text (no markdown fences)
        """
        client = self._get_openai_client()

        system_prompt = (
            "You are a senior software engineer that translates code from one "
            "language to another. Preserve logic and structure. Output only the "
            "translated code, with no explanations and no markdown formatting."
        )

        user_prompt = f"""Translate the following {source_lang} code to {target_lang}.

Return only the {target_lang} code, without comments or explanations,
and without markdown code fences.

{source_lang} code:
```{source_lang}
{source_code}
```"""

        logger.info("Translating code from %s to %s", source_lang, target_lang)

        try:
            response = client.chat.completions.create(
                model=config.INFERENCE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=getattr(config, "LLM_MAX_TOKENS", 2048),
                temperature=getattr(config, "LLM_TEMPERATURE", 0.2),
            )

            if not hasattr(response, "choices") or not response.choices:
                logger.error("Unexpected response structure: %r", response)
                return ""

            content = response.choices[0].message.content or ""
            translated = self._strip_code_fences(content)
            logger.info("Successfully translated code (%d characters)", len(translated))
            return translated

        except Exception as exc:
            logger.error("Error translating code: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        Remove common markdown code fences and language tags.
        """
        stripped = text.strip()

        # Remove surrounding ```...``` if present
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped[3:-3].strip()

        # Remove a leading language tag like "python\n"
        first_newline = stripped.find("\n")
        if first_newline != -1:
            first_line = stripped[:first_newline].strip().lower()
            if first_line in {"python", "java", "c", "cpp", "rust", "go", "ts", "js", "javascript", "csharp"}:
                stripped = stripped[first_newline + 1 :].lstrip()

        return stripped

    def is_authenticated(self) -> bool:
        """
        Consider the client authenticated if we either:
          - have a Keycloak token, or
          - have an API key configured
        """
        if self.auth_mode == "keycloak" and self.token:
            return True
        if self.auth_mode == "api_key" and self.api_key:
            return True
        return False

    def __del__(self) -> None:
        if self.http_client:
            try:
                self.http_client.close()
            except Exception:
                pass


# Global API client instance
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """
    Get or create the global API client instance.
    """
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
