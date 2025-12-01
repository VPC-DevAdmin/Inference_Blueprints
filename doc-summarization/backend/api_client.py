import logging

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
      3) Open mode (no auth)
    """

    def __init__(self) -> None:
        self.base_url = getattr(config, "BASE_URL", None)
        self.token = None          # Keycloak access token
        self.api_key = getattr(config, "INFERENCE_API_KEY", None)
        self.http_client = httpx.Client(verify=False)
        # "keycloak", "api_key", or "open"
        self.auth_mode = "open"

        self._init_auth()

    def _init_auth(self) -> None:
        """Establish auth mode: try Keycloak, then API key, then open."""
        # Tier 1: Keycloak, only if fully configured
        if (
            self.base_url
            and getattr(config, "KEYCLOAK_CLIENT_ID", None)
            and getattr(config, "KEYCLOAK_CLIENT_SECRET", None)
        ):
            try:
                token_url = f"{self.base_url}/token"
                logger.info("Authenticating with Keycloak at %s", token_url)

                payload = {
                    "grant_type": "client_credentials",
                    "client_id": config.KEYCLOAK_CLIENT_ID,
                    "client_secret": config.KEYCLOAK_CLIENT_SECRET,
                }

                response = requests.post(token_url, data=payload, verify=False)

                if response.status_code == 200:
                    self.token = response.json().get("access_token")
                    self.auth_mode = "keycloak"
                    logger.info("Keycloak authentication successful")
                    return
                else:
                    logger.warning(
                        "Keycloak auth failed: %s - %s",
                        response.status_code,
                        response.text,
                    )
            except Exception as exc:
                logger.warning("Keycloak auth error: %s", exc)

        # Tier 2: inference API key
        if self.api_key:
            self.auth_mode = "api_key"
            logger.info("Using INFERENCE_API_KEY for authentication")
            return

        # Tier 3: open mode
        self.auth_mode = "open"
        logger.info("No authentication configured, running in open mode")

    def get_inference_client(self) -> OpenAI:
        """
        Get OpenAI style client for inference.

        Uses:
          - Keycloak token if auth_mode is keycloak
          - INFERENCE_API_KEY if auth_mode is api_key
          - Dummy key in open mode, since OpenAI client requires api_key
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

        return OpenAI(
            api_key=key,
            base_url=f"{self.base_url}/{config.INFERENCE_MODEL_ENDPOINT}/v1",
            http_client=self.http_client,
        )

    def is_authenticated(self) -> bool:
        """
        In this model, "authenticated" means we have either
        a Keycloak token or an API key.
        """
        return self.auth_mode in ("keycloak", "api_key")


# Global instance
_api_client = None


def get_api_client() -> APIClient:
    """Get or create global API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
