import logging
import requests
import httpx
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class APIClient:
    """
    Auth priority:
      1) Keycloak (if BASE_URL, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET)
      2) INFERENCE_API_KEY from config
      3) Open/no-auth mode
    """

    def __init__(self) -> None:
        self.base_url = getattr(config, "BASE_URL", "").rstrip("/")
        self.token = None                  # Keycloak access token
        self.api_key = getattr(config, "INFERENCE_API_KEY", None)
        self.http_client = httpx.Client(verify=False)
        self.auth_mode = "open"            # "keycloak", "api_key", or "open"

        self._init_auth()

    def _init_auth(self) -> None:
        """Try Keycloak, then API key, then open mode."""
        # Tier 1: Keycloak
        if (
            self.base_url
            and getattr(config, "KEYCLOAK_CLIENT_ID", None)
            and getattr(config, "KEYCLOAK_CLIENT_SECRET", None)
        ):
            try:
                token_url = f"{self.base_url}/token"
                logger.info(f"Authenticating with Keycloak at {token_url}")

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
                        f"Keycloak auth failed: {response.status_code} - {response.text}"
                    )
            except Exception as exc:
                logger.warning(f"Keycloak auth error: {exc}")

        # Tier 2: direct inference API key
        if self.api_key:
            self.auth_mode = "api_key"
            logger.info("Using INFERENCE_API_KEY for authentication")
            return

        # Tier 3: open mode
        self.auth_mode = "open"
        logger.info("No authentication configured, using open mode")

    def get_inference_client(self) -> OpenAI:
        """
        Create an OpenAI-compatible client.

        URL structure:
          BASE_URL/v1/chat/completions

        The *model* is passed inside the request body and does NOT appear in the URL.
        """
        if not self.base_url:
            raise ValueError("BASE_URL must be set in environment or config")

        # Pick the correct key based on auth tier
        if self.auth_mode == "keycloak" and self.token:
            key = self.token
        elif self.auth_mode == "api_key" and self.api_key:
            key = self.api_key
        else:
            # Open mode still requires a value to satisfy OpenAI client
            key = "no-auth"

        # Correct OpenAI-compatible API path
        api_base = f"{self.base_url}/v1"

        return OpenAI(
            api_key=key,
            base_url=api_base,
            http_client=self.http_client,
        )

    def is_authenticated(self) -> bool:
        """Authenticated means keycloak-token or api-key."""
        return self.auth_mode in ("keycloak", "api_key")


# Global shared instance
_api_client = None

def get_api_client() -> APIClient:
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
