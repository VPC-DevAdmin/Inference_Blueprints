# app/core/tts_client.py

import logging
from typing import Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class TTSClient:
    """
    Client for an OpenAI compatible TTS endpoint behind TTS_BASE_URL.
    """

    def __init__(self, base_url: Optional[str] = None, model: str = "tts-1-hd"):
        raw_base = base_url or settings.TTS_BASE_URL
        self.base_url: Optional[str] = raw_base.rstrip("/") if raw_base else None
        self.model = model or settings.TTS_MODEL
        self.auth_mode: str = "none"
        self.api_key: Optional[str] = None

        if not self.base_url:
            logger.error("TTSClient: TTS_BASE_URL is not set.")
            return

        # Try Keycloak first
        token = self._try_keycloak_token()
        if token:
            self.api_key = token
            self.auth_mode = "keycloak"
            logger.info("TTSClient: using Keycloak token")

        # Fall back to static key
        if not self.api_key and settings.TTS_API_KEY:
            self.api_key = settings.TTS_API_KEY
            self.auth_mode = "tts_api_key"
            logger.info("TTSClient: using TTS_API_KEY")

        if not self.api_key:
            logger.warning(
                "TTSClient: no auth configured. Calls will be unauthenticated."
            )

    def _tts_v1_base(self) -> Optional[str]:
        if not self.base_url:
            return None
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    def _build_headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _try_keycloak_token(self) -> Optional[str]:
        """
        Obtain an access token using KEYCLOAK_* if configured.
        """
        if not self.base_url:
            return None

        if not (
            settings.KEYCLOAK_CLIENT_ID
            and settings.KEYCLOAK_CLIENT_SECRET
            and settings.KEYCLOAK_AUDIENCE
        ):
            return None

        token_url = f"{self.base_url.rstrip('/')}/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": settings.KEYCLOAK_CLIENT_ID,
            "client_secret": settings.KEYCLOAK_CLIENT_SECRET,
            "audience": settings.KEYCLOAK_AUDIENCE,
        }

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(token_url, data=data)
            resp.raise_for_status()
            token = resp.json().get("access_token")
            if not token:
                logger.error("TTSClient: Keycloak response missing access_token")
                return None
            return token
        except Exception as exc:
            logger.warning("TTSClient: Keycloak token fetch failed: %s", exc)
            return None

    async def generate_speech(
        self,
        text: str,
        voice: str,
        output_path,
        format_: str = "mp3",
    ) -> None:
        """
        Call /v1/audio/speech on the configured gateway and write audio to output_path.
        """
        if not self.base_url:
            raise ValueError("TTS_BASE_URL is not configured")

        url = f"{self._tts_v1_base()}/audio/speech"

        payload = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "format": format_,
        }

        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resp.content)

    def is_available(self) -> bool:
        return self.base_url is not None
