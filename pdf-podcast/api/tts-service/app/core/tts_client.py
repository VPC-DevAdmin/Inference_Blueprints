import logging
from pathlib import Path
from typing import Optional
import asyncio

import requests
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class TTSClient:
    """
    Client for an OpenAI compatible Text to Speech API.

    Works with any gateway that implements /v1/audio/speech.
    Auth priority:
      1) Keycloak (using TTS_BASE_URL plus a token endpoint)
      2) TTS_API_KEY
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.model = model or settings.TTS_MODEL
        self.auth_mode: str = "none"
        self.client: Optional[OpenAI] = None

        # Resolve base URL
        self.base_url = base_url or settings.TTS_BASE_URL
        if not self.base_url:
            logger.error("TTSClient: TTS_BASE_URL is not set, gateway is required")
            return

        base = self.base_url.rstrip("/")
        # Normalize so we end up with a single /v1
        if base.endswith("/v1"):
            final_base = base
        else:
            final_base = f"{base}/v1"

        # Resolve auth
        self.api_key = None

        # 1) Try Keycloak if client id and secret are present
        if settings.KEYCLOAK_CLIENT_ID and settings.KEYCLOAK_CLIENT_SECRET:
            token = self._try_keycloak_token(base)
            if token:
                self.api_key = token
                self.auth_mode = "keycloak"
                logger.info("TTSClient: using Keycloak access token")
            else:
                logger.warning("TTSClient: Keycloak auth failed, will try TTS_API_KEY")

        # 2) Fall back to direct TTS_API_KEY
        if not self.api_key and (api_key or settings.TTS_API_KEY):
            self.api_key = api_key or settings.TTS_API_KEY
            self.auth_mode = "tts_api_key"
            logger.info("TTSClient: using TTS_API_KEY")

        if not self.api_key:
            logger.error(
                "TTSClient: no auth configured, set either Keycloak credentials "
                "or TTS_API_KEY"
            )
            return

        # Create OpenAI style client pointed at the gateway
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=final_base,
        )

        logger.info(
            "TTSClient: configured with base_url=%s, model=%s, auth_mode=%s",
            final_base,
            self.model,
            self.auth_mode,
        )

    def _try_keycloak_token(self, base: str) -> Optional[str]:
        """
        Try to obtain an access token using client credentials.

        This assumes your gateway exposes a token endpoint under the same base,
        for example: {TTS_BASE_URL}/token

        If this pattern is different in your environment, adjust this function.
        """
        token_url = f"{base.rstrip('/')}/token"
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
                logger.error("TTSClient: Keycloak response missing access_token")
            else:
                logger.error(
                    "TTSClient: Keycloak token request failed %s: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as e:
            logger.error("TTSClient: Keycloak token request error: %s", str(e))

        return None

    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        speed: float = 1.0,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """
        Generate speech audio from text using /v1/audio/speech.
        """
        if not self.client:
            raise RuntimeError(
                "TTSClient is not configured, check TTS_BASE_URL and auth settings"
            )

        try:
            logger.info(
                "Generating speech: voice=%s, length=%d chars",
                voice,
                len(text),
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.audio.speech.create(
                    model=self.model,
                    voice=voice,
                    input=text,
                    speed=speed,
                ),
            )

            audio_bytes = response.content

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                logger.info("Saved audio to %s", output_path)

            logger.info("Generated %d bytes of audio", len(audio_bytes))
            return audio_bytes

        except Exception as e:
            logger.error("Speech generation failed: %s", str(e))
            raise

    async def generate_speech_batch(
        self,
        texts: list[str],
        voices: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate speech for multiple texts in parallel.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        output_paths = []

        for i, (text, voice) in enumerate(zip(texts, voices)):
            output_path = output_dir / f"segment_{i:03d}.mp3"
            output_paths.append(output_path)

            task = self.generate_speech(
                text=text,
                voice=voice,
                output_path=output_path,
            )
            tasks.append(task)

        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

        async def bounded(task):
            async with semaphore:
                return await task

        await asyncio.gather(*[bounded(t) for t in tasks])

        logger.info("Generated %d audio segments", len(output_paths))
        return output_paths

    def get_available_voices(self) -> list[str]:
        """Get list of available voices."""
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def is_available(self) -> bool:
        """Check if TTS client has been configured."""
        return self.client is not None
