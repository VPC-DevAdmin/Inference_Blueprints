"""
LLM Service for Document Summarization
Uses Enterprise Inference API via Keycloak, API key, or open mode
"""

from typing import Iterator, Dict, Any
import logging
import re

import config
from api_client import get_api_client

logger = logging.getLogger(__name__)


def clean_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting symbols from text.

    Args:
        text: Text that may contain markdown formatting.

    Returns:
        Clean text without markdown symbols.
    """
    # Remove bold (**text** or __text__)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)

    # Remove italic (*text* or _text_)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # Remove code blocks (```text```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code (`text`)
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove headers (# text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove bullet points and list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    return text.strip()


class LLMService:
    """
    LLM service for document summarization using enterprise inference.

    Auth priority is delegated to APIClient:
      1) Keycloak (if configured)
      2) INFERENCE_API_KEY
      3) Open mode (no real auth)
    """

    def __init__(self) -> None:
        """Initialize LLM service (authenticates on first use)."""
        self.client = None
        # Prefer SELECTED_MODEL if you added it in config.py, else fall back
        self.model = getattr(config, "SELECTED_MODEL", config.INFERENCE_MODEL_NAME)
        self._authenticated = False

    def _ensure_authenticated(self) -> None:
        """
        Initialize the OpenAI style client via APIClient.

        No longer requires KEYCLOAK_CLIENT_SECRET. Uses whatever auth
        mode APIClient has selected.
        """
        if self._authenticated:
            return

        if not config.BASE_URL:
            raise ValueError("BASE_URL must be set in environment variables for summarization")

        api_client = get_api_client()

        # This will:
        #  - use Keycloak token if available
        #  - otherwise use INFERENCE_API_KEY
        #  - otherwise use a dummy key in open mode
        self.client = api_client.get_inference_client()
        self._authenticated = True

        auth_mode = getattr(api_client, "auth_mode", "unknown")
        logger.info("LLM Service initialized")
        logger.info("Base URL: %s", config.BASE_URL)
        logger.info("Model: %s", self.model)
        logger.info("Auth mode: %s", auth_mode)

    def summarize(
        self,
        text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Summarize text using enterprise LLM.

        Args:
            text: Text to summarize.
            max_tokens: Maximum tokens in summary.
            temperature: Generation temperature.
            stream: Whether to stream response.

        Returns:
            Summary text or iterator of chunks if streaming.
        """
        # Ensure we are ready before making API calls
        self._ensure_authenticated()

        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        temperature = temperature or config.LLM_TEMPERATURE

        system_prompt = (
            "You are a professional document summarizer.\n"
            "Your task is to create clear, concise, and accurate summaries of the provided text.\n"
            "Focus on the main points and key information while maintaining the original meaning.\n\n"
            "IMPORTANT: Provide the summary in plain text format only. "
            "Do not use any markdown formatting symbols like **, *, _, or other special characters "
            "for formatting. Write in a clean, readable paragraph format."
        )

        user_prompt = f"""Please provide a comprehensive summary of the following text:

{text}

Summary:"""

        try:
            logger.info("Generating summary with %s (stream=%s)", self.model, stream)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )

            if stream:
                return self._stream_response(response)
            else:
                summary = response.choices[0].message.content
                summary = clean_markdown_formatting(summary)
                logger.info("Generated summary: %d characters", len(summary))
                return summary

        except Exception as exc:
            logger.error("LLM summarization error: %s", exc)
            raise Exception(f"Failed to generate summary: {exc}")

    def _stream_response(self, response) -> Iterator[str]:
        """Stream LLM response chunks with markdown cleaning."""
        accumulated = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                accumulated += chunk.choices[0].delta.content
                if accumulated.endswith((".", "!", "?", "\n")):
                    cleaned = clean_markdown_formatting(accumulated)
                    yield cleaned
                    accumulated = ""

        if accumulated:
            cleaned = clean_markdown_formatting(accumulated)
            yield cleaned

    def health_check(self) -> Dict[str, Any]:
        """
        Check if LLM service is healthy.

        Returns:
            Health status dictionary.
        """
        try:
            api_client = get_api_client()
            auth_mode = getattr(api_client, "auth_mode", "unknown")

            self._ensure_authenticated()

            # Simple ping
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10,
            )

            return {
                "status": "healthy",
                "provider": f"Enterprise Inference ({auth_mode})",
                "model": self.model,
                "base_url": config.BASE_URL,
            }

        except Exception as exc:
            logger.error("Health check failed: %s", exc)
            return {
                "status": "unhealthy",
                "provider": "Enterprise Inference",
                "error": str(exc),
            }


# Global LLM service instance
llm_service = LLMService()
