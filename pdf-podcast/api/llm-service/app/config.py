from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """LLM + TTS Service Configuration (OpenAI compatible, not OpenAI specific)"""

    # Pydantic settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore stray OPENAI_* env vars
    )

    # Service info
    SERVICE_NAME: str = "LLM Script Generation Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_PORT: int = 8002

    # LLM inference gateway
    # Example: http://inference-gateway:8080  or  https://api.openai.com
    BASE_URL: Optional[str] = None
    INFERENCE_API_KEY: Optional[str] = None

    # Optional Keycloak for LLM
    KEYCLOAK_REALM: str = "master"
    KEYCLOAK_CLIENT_ID: str = "api"
    KEYCLOAK_CLIENT_SECRET: Optional[str] = None

    # Model configuration
    # Name as exposed by the gateway, works for local or OpenAI
    INFERENCE_MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # Script generation defaults
    # Use INFERENCE_MODEL_NAME as the logical default to avoid gpt-* leakage
    DEFAULT_MODEL: str = INFERENCE_MODEL_NAME
    DEFAULT_TONE: str = "conversational"
    DEFAULT_MAX_LENGTH: int = 2000

    # Generation parameters
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4000
    MAX_RETRIES: int = 3

    # TTS gateway - only used by tts-service or shared code if you want
    # Examples: http://tts-gateway:8080 or https://api.openai.com
    TTS_BASE_URL: Optional[str] = None
    TTS_API_KEY: Optional[str] = None


settings = Settings()
