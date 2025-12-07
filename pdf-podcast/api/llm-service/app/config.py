from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """LLM + TTS Service Configuration"""

    # Service info
    SERVICE_NAME: str = "LLM Script Generation Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_PORT: int = 8002

    # ----------------------------
    # LLM Inference Server
    # ----------------------------
    BASE_URL: Optional[str] = None                 
    INFERENCE_API_KEY: Optional[str] = None        
    KEYCLOAK_REALM: str = "master"
    KEYCLOAK_CLIENT_ID: str = "api"
    KEYCLOAK_CLIENT_SECRET: Optional[str] = None

    # ----------------------------
    # Model Configuration
    # ----------------------------
    INFERENCE_MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # Script generation defaults
    DEFAULT_MODEL: str = "gpt-4o-mini"             
    DEFAULT_TONE: str = "conversational"
    DEFAULT_MAX_LENGTH: int = 2000

    # Generation parameters
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4000
    MAX_RETRIES: int = 3

    # ----------------------------
    # TTS Server (OpenAI-compatible)
    # Defaults work out-of-the-box with OpenAI
    # but users can override both via docker-compose.
    # ----------------------------
    TTS_BASE_URL: str = "https://api.openai.com/v1"
    TTS_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
