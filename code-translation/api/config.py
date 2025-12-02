"""
Configuration settings for Code Translation Application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


# ============================================================================
# Core API and Authentication
# ============================================================================

# Base URL of the inference server
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")

# Enterprise / Keycloak configuration (optional)
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "master")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")  # optional
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")  # optional

# Direct API key support for OpenAI compatible endpoints
# Example: INFERENCE_API_KEY="sk-..."
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")

# Model configuration
# This is passed to the chat completion call as the "model" field
INFERENCE_MODEL_NAME = os.getenv(
    "INFERENCE_MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
)

# LLM generation settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))


# ============================================================================
# Application metadata
# ============================================================================

APP_TITLE = "Code Translation Service"
APP_DESCRIPTION = (
    "AI powered code translation service that converts code between "
    "languages using an OpenAI compatible inference endpoint."
)
APP_VERSION = "1.0.0"


# ============================================================================
# Code and file limits
# ============================================================================

# Maximum length of source code accepted by the API
MAX_CODE_LENGTH = int(os.getenv("MAX_CODE_LENGTH", "10000"))

# Maximum upload size for PDF files (in bytes)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB

# Supported programming languages for translation
SUPPORTED_LANGUAGES = [
    "python",
    "java",
    "c",
    "cpp",
    "go",
    "rust",
]


# ============================================================================
# CORS configuration
# ============================================================================

# CORS settings used by FastAPI middleware
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
