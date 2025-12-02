"""
Configuration settings for Code Translation API
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base URL for the inference endpoint (OpenAI style)
# Example for your IBM endpoint:
#   BASE_URL = "https://inference-on-ibm.edgecollaborate.com"
BASE_URL = os.getenv("BASE_URL", "")

# Optional Keycloak configuration
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "master")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "api")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")

# Optional direct API key (OpenAI style)
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")

# Model configuration
# Endpoint suffix that will be appended in the API client
# For your IBM example this could be "v1/chat/completions"
INFERENCE_MODEL_ENDPOINT = os.getenv(
    "INFERENCE_MODEL_ENDPOINT",
    "v1/chat/completions",
)

# The model name passed in the OpenAI-style request
INFERENCE_MODEL_NAME = os.getenv(
    "INFERENCE_MODEL_NAME",
    "codellama/CodeLlama-34b-Instruct-hf",
)

# Application settings
APP_TITLE = "Code Translation API"
APP_DESCRIPTION = "AI-powered code translation service using CodeLlama-34b-instruct"
APP_VERSION = "1.0.0"

# File upload settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
ALLOWED_EXTENSIONS = {".pdf"}

# Code translation settings
SUPPORTED_LANGUAGES = ["java", "c", "cpp", "python", "rust", "go"]
MAX_CODE_LENGTH = int(os.getenv("MAX_CODE_LENGTH", "10000"))  # characters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# CORS settings
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
