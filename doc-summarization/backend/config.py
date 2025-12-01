"""
Configuration settings for Doc-Sum Application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Inference Gateway base URL
BASE_URL = os.getenv("BASE_URL")

# Keycloak settings (optional)
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "master")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

# OpenAI style API key (optional)
# Allows both INFERENCE_API_KEY or legacy OPENAI_API_KEY
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY") or os.getenv("OPENAI_API_KEY")

# Model configuration from the inference gateway
INFERENCE_MODEL_ENDPOINT = os.getenv("INFERENCE_MODEL_ENDPOINT", "Llama-3.1-8B-Instruct")
INFERENCE_MODEL_NAME = os.getenv(
    "INFERENCE_MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
)

# Selected model to send to the inference endpoint
# Priority:
# 1. Explicit INFERENCE_MODEL_NAME
# 2. Endpoint name (INFERENCE_MODEL_ENDPOINT)
# 3. Default fallback
SELECTED_MODEL = (
    os.getenv("INFERENCE_MODEL_NAME")
    or os.getenv("INFERENCE_MODEL_ENDPOINT")
    or "meta-llama/Llama-3.1-8B-Instruct"
)

# LLM configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# Application settings
APP_TITLE = "Document Summarization Service"
APP_DESCRIPTION = "AI powered document summarization with enterprise inference integration"
APP_VERSION = "2.0.0"

# Service configuration
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File upload limits
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(500 * 1024 * 1024)))  # 500 MB
MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE", str(50 * 1024 * 1024)))    # 50 MB

# Processing limits
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
WARN_PDF_PAGES = 50

# CORS settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
