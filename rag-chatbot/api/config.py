"""
Configuration settings for RAG Chatbot API
Gateway-centric inference via an OpenAI-compatible API surface.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------------------
# Gateway and auth configuration
# -------------------------------------------------------------------

# Base URL of your inference gateway (OpenAI-compatible API)
BASE_URL = os.getenv("BASE_URL", "https://api.example.com")

# Optional different base URL for embeddings
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", BASE_URL)

# Keycloak client-credentials (preferred if provided)
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "master")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

# Static API key fallback for the gateway
# Used when Keycloak is not configured or fails
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")

# -------------------------------------------------------------------
# Model configuration
# These are logical model names passed to the gateway
# -------------------------------------------------------------------

# Embedding model that the gateway exposes
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-base-en-v1.5")

# Chat / completion model that the gateway exposes
INFERENCE_MODEL_NAME = os.getenv(
    "INFERENCE_MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
)

# Optional legacy endpoint hints that some gateways use
# Kept only as hints for routing if you decide to re-use them
EMBEDDING_MODEL_ENDPOINT = os.getenv(
    "EMBEDDING_MODEL_ENDPOINT",
    "bge-base-en-v1.5",
)
INFERENCE_MODEL_ENDPOINT = os.getenv(
    "INFERENCE_MODEL_ENDPOINT",
    "Llama-3.1-8B-Instruct",
)

# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

if not (KEYCLOAK_CLIENT_SECRET and KEYCLOAK_CLIENT_ID) and not INFERENCE_API_KEY:
    raise ValueError(
        "RAG chatbot gateway auth is not configured. "
        "Set either KEYCLOAK_CLIENT_ID + KEYCLOAK_CLIENT_SECRET or INFERENCE_API_KEY."
    )

# -------------------------------------------------------------------
# Application Settings
# -------------------------------------------------------------------

APP_TITLE = "RAG QnA Chatbot"
APP_DESCRIPTION = (
    "A RAG-based chatbot API using a generic inference gateway (OpenAI-compatible) and FAISS"
)
APP_VERSION = "2.0.0"

# File Upload Settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
ALLOWED_EXTENSIONS = {".pdf"}

# Vector Store Settings
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./dmv_index")

# Text Splitting Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", " ", ""]

# Retrieval Settings
TOP_K_DOCUMENTS = 4

# CORS Settings
CORS_ALLOW_ORIGINS = ["*"]  # Update with specific origins in production
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
