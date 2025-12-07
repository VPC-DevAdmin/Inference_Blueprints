from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from app.api.routes import router
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Script Generation Service",
    description="Generate podcast scripts from PDF text using a local OpenAI compatible LLM endpoint",
    version=settings.SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, tags=["Script Generation"])


@app.on_event("startup")
async def startup_event():
    logger.info("============================================================")
    logger.info("LLM Script Generation Service starting up...")
    logger.info("============================================================")
    logger.info(f"Service running on port {settings.SERVICE_PORT}")

    if settings.BASE_URL:
        logger.info(f"LLM base URL: {settings.BASE_URL}")
    else:
        logger.warning(
            "BASE_URL is not set. LLM service will not be able to reach the local gateway."
        )

    # Auth mode is handled inside LLMClient, but we can hint here
    if getattr(settings, "KEYCLOAK_CLIENT_SECRET", None):
        logger.info("Auth: Keycloak client credentials configured")
    elif getattr(settings, "INFERENCE_API_KEY", None):
        logger.info("Auth: INFERENCE_API_KEY configured")
    else:
        logger.warning(
            "No LLM auth configured. Set either KEYCLOAK_CLIENT_SECRET or INFERENCE_API_KEY."
        )

    logger.info(f"Default model: {settings.INFERENCE_MODEL_NAME}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("LLM Script Generation Service shutting down...")


@app.get("/")
async def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "description": "Generate podcast scripts from PDF text using a local OpenAI compatible LLM endpoint",
        "endpoints": {
            "generate": "POST /generate-script",
            "refine": "POST /refine-script",
            "validate": "POST /validate-content",
            "health": "GET /health",
            "tones": "GET /tones",
            "models": "GET /models",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
