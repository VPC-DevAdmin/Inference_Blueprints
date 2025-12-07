from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import sys
import os

from app.api.routes import router
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TTS Audio Generation Service",
    description="Generate podcast audio from scripts using OpenAI TTS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(settings.OUTPUT_DIR)
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(router, tags=["Audio Generation"])

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("TTS Audio Generation Service starting up...")
    logger.info("Service running on port %s", getattr(settings, "SERVICE_PORT", 8003))

    # Report TTS gateway configuration
    if getattr(settings, "TTS_BASE_URL", None):
        auth_modes = []

        # Shared Keycloak config (same pattern as LLM service)
        if getattr(settings, "KEYCLOAK_CLIENT_ID", None) and getattr(
            settings, "KEYCLOAK_CLIENT_SECRET", None
        ):
            auth_modes.append("Keycloak")

        if getattr(settings, "TTS_API_KEY", None):
            auth_modes.append("static API key")

        if auth_modes:
            auth_desc = " + ".join(auth_modes)
        else:
            auth_desc = "no auth"

        logger.info(
            "TTS gateway configured at %s (%s)",
            settings.TTS_BASE_URL,
            auth_desc,
        )
    else:
        logger.warning(
            "TTS_BASE_URL is not set. TTSClient will not be able to reach a gateway "
            "until this is configured."
        )

    # Create output directory
    static_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", static_dir.absolute())


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("TTS Audio Generation Service shutting down...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "TTS Audio Generation Service",
        "version": "1.0.0",
        "description": "Generate podcast audio from scripts using OpenAI TTS",
        "endpoints": {
            "generate_audio": "POST /generate-audio - Generate podcast audio",
            "status": "GET /status/{job_id} - Check generation status",
            "download": "GET /download/{job_id} - Download audio file",
            "voices": "GET /voices - List available voices",
            "voice_sample": "GET /voice-sample/{voice_id} - Get voice sample",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
