import os
from typing import List, Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "BAAI/bge-base-en-v1.5")

app = FastAPI(title="Local Embeddings Service")

# Load model once at startup
model = SentenceTransformer(MODEL_NAME)


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    # Allow override of model name in request, but ignore for now
    texts: List[str]
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input

    # sentence-transformers returns a numpy array when convert_to_numpy=True
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    data = []
    for idx, emb in enumerate(embeddings):
        data.append(
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx,
            }
        )

    # Very rough token usage estimate, just to satisfy clients that expect it
    total_tokens = sum(len(t.split()) for t in texts)

    return {
        "object": "list",
        "data": data,
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }
