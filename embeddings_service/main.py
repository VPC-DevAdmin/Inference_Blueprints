from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local Embeddings API")

# Load BGE on CPU
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    data = []
    for idx, emb in enumerate(embeddings):
        data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": emb,
            }
        )

    # Token counts are just placeholders so the client does not crash
    usage = {
        "prompt_tokens": 0,
        "total_tokens": 0,
    }

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": usage,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

