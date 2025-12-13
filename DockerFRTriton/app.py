import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from pipeline import calculate_face_similarity, get_embedding
from triton_service import (
    TRITON_HTTP_PORT,
    create_triton_client,
    prepare_model_repository,
    start_triton_server,
    stop_triton_server,
)

MODEL_REPO = Path(__file__).parent / "model_repository"

app = FastAPI(
    title="FR Triton API",
    description="Minimal FastAPI wrapper around Triton Inference Server for FR embeddings.",
    version="0.1.0",
)

_server_handle: Optional[Any] = None
_triton_client: Optional[Any] = None
logger = logging.getLogger("fr_triton_app")


@app.on_event("startup")
def startup_event() -> None:
    """
    Prepare the Triton model repo, launch the server, and create an HTTP client.
    This is a reference implementation for students; adjust paths as needed.
    """
    global _server_handle, _triton_client
    if os.getenv("SKIP_TRITON"):
        logger.warning("SKIP_TRITON is set; FastAPI will run without Triton. Endpoints will return 503.")
        return

    try:
        prepare_model_repository(MODEL_REPO)
        _server_handle = start_triton_server(MODEL_REPO)
        _triton_client = create_triton_client(f"localhost:{TRITON_HTTP_PORT}")
    except FileNotFoundError as exc:
        logger.error("Model repository missing required ONNX/model files: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to start Triton: %s", exc)


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Stop the Triton server when the FastAPI app shuts down."""
    global _server_handle
    stop_triton_server(_server_handle)
    _server_handle = None


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embedding", tags=["Face Recognition"])
async def embedding(image: UploadFile = File(..., description="Face image to embed")) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    content = await image.read()
    try:
        embedding_arr = get_embedding(_triton_client, content)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    embedding_list = embedding_arr.reshape(embedding_arr.shape[0], -1).tolist()
    return {"embedding": embedding_list}


@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image (aligned to model input size)"),
    image_b: UploadFile = File(..., description="Second face image (aligned to model input size)"),
) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    content_a, content_b = await image_a.read(), await image_b.read()
    try:
        score = calculate_face_similarity(_triton_client, content_a, content_b)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Similarity failed: {exc}") from exc

    return {"similarity": score}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5004, reload=False)
