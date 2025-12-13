from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import torch
from util import calculate_face_similarity

app = FastAPI(
    title="My FastAPI Service",
    description="A simple demo API running in Docker. Swagger is at /docs and ReDoc at /redoc.",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI URL
    redoc_url="/redoc",        # ReDoc URL
    openapi_url="/openapi.json" # OpenAPI JSON spec URL
)

class Echo(BaseModel):
    text: str


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello, FastAPI in Docker!"}

@app.get("/torch-version", tags=["info"])
def torch_version():
    return {"torch_version": torch.__version__}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/echo", tags=["Demo"])
def echo(body: Echo):
    return {"you_sent": body.text}


@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image file"),
    image_b: UploadFile = File(..., description="Second face image file"),
):
    """
    Compare two face images and return a similarity score between them.

    The implementation is left to students; the current version raises 501
    until the underlying utilities are completed.
    """
    try:
        content_a = await image_a.read()
        content_b = await image_b.read()
        similarity = calculate_face_similarity(content_a, content_b)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return {"similarity": similarity}
