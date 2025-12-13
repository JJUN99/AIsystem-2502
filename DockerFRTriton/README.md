# DockerFRTriton

Serve a face-recognition (FR) system on Triton Inference Server (CPU) with a FastAPI wrapper. Students must export models to ONNX, build a Triton-ready model repository, and keep all inference on Triton (FR backbone, detector, and any future alignment/spoofing).

## What you must implement
- Export your FR backbone to ONNX for CPU in `convert_to_onnx.py` (replace the toy model, load your weights, keep dynamic batch).
- Export your face detector to ONNX (example script in `DockerOnnxDetector/export_retinaface_to_onnx.py`).
- Ensure the Triton model repository is correct and configs match your input/output shapes and names.
- Extend `triton_service.py` preprocessing or shapes if your model differs.
- Extend `pipeline.py` to call detector/alignment/antispoof models (all should live in the Triton repo).
- Keep FastAPI (`app.py`) thin; it should just call Triton endpoints via the client.

## Model repository layout
```
DockerFRTriton/
└── model_repository/
    ├── fr_model/
    │   ├── 1/
    │   │   └── model.onnx
    │   └── config.pbtxt
    └── face_detector/
        ├── 1/
        │   └── model.onnx
        └── config.pbtxt
```
Use the same root path for export, config generation, and Triton startup.

## Export models locally
```bash
# Export your FR model to ONNX
python convert_to_onnx.py \
  --weights-path weights/your_fr_model.pth \
  --onnx-path model_repository/fr_model/1/model.onnx

# (Optional) Export detector to ONNX via DockerOnnxDetector
python ../DockerOnnxDetector/export_retinaface_to_onnx.py \
  --weights-path ../DockerOnnxDetector/weights/mobilenet0.25_Final.pth \
  --onnx-path model_repository/face_detector/1/model.onnx
```

## Run with Docker + FastAPI (all-in-one container)
- Build image (uses `model_repository/*`):
  ```bash
  docker build -t fr-triton -f Docker/Dockerfile .
  ```
- Run container (Triton on 8000/8001/8002, FastAPI on 3000):
  ```bash
  docker run --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -p 3000:3000 \
    --name fr_triton \
    fr-triton
  ```
- Open Swagger UI at:
  ```
  http://0.0.0.0:3000
  ```
  (FastAPI inside the container proxies to Triton on localhost:8000.)

If `model_repository/fr_model/1/model.onnx` is missing, the container still starts and serves Swagger, but Triton-dependent endpoints will return 503 until you add the models. For local (non-Docker) development, you can run `python run_fastapi.py` to create a venv, install deps, and start Uvicorn.

## Key files
- `convert_to_onnx.py`: Replace `ToyFRModel`, load your weights, export ONNX (dynamic batch). Adjust input size/names if needed.
- `triton_service.py`: Prepares `config.pbtxt`, starts/stops Triton, creates HTTP client, preprocesses inputs, and calls `infer`. Update shapes/names to match your models.
- `pipeline.py`: Currently calls the FR model twice and returns cosine similarity. Extend to call detector/alignment/antispoof models once they are in `model_repository`.
- `app.py`: FastAPI wrapper exposing `/embedding`, `/face-similarity`, `/health`. Keep business logic inside Triton via `pipeline.py`.

## Notes
- Always align the config (`config.pbtxt`) with the ONNX model’s input/output names, shapes, and types.
- All models (FR, detector, etc.) should be served by Triton; FastAPI should not run heavy model code locally.
- If you change the model repository root, pass the same Path to `prepare_model_repository` and `start_triton_server`.
- **Submission reminder:** include your repository URL/ID and a port-forwarded Swagger URL for the running API when you submit.
