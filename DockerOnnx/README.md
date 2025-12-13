# DockerOnnx

Example for exporting a PyTorch MobileNetV2 model to ONNX. The script downloads the pretrained `.pth` weights (about 14 MB) and writes an ONNX file with dynamic batch axes.

## Setup

```bash
cd DockerOnnx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the export

```bash
python export_mobilenet_to_onnx.py \
  --weights-path weights/mobilenet_v2-b0353104.pth \
  --onnx-path artifacts/mobilenet_v2.onnx \
  --opset 12
```

- Weights are downloaded only if the file is missing.
- ONNX is validated after export; look for `ONNX export completed`.
- Adjust `--opset` if you need a different opset version.
