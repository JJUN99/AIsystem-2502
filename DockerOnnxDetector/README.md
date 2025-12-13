# DockerOnnxDetector

Export the RetinaFace MobileNet0.25 detector to ONNX. The script pulls the pretrained `.pth` weights (about 2 MB) and writes an ONNX graph with the classification, box regression, and landmark heads. Non‑maximum suppression is intentionally left blank in `nms.py` for you to implement later.

## Setup

```bash
cd DockerOnnxDetector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the export

```bash
python export_retinaface_to_onnx.py \
  --weights-path weights/mobilenet0.25_Final.pth \
  --onnx-path artifacts/retinaface_mnet025.onnx \
  --image-size 640 \
  --opset 12
```

- The weight file downloads automatically if missing.
- The exported ONNX has dynamic batch size for the input and raw head outputs; postprocess and NMS are not part of the graph.
- Update `--image-size` if you need a different static resolution for the export.
