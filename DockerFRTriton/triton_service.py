import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch
from itertools import product as product
from math import ceil
from PIL import Image

# ==========================================
# 1. 서버 설정 및 포트
# ==========================================
TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

# ==========================================
# 2. 모델 설정 상수
# ==========================================
DETECTION_MODEL_NAME = 'retinaface_mnet'
DETECTION_MODEL_VERSION = '1'
DETECTION_MODEL_INPUT_NAME = 'input'
DETECTION_OUTPUT_NAMES = ["loc", "conf", "landms"]
DETECTION_INPUT_SIZE = (640, 640) 

FR_MODEL_NAME = "fr_model"
FR_MODEL_VERSION = "1"
FR_MODEL_INPUT_NAME = "input"
FR_MODEL_OUTPUT_NAME = "embedding"
FR_MODEL_IMAGE_SIZE = (160, 160)

ANTISPOOF_MODEL_NAME = 'spoofnet'
ANTISPOOF_MODEL_VERSION = "1"
ANTISPOOF_MODEL_INPUT_NAME = 'input'
ANTISPOOF_MODEL_OUTPUT_NAME = 'x'

# RetinaFace Anchor 설정
cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
}

# ==========================================
# 3. Helper Functions (PriorBox, Decode, NMS)
# ==========================================
def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]; y1 = dets[:, 1]; x2 = dets[:, 2]; y2 = dets[:, 3]; scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1); order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h; ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]; order = order[inds + 1]
    return keep

class PriorBox:
    def __init__(self, cfg, image_size=None):
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]; s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx): anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip: output.clamp_(max=1, min=0)
        return output

def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2; boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), dim=1)
    return landms

# ==========================================
# 4. Triton Server Management
# ==========================================
def prepare_model_repository(model_repo: Path) -> None:
    detector_dir = model_repo / DETECTION_MODEL_NAME / DETECTION_MODEL_VERSION
    detector_path = detector_dir / "retinaface_mnet025.onnx"
    fr_dir = model_repo / FR_MODEL_NAME / FR_MODEL_VERSION
    fr_path = fr_dir / "model.onnx"
    antispoof_dir = model_repo / ANTISPOOF_MODEL_NAME / ANTISPOOF_MODEL_VERSION
    antispoof_path = antispoof_dir / "model.onnx"

    detector_config_path = detector_dir.parent / "config.pbtxt"
    fr_config_path = fr_dir.parent / "config.pbtxt"
    antispoof_config_path = antispoof_dir.parent / "config.pbtxt"

    # 파일 존재 확인 (생략 가능하나 안전을 위해 유지)
    if not detector_path.exists():
        raise FileNotFoundError(f"Missing ONNX model at {detector_path}")
    if not fr_path.exists():
        raise FileNotFoundError(f"Missing ONNX model at {fr_path}")
    if not antispoof_path.exists():
        raise FileNotFoundError(f"Missing ONNX model at {antispoof_path}")

    # 1. Detection Config (640x640 고정 + 앵커 16800개 고정)
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_config_text = textwrap.dedent(
        f"""
        name: "{DETECTION_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "retinaface_mnet025.onnx"
        input [
          {{
            name: "{DETECTION_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, 640, 640]
          }}
        ]
        output [
          {{ name: "loc", data_type: TYPE_FP32, dims: [16800, 4] }},
          {{ name: "conf", data_type: TYPE_FP32, dims: [16800, 2] }},
          {{ name: "landms", data_type: TYPE_FP32, dims: [16800, 10] }}
        ]
        instance_group [ {{ kind: KIND_CPU }} ]
        """
    ).strip() + "\n"
    detector_config_path.write_text(detector_config_text)

    # 2. FR Config (160x160)
    fr_dir.mkdir(parents=True, exist_ok=True)
    fr_config_text = textwrap.dedent(
        f"""
        name: "{FR_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{FR_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, 160, 160]
          }}
        ]
        output [
          {{ name: "{FR_MODEL_OUTPUT_NAME}", data_type: TYPE_FP32, dims: [512] }}
        ]
        instance_group [ {{ kind: KIND_CPU }} ]
        """
    ).strip() + "\n"
    fr_config_path.write_text(fr_config_text)

    # 3. AntiSpoof Config (224x224)
    antispoof_dir.mkdir(parents=True, exist_ok=True)
    antispoof_config_text = textwrap.dedent(
        f"""
        name: "{ANTISPOOF_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{ANTISPOOF_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, 224, 224]
          }}
        ]
        output [
          {{ name: "{ANTISPOOF_MODEL_OUTPUT_NAME}", data_type: TYPE_FP32, dims: [1] }}
        ]
        instance_group [ {{ kind: KIND_CPU }} ]
        """
    ).strip() + "\n"
    antispoof_config_path.write_text(antispoof_config_text)
    print(f"[triton] All model configurations prepared.")

def start_triton_server(model_repo: Path) -> Any:
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin: raise RuntimeError("Could not find `tritonserver` binary.")
    cmd = [
        triton_bin, f"--model-repository={model_repo}", f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}", f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true", "--log-verbose=0"
    ]
    process = subprocess.Popen(cmd, stdout=None, stderr=None, text=True)
    time.sleep(3)
    return process

def stop_triton_server(server_handle: Any) -> None:
    if server_handle: server_handle.terminate()

def create_triton_client(url: str) -> Any:
    from tritonclient import http as httpclient
    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live(): raise RuntimeError(f"Triton server at {url} is not live.")
    return client

# ==========================================
# 5. Inference Logic (Safe Squeeze & Resize)
# ==========================================
def run_detection(client: Any, image_bytes: bytes) -> Any:
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("Dependencies missing") from exc

    # 1. 이미지 로드
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        np_img_raw = np.asarray(img) # RGB
        np_img_bgr = np_img_raw[:, :, ::-1] # BGR

    orig_h, orig_w = np_img_bgr.shape[:2]

    # 2. 640x640 리사이즈 (단순 Stretching)
    target_size = DETECTION_INPUT_SIZE[0] # 640
    scale_x = orig_w / target_size
    scale_y = orig_h / target_size

    np_img_resized = cv2.resize(np_img_bgr, (target_size, target_size))

    # 3. 전처리 (Mean Subtraction)
    np_img_float = np_img_resized.astype(np.float32) - np.array([104, 117, 123], dtype=np.float32)
    np_img_chw = np.transpose(np_img_float, (2, 0, 1))
    batch = np.expand_dims(np_img_chw, axis=0).astype(np.float32)

    # 4. 추론
    infer_input = httpclient.InferInput(DETECTION_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_outputs = [httpclient.InferRequestedOutput(name) for name in DETECTION_OUTPUT_NAMES]
    
    response = client.infer(model_name=DETECTION_MODEL_NAME, inputs=[infer_input], outputs=infer_outputs)

    # 5. 결과 파싱 및 안전한 Squeeze (차원 에러 방지)
    loc = torch.from_numpy(response.as_numpy("loc"))     
    conf = torch.from_numpy(response.as_numpy("conf"))   
    landms = torch.from_numpy(response.as_numpy("landms"))

    # 🚨 [핵심 수정] 3차원(배치 차원 포함)일 때만 squeeze 수행
    if loc.dim() == 3: loc = loc.squeeze(0)
    if conf.dim() == 3: conf = conf.squeeze(0)
    if landms.dim() == 3: landms = landms.squeeze(0)

    # 6. Decode
    priorbox = PriorBox(cfg_mnet, image_size=(target_size, target_size))
    priors = priorbox.forward()
    
    scale_box = torch.Tensor([target_size, target_size, target_size, target_size])
    scale_landms = torch.Tensor([target_size, target_size] * 5)

    # 🚨 .squeeze(0) 제거됨 (위에서 처리)
    boxes = decode(loc.data, priors.data, cfg_mnet['variance']) * scale_box
    scores = conf[:, 1] 
    landmarks = decode_landm(landms.data, priors.data, cfg_mnet['variance']) * scale_landms

    # 7. Filter & NMS
    inds = np.where(scores > 0.6)[0]
    if len(inds) == 0:
        raise RuntimeError(f"No face detected. (Max score: {scores.max().item():.4f})")
    
    dets = np.hstack((boxes[inds].numpy(), scores[inds].numpy()[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(dets, 0.4)
    
    best_det_idx = keep[0]
    best_orig_idx = inds[best_det_idx]
    
    best_box = boxes[best_orig_idx].numpy()
    best_score = scores[best_orig_idx].item()
    best_landmark = landmarks[best_orig_idx].numpy().reshape(5, 2)
    
    # 8. 좌표 복원 (640 -> 원본)
    best_box[[0, 2]] *= scale_x
    best_box[[1, 3]] *= scale_y
    best_landmark[:, 0] *= scale_x
    best_landmark[:, 1] *= scale_y
    
    return {
        "boxes": np.expand_dims(best_box, axis=0),
        "scores": np.array([best_score]),
        "landmarks": np.expand_dims(best_landmark.flatten(), axis=0)
    }

def run_face_recognition(client: Any, aligned_face_rgb: np.ndarray) -> np.ndarray:
    try:
        from tritonclient import http as httpclient
    except ImportError:
        raise RuntimeError("tritonclient required")

    # 160x160 이미지 전처리 (0~1, Normalize -1~1)
    np_img_float = aligned_face_rgb.astype(np.float32) / 255.0
    np_img_norm = (np_img_float - 0.5) / 0.5
    np_img_chw = np.transpose(np_img_norm, (2, 0, 1))
    batch = np.expand_dims(np_img_chw, axis=0).astype(np.float32)

    infer_input = httpclient.InferInput(FR_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput(FR_MODEL_OUTPUT_NAME)
    
    response = client.infer(model_name=FR_MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(FR_MODEL_OUTPUT_NAME).flatten()

def run_face_antispoof(client: Any, aligned_face_rgb: np.ndarray):
    try:
        from tritonclient import http as httpclient
    except ImportError:
        raise RuntimeError("tritonclient required")
    
    img_resized = cv2.resize(aligned_face_rgb, (224, 224))
    np_img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np_img_norm = (np_img_float - mean) / std

    np_img_chw = np.transpose(np_img_norm, (2, 0, 1))
    batch = np.expand_dims(np_img_chw, axis=0).astype(np.float32)

    infer_input = httpclient.InferInput(ANTISPOOF_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput(ANTISPOOF_MODEL_OUTPUT_NAME)
    
    response = client.infer(model_name=ANTISPOOF_MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(ANTISPOOF_MODEL_OUTPUT_NAME)