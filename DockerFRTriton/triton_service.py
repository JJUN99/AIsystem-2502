import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

import torch
from itertools import product as product
from math import ceil
from PIL import Image

TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

DETECTION_MODEL_NAME = 'retinaface_mnet'
DETECTION_MODEL_VERSION = '1'
DETECTION_MODEL_INPUT_NAME = 'input'
DETECTION_OUTPUT_NAMES = ["loc", "conf", "landms"]

FR_MODEL_NAME = "fr_model"
FR_MODEL_VERSION = "1"
FR_MODEL_INPUT_NAME = "input"
FR_MODEL_OUTPUT_NAME = "embedding"
FR_MODEL_IMAGE_SIZE = (112, 112)

ANTISPOOF_MODEL_NAME = 'spoofnet'
ANTISPOOF_MODEL_VERSION = "1"
ANTISPOOF_MODEL_INPUT_NAME = 'input'
ANTISPOOF_MODEL_OUTPUT_NAME = 'x'

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
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

# RetinaFace 기본 설정 (MobileNet0.25 기준)
cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
}

def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the FR ONNX model and config.pbtxt.
    """
    detector_dir = model_repo / DETECTION_MODEL_NAME / DETECTION_MODEL_VERSION
    detector_path = detector_dir / "retinaface_mnet025.onnx"

    fr_dir = model_repo / FR_MODEL_NAME / FR_MODEL_VERSION
    fr_path = fr_dir / "model.onnx"

    antispoof_dir = model_repo / ANTISPOOF_MODEL_NAME / ANTISPOOF_MODEL_VERSION
    antispoof_path = antispoof_dir / "model.onnx"

    detector_config_path = detector_dir.parent / "config.pbtxt"
    fr_config_path = fr_dir.parent / "config.pbtxt"
    antispoof_config_path = antispoof_dir.parent / "config.pbtxt"

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {detector_path}. "
            "Run export_retinaface_to_onnx.py first or place your exported model there."
        )

    if not fr_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {fr_path}. "
            "Run convert_to_onnx.py first or place your exported model there."
        )
    
    if not antispoof_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {antispoof_path}. "
            "Run export_spoofnet_to_onnx.py first or place your exported model there."
        )

    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_config_text = textwrap.dedent(
        f"""
        name: "{DETECTION_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        default_model_filename: "retinaface_mnet025.onnx"
        input [
          {{
            name: "{DETECTION_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 3, 640, 640]
          }}
        ]
        output [
          {{
            name: "loc" # BboxHead의 출력
            data_type: TYPE_FP32
            dims: [1,16800,4] 
            }},
            {{
            name: "conf" # ClassHead의 출력
            data_type: TYPE_FP32
            dims: [1,16800,2] 
            }},
            {{
            name: "landms" # LandmarkHead의 출력
            data_type: TYPE_FP32
            dims: [1,16800,10] 
           }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    detector_config_path.write_text(detector_config_text)
    print(f"[triton] Prepared detector repository at {detector_dir.parent}")

    fr_dir.mkdir(parents=True, exist_ok=True)
    fr_config_text = textwrap.dedent(
        f"""
        name: "{FR_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{FR_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 3, 112,112]
          }}
        ]
        output [
          {{
            name: "{FR_MODEL_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 512] 
            }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    fr_config_path.write_text(fr_config_text)
    print(f"[triton] Prepared FR model repository at {fr_dir.parent}")

    antispoof_dir.mkdir(parents=True, exist_ok=True)
    antispoof_config_text = textwrap.dedent(
        f"""
        name: "{ANTISPOOF_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{ANTISPOOF_MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1, 3, 224, 224]
          }}
        ]
        output [
          {{
            name: "{ANTISPOOF_MODEL_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [1,1] 
            }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    antispoof_config_path.write_text(antispoof_config_text)
    print(f"[triton] Prepared Antispoof model repository at {antispoof_dir.parent}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process = subprocess.Popen(cmd, stdout=None, stderr=None, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def preprocess_image_letterbox(image_pil: Image.Image, target_size=(640, 640)):
    """
    이미지 비율을 유지하면서 target_size로 맞춥니다. (남는 공간은 검은색 padding)
    """
    iw, ih = image_pil.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_pil = image_pil.resize((nw, nh), Image.BILINEAR)
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    new_image.paste(image_pil, (0, 0))
    
    return new_image, scale

def run_detection(client: Any, image_bytes: bytes) -> Any:
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("Dependencies missing") from exc

    # 1. 이미지 로드
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB") # RGBA -> RGB 필수 변환
        
        img_resized, scale = preprocess_image_letterbox(img, target_size=(640, 640))
        
        np_img = np.asarray(img_resized, dtype=np.float32)
        
        # [RGB -> BGR 변환]
        # RetinaFace는 OpenCV(BGR) 기준으로 학습되었습니다. PIL은 RGB입니다.
        # 이 줄이 없으면 색상이 반전되어 얼굴 인식이 안 됩니다.
        np_img = np_img[:, :, ::-1] 

    # 4. Mean Subtraction (BGR 기준)
    np_img -= np.array([104, 117, 123], dtype=np.float32)
    
    np_img = np.transpose(np_img, (2, 0, 1))
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(DETECTION_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    
    infer_outputs = [httpclient.InferRequestedOutput(name) for name in DETECTION_OUTPUT_NAMES]
    response = client.infer(model_name=DETECTION_MODEL_NAME, inputs=[infer_input], outputs=infer_outputs)

    
    loc = torch.from_numpy(response.as_numpy("loc"))     
    conf = torch.from_numpy(response.as_numpy("conf"))   
    landms = torch.from_numpy(response.as_numpy("landms"))

    priorbox = PriorBox(cfg_mnet, image_size=(640, 640))
    priors = priorbox.forward()
    prior_data = priors.data
    
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * 640 
    
    scores = conf.squeeze(0)[:, 1] 
    
    landmarks = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    landmarks = landmarks * 640 

    
    max_score = scores.max().item()
    print(f"[DEBUG] Max Face Score detected: {max_score:.4f}")

    
    inds = np.where(scores > 0.1)[0]
    if len(inds) == 0:
        raise RuntimeError(f"No face detected. (Max score: {max_score:.4f})")
    
    boxes = boxes[inds]
    scores = scores[inds]
    landmarks = landmarks[inds]

    
    boxes /= scale
    landmarks /= scale

    return {
        "boxes": boxes.cpu().numpy(),
        "scores": scores.cpu().numpy(),
        "landmarks": landmarks.cpu().numpy(),
    }

def run_face_recognition(client:Any, aligned_face_rgb:np.ndarray) -> np.ndarray:
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required.") from exc

    np_img_normalized = aligned_face_rgb.astype(np.float32) / 255.0
    np_img_chw = np.transpose(np_img_normalized, (2, 0, 1))
    batch = np.expand_dims(np_img_chw, axis=0)

    infer_input = httpclient.InferInput(FR_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput(FR_MODEL_OUTPUT_NAME)
    
    response = client.infer(
        model_name=FR_MODEL_NAME, 
        inputs=[infer_input], 
        outputs=[infer_output]
    )
    return response.as_numpy(FR_MODEL_OUTPUT_NAME)


def run_face_antispoof(client:Any, aligned_face_rgb:np.ndarray):
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] is required.") from exc
    
    np_img_normalized = aligned_face_rgb.astype(np.float32) / 255.0
    np_img_chw = np.transpose(np_img_normalized, (2, 0, 1))
    batch = np.expand_dims(np_img_chw, axis=0)

    infer_input = httpclient.InferInput(ANTISPOOF_MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput(ANTISPOOF_MODEL_OUTPUT_NAME)
    
    response = client.infer(
        model_name=ANTISPOOF_MODEL_NAME, 
        inputs=[infer_input], 
        outputs=[infer_output]
    )
    return response.as_numpy(ANTISPOOF_MODEL_OUTPUT_NAME)