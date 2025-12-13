from typing import Any, Tuple

import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from triton_service import run_face_recognition, run_detection, run_face_antispoof

RETINAFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))

def _warp_face(image_rgb: Any, landmarks: Any) -> Any:
    # OpenCV는 BGR을 기본으로 한다
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    if landmarks.shape != (5, 2):
        raise ValueError("Expected 5-point landmarks")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Compute affine transform from detected landmarks to template
    src = landmarks.astype(np.float32)
    dst = RETINAFACE_TEMPLATE

    # estimateAffinePartial2D is more stable than getAffineTransform for noisy points
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Failed to estimate alignment transform")

    aligned = cv2.warpAffine(
        image_bgr,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned_rgb

def run_retina_face_and_align(client: Any, image: bytes):
    retina_face = run_detection(client, image)

    boxes = retina_face["boxes"]
    scores = retina_face["scores"]
    landmarks = retina_face["landmarks"]

    best_idx = np.argmax(scores)
    
    with Image.open(BytesIO(image)) as img:
        # Warping은 RGB 이미지를 사용해도 무방합니다 (OpenCV 변환 내부 처리됨)
        image_np = np.asarray(img.convert("RGB"), dtype=np.uint8)

    best_landmarks = landmarks[best_idx].reshape(5, 2)
    
    aligned_face_rgb = _warp_face(image_np, best_landmarks)
    return aligned_face_rgb


def get_embedding(client: Any, image: bytes):
    aligned_a = run_retina_face_and_align(client, image)
    
    for align in [aligned_a]:
        aligned_face_resized = cv2.resize(align, (224, 224))
        if run_face_antispoof(client, aligned_face_resized) < 0.5:
            raise RuntimeError("There's fake face image")

    emb_a = run_face_recognition(client, aligned_a)
    return emb_a.squeeze(0)


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """

    aligned_a = run_retina_face_and_align(client, image_a)
    aligned_b = run_retina_face_and_align(client, image_b)
    
    for align in [aligned_a, aligned_b]:
        aligned_face_resized = cv2.resize(align, (224, 224))
        if run_face_antispoof(client, aligned_face_resized) < 0.5:
            raise RuntimeError("There's fake face image")

    emb_a = run_face_recognition(client, aligned_a)
    emb_b = run_face_recognition(client, aligned_b)
    return emb_a.squeeze(0), emb_b.squeeze(0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    return _cosine_similarity(emb_a, emb_b)
