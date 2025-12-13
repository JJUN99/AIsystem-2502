"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List
import os
import cv2, numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from insightface import model_zoo

RETINAFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)

def _ensure_bgr(image):
    if isinstance(image, bytes):
        arr = np.frombuffer(image, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("decode failed")
        return decoded
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8, copy=False)
    raise TypeError("Unsupported image type")

from functools import lru_cache
@lru_cache(maxsize=1)
def _get_detector():
    """ 캐시된 FaceAnalysis 탐지기를 반환합니다. """
    detector = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector

@lru_cache(maxsize=1)
def _get_embedder():
    embedder = model_zoo.get_model("./webface_r50.onnx")
    embedder.prepare(ctx_id=0)
    return embedder




def detect_faces(image_bgr: Any) -> List[Any]:
    """
    Detect faces within the provided image.

    Parameters can be raw image bytes or a decoded image object, depending on
    the student implementation. Expected to return a list of face regions
    (e.g., bounding boxes or cropped images).
    """
    detector = _get_detector()

    faces = detector.get(image_bgr)
    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.astype(int).tolist(),   # [x1, y1, x2, y2]
            "score": float(face.det_score),           # 신뢰도
        })
    results.sort(key=lambda f: f["score"], reverse=True)
    return results


def compute_face_embedding(aligned_bgr: Any) -> Any:
    """
    Compute a numerical embedding vector for the provided face image.

    The embedding should capture discriminative facial features for comparison.
    """
    embedder = _get_embedder()

    face = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = np.transpose(face, (2, 0, 1))  # HWC → CHW
    face = np.expand_dims(face, axis=0)   # add batch dimension
    face = face.astype(np.float32)
    emb = embedder.forward(face)          # returns numpy array [1, 512]
    return emb[0]



def detect_face_keypoints(image_bgr: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    detector = _get_detector()

    faces = detector.get(image_bgr)
    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.astype(int).tolist(),   # [x1, y1, x2, y2]
            "score": float(face.det_score),           # 신뢰도
            "landmarks": face.kps.tolist(),           # 5점 키포인트 
        })
    results.sort(key=lambda f: f["score"], reverse=True)
    return results

def warp_face(image_bgr: Any, landmarks: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    if landmarks.shape != (5, 2):
        raise ValueError("Expected 5-point landmarks")

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
    return aligned


def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    bgr_a = _ensure_bgr(image_a)
    faces_a = detect_face_keypoints(bgr_a)
    aligned_faces_a = []

    for idx, face in enumerate(faces_a, start=1):
        landmarks = np.array(face["landmarks"], dtype=np.float32)
        aligned = warp_face(bgr_a, landmarks)           # 112×112 BGR
        aligned_faces_a.append(aligned)
    embeddings_a = []
    for aligned in aligned_faces_a:
        emb = compute_face_embedding(aligned)
        norm = np.linalg.norm(emb)
        if norm < 1e-12:
            raise ValueError("Embedding norm is zero; cannot normalize.")
        embeddings_a.append(emb / norm)

    bgr_b = _ensure_bgr(image_b)
    faces_b = detect_face_keypoints(bgr_b)
    aligned_faces_b = []

    for idx, face in enumerate(faces_b, start=1):
        landmarks = np.array(face["landmarks"], dtype=np.float32)
        aligned = warp_face(bgr_b, landmarks)           # 112×112 BGR
        aligned_faces_b.append(aligned)
    embeddings_b = []
    for aligned in aligned_faces_b:
        emb = compute_face_embedding(aligned)
        norm = np.linalg.norm(emb)
        if norm < 1e-12:
            raise ValueError("Embedding norm is zero; cannot normalize.")
        embeddings_b.append(emb / norm)

    max_similarity = -1.0  # 모든 내적이 음수일 수도 있으니 -1에서 시작
    for emb_a in embeddings_a:
        for emb_b in embeddings_b:
            sim = float(np.dot(emb_a, emb_b))
            if sim > max_similarity:
                max_similarity = sim

    return max_similarity
    

if __name__ == "__main__":
    # image_path = "IMG_1078_converted.jpg"
    # output_path = "IMG_1078_detected_keypoints.jpg"

    # with open(image_path, "rb") as handle:
    #     image_bytes = handle.read()

    # bgr = _ensure_bgr(image_bytes)
    # faces = detect_face_keypoints(bgr)

    # for face in faces:
    #     x1, y1, x2, y2 = face["bbox"]
    #     score = face["score"]
    #     landmarks = np.array(face["landmarks"], dtype=int)

    #     cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(
    #         bgr,
    #         f"{score:.2f}",
    #         (x1, max(y1 - 10, 0)),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (0, 255, 0),
    #         2,
    #         cv2.LINE_AA,
    #     )

    #     for idx, (px, py) in enumerate(landmarks):
    #         cv2.circle(bgr, (px, py), 3, (0, 0, 255), -1)
    #         cv2.putText(
    #             bgr,
    #             str(idx + 1),
    #             (px + 4, py - 4),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4,
    #             (0, 0, 255),
    #             1,
    #             cv2.LINE_AA,
    #         )

    # cv2.imwrite(output_path, bgr)
    # print(f"Saved detection+keypoints overlay to {output_path}")

    # image_path = "IMG_1078_converted.jpg"
    # output_dir = "aligned_faces"
    # os.makedirs(output_dir, exist_ok=True)

    # with open(image_path, "rb") as handle:
    #     image_bytes = handle.read()

    # bgr = _ensure_bgr(image_bytes)
    # faces = detect_face_keypoints(bgr)  # bbox + score + landmarks


    # aligned_faces = []
    # for idx, face in enumerate(faces, start=1):
    #     landmarks = np.array(face["landmarks"], dtype=np.float32)
    #     aligned = warp_face(bgr, landmarks)           # 112×112 BGR
    #     aligned_faces.append(aligned)

    #     filename = os.path.join(output_dir, f"face_{idx:02d}.png")
    #     cv2.imwrite(filename, aligned)
    #     print(f"Saved aligned face #{idx} -> {filename}")

    # embeddings = [compute_face_embedding(aligned) for aligned in aligned_faces]
    # print(embeddings)
    
    image_path_a, image_path_b = './IMG_0647.jpg', './IMG_1078_converted.jpg'
    with open(image_path_a, "rb") as handle:
        image_bytes_a = handle.read()
    with open(image_path_b, "rb") as handle:
        image_bytes_b = handle.read()

    print(calculate_face_similarity(image_bytes_a, image_bytes_b))