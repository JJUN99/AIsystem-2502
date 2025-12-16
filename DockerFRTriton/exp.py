import numpy as np
import cv2
import torch
import onnxruntime
from math import ceil
from itertools import product as product
import os

# ==========================================
# 1. 설정 및 경로
# ==========================================
# 🚨 모델 경로를 본인 환경에 맞게 수정하세요.
RETINAFACE_PATH = "model_repository/retinaface_mnet/1/retinaface_mnet025.onnx"
FR_MODEL_PATH = "model_repository/fr_model/1/model.onnx"

# 테스트할 이미지 경로
IMAGE_A_PATH = "IMG_4064.jpg"
IMAGE_B_PATH = "IMG_9736.jpg" # 없으면 A와 동일하게 설정됨

# 🚨 [중요] FR 모델이 요구하는 입력 크기에 맞춰 템플릿 선택
# 대부분의 공개된 InsightFace 모델은 112x112를 사용합니다.

# --- 옵션 1: 표준 112x112 템플릿 (기본값) ---
ALIGN_SIZE = (112, 112)
RETINAFACE_TEMPLATE = np.array([
    [30.2946, 51.6963], [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655], [62.7299, 92.2041]
], dtype=np.float32)

# --- 옵션 2: 160x160 템플릿 (필요시 주석 해제 후 사용) ---
# ALIGN_SIZE = (160, 160)
# RETINAFACE_TEMPLATE = np.array([
#     [43.2780, 73.8518], [93.6168, 73.5734],
#     [68.6074, 102.4808],
#     [47.9275, 131.9507], [89.6141, 131.7201]
# ], dtype=np.float32)


cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
}

# ==========================================
# 2. Helper Functions (변경 없음)
# ==========================================
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

def _warp_face(image_rgb, landmarks):
    src = landmarks.astype(np.float32)
    dst = RETINAFACE_TEMPLATE
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    # 설정된 ALIGN_SIZE로 워핑
    return cv2.warpAffine(image_rgb, matrix, ALIGN_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ==========================================
# 3. Core Logic (수정됨)
# ==========================================
def draw_debug(img, box, landms, filename):
    """디버깅용 이미지 그리기 및 저장 함수"""
    debug_img = img.copy()
    if box is not None:
        b = list(map(int, box[:4]))
        cv2.rectangle(debug_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{box[4]:.2f}", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if landms is not None:
        colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,255,0)]
        for i, (lx, ly) in enumerate(landms):
            cv2.circle(debug_img, (int(lx), int(ly)), 4, colors[i], -1)
            cv2.putText(debug_img, str(i), (int(lx)+2, int(ly)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            
    cv2.imwrite(filename, debug_img)
    print(f"📸 Saved debug image: {filename}")


def get_embedding(img_path, det_sess, fr_sess, debug_name="A"):
    print(f"\n--- Processing {debug_name}: {img_path} ---")
    # 1. Load Original Image
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_raw is None: 
        print(f"❌ Failed to load image: {img_path}")
        return None
    orig_h, orig_w = img_raw.shape[:2]

    # =========================================================
    # 🚨 [Step 1] 강제 640x640 리사이즈 (Stretching)
    # =========================================================
    target_size = 640
    img_resized = cv2.resize(img_raw, (target_size, target_size))
    print(f"ℹ️ Forced Resize: {orig_w}x{orig_h} -> 640x640")

    # 2. Preprocess for Detection (on 640x640 image)
    img_det = np.float32(img_resized) - np.array([104, 117, 123], dtype=np.float32)
    img_det = img_det.transpose(2, 0, 1)[None, ...]
    img_det = img_det.astype(np.float32)

    # 3. Detection Inference
    det_out = det_sess.run(None, {det_sess.get_inputs()[0].name: img_det})
    
    # 4. Output Parsing
    output_names = [x.name for x in det_sess.get_outputs()]
    loc, conf, landms = None, None, None
    for name, val in zip(output_names, det_out):
        if name == 'loc': loc = torch.from_numpy(val)
        if name == 'conf': conf = torch.from_numpy(val)
        if name == 'landms': landms = torch.from_numpy(val)
    if loc is None: loc, conf, landms = [torch.from_numpy(x) for x in det_out] 

    # 5. Decode (640x640 기준)
    priorbox = PriorBox(cfg_mnet, image_size=(target_size, target_size))
    priors = priorbox.forward()
    scale_box = torch.Tensor([target_size, target_size, target_size, target_size])
    scale_landms = torch.Tensor([target_size, target_size] * 5)
    
    boxes = decode(loc.data.squeeze(0), priors.data, cfg_mnet['variance']) * scale_box
    scores = conf.squeeze(0).data[:, 1]
    landmarks = decode_landm(landms.data.squeeze(0), priors.data, cfg_mnet['variance']) * scale_landms

    # =========================================================
    # 🚨 [Step 2] 필터링 (임계값 0.8) 및 NMS & IndexError 수정
    # =========================================================
    inds = np.where(scores > 0.8)[0] # 점수 0.8 이상만
    if len(inds) == 0: 
        print(f"❌ No face detected (Score < 0.8). Try lowering threshold if needed.")
        # 디버깅을 위해 리사이즈된 원본이라도 저장
        cv2.imwrite(f"step1_resized_det_{debug_name}_FAILED.jpg", img_resized)
        return None
        
    dets = np.hstack((boxes[inds].numpy(), scores[inds].numpy()[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(dets, 0.4)
    
    # 🚨 [버그 수정] 올바른 인덱싱
    best_det_idx = keep[0]         # NMS 결과 내 인덱스
    best_orig_idx = inds[best_det_idx] # 전체 결과 내 원본 인덱스
    
    best_box_640 = dets[best_det_idx].copy() # (x1, y1, x2, y2, score)
    best_lm_640 = landmarks[best_orig_idx].numpy().reshape(5, 2).copy()
    
    # 📸 [시각화 1] 640x640 리사이즈 이미지에서의 검출 결과
    draw_debug(img_resized, best_box_640, best_lm_640, f"step1_resized_det_{debug_name}.jpg")

    # =========================================================
    # 🚨 [Step 3] 좌표 복원 (원본 해상도로 스케일링)
    # =========================================================
    scale_x = orig_w / target_size
    scale_y = orig_h / target_size
    
    # 박스 좌표 복원
    best_box_orig = best_box_640.copy()
    best_box_orig[[0, 2]] *= scale_x
    best_box_orig[[1, 3]] *= scale_y
    
    # 랜드마크 좌표 복원
    best_lm_orig = best_lm_640.copy()
    best_lm_orig[:, 0] *= scale_x
    best_lm_orig[:, 1] *= scale_y
    
    # 📸 [시각화 2] 원본 이미지에서의 검출 결과 (복원된 좌표 사용)
    draw_debug(img_raw, best_box_orig, best_lm_orig, f"step2_original_det_{debug_name}.jpg")

    # =========================================================
    # 🚨 [Step 4] Alignment (원본 이미지 + 복원된 랜드마크 사용)
    # =========================================================
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    aligned_rgb = _warp_face(img_rgb, best_lm_orig)
    
    # 📸 [시각화 3] 최종 정렬된 얼굴 저장 (RGB -> BGR 변환)
    cv2.imwrite(f"step3_aligned_{debug_name}.jpg", cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
    print(f"📸 Saved aligned face: step3_aligned_{debug_name}.jpg ({ALIGN_SIZE[0]}x{ALIGN_SIZE[1]})")

    # 7. Recognition Preprocess (-1 ~ 1 정규화)
    img_fr = aligned_rgb.astype(np.float32) / 255.0
    img_fr = (img_fr - 0.5) / 0.5
    img_fr = img_fr.transpose(2, 0, 1)[None, ...]
    img_fr = img_fr.astype(np.float32)

    # 8. Embedding Inference
    embedding = fr_sess.run(None, {fr_sess.get_inputs()[0].name: img_fr})[0]
    return embedding.flatten()

def compute_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # 기존 디버그 이미지 삭제 (선택 사항)
    for f in os.listdir():
        if f.startswith("step") and f.endswith(".jpg"):
            os.remove(f)
            
    print("[INFO] Initializing sessions...")
    det_sess = onnxruntime.InferenceSession(RETINAFACE_PATH, providers=['CPUExecutionProvider'])
    fr_sess = onnxruntime.InferenceSession(FR_MODEL_PATH, providers=['CPUExecutionProvider'])

    emb_a = get_embedding(IMAGE_A_PATH, det_sess, fr_sess, debug_name="A")
    
    image_b_path = IMAGE_B_PATH if cv2.imread(IMAGE_B_PATH) is not None else IMAGE_A_PATH
    if image_b_path == IMAGE_A_PATH:
        print(f"\n⚠️ Image B not found. Comparing Image A with itself.")

    emb_b = get_embedding(image_b_path, det_sess, fr_sess, debug_name="B")

    if emb_a is None or emb_b is None:
        print("\n❌ Failed to extract embeddings. Check the generated debug images.")
        return

    similarity = compute_cosine_similarity(emb_a, emb_b)
    print(f"\n{'='*40}")
    print(f"👉 Final Cosine Similarity: {similarity:.4f}")
    print(f"{'='*40}")
    
    # 112x112 모델 기준 임계값 (모델마다 다름)
    threshold = 0.5 
    if similarity > threshold:
        print(f"✅ SAME PERSON (Match > {threshold})")
    else:
        print(f"❌ DIFFERENT PERSON (Mismatch <= {threshold})")

if __name__ == "__main__":
    main()