# backend/utils.py
import os, io, numpy as np
from typing import List, Tuple, Iterable
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None

# ---------- FS helpers ----------
def ensure_dirs(paths: Iterable[str]):
    for p in paths:
        absp = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
        os.makedirs(absp, exist_ok=True)

# ---------- Safe decoders ----------
def read_image_any_to_bgr_bytes(data: bytes) -> np.ndarray:
    """
    Decode image bytes to BGR np.ndarray using OpenCV if available, else PIL.
    Supports PNG/JPG/TIFF and most formats.
    """
    if cv2 is not None:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img  # BGR
    pil = Image.open(io.BytesIO(data))
    if getattr(pil, "is_animated", False):
        pil.seek(0)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR) if cv2 is not None else np.array(pil.convert("RGB"))

def bgr_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    if cv2 is not None:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    # If cv2 missing, assume already RGB ndarray
    return Image.fromarray(img_bgr)

# ---------- Drawing ----------
def draw_boxes_on_pil(pil_img: Image.Image, boxes: List[List[int]]) -> Image.Image:
    """
    Draw [x1,y1,x2,y2] red rectangles on PIL image.
    """
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return out

# ---------- Preprocess (your pipeline) ----------
class AdvancedPreprocessor:
    """
    Matches your Colab training-time preprocessing:
    - Perona–Malik anisotropic diffusion (per channel)
    - CLAHE on L channel
    - Morphological close (3x3 ellipse)
    """
    def __init__(self, iterations=5, delta=0.14, kappa=15, clahe_clip=2.0):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for AdvancedPreprocessor.")
        self.iterations = iterations
        self.delta = delta
        self.kappa = kappa
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def perona_malik_diffusion(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        for _ in range(self.iterations):
            deltaN = np.zeros_like(img); deltaS = np.zeros_like(img)
            deltaE = np.zeros_like(img); deltaW = np.zeros_like(img)
            deltaN[:-1, :] = img[1:, :] - img[:-1, :]
            deltaS[1:, :]  = img[:-1, :] - img[1:, :]
            deltaE[:, :-1] = img[:, 1:] - img[:, :-1]
            deltaW[:, 1:]  = img[:, :-1] - img[:, 1:]
            cN = np.exp(-(deltaN / self.kappa) ** 2)
            cS = np.exp(-(deltaS / self.kappa) ** 2)
            cE = np.exp(-(deltaE / self.kappa) ** 2)
            cW = np.exp(-(deltaW / self.kappa) ** 2)
            img += self.delta * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)
        return np.clip(img, 0, 255).astype(np.uint8)

    def enhance_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None:
            return None
        if len(bgr.shape) == 3 and bgr.shape[2] == 3:
            enhanced = np.zeros_like(bgr)
            for i in range(3):
                enhanced[:, :, i] = self.perona_malik_diffusion(bgr[:, :, i])
        else:
            enhanced = self.perona_malik_diffusion(bgr)
        # CLAHE on L of LAB
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # Morph close
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, self.kernel)
        return enhanced

def center_square_resize_bgr(bgr: np.ndarray, size: int = 896) -> np.ndarray:
    """
    Center-crop to square then resize (same behavior you used).
    """
    h, w = bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = bgr[top:top+side, left:left+side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4) if cv2 is not None else crop

# ---------- Post-processing ----------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def post_nms(dets, post_iou=0.70):
    """
    dets: list[(x1,y1,x2,y2,conf,cls)]
    Keep highest-conf boxes; drop those with IoU >= post_iou vs kept.
    """
    if not dets: return dets
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    for d in dets:
        if all(iou(d[:4], k[:4]) < post_iou for k in keep):
            keep.append(d)
    return keep
