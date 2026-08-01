#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ========= EDIT THESE TWO LINES =========
MODEL_PATH = r"D:\Desktop\Desktop\Downloads\best.pt"                # <-- put your saved model path here
IMAGE_PATH = r"E:\digital viewer\2032 IMAGES\S20250918_0002.jpg"               # <-- put your image path here
# ========================================

# --- Tunable knobs (safe defaults; adjust if needed) ---
CONF      = float(os.environ.get("YOLO_CONF",  "0.12"))
NMS_IOU   = float(os.environ.get("YOLO_IOU",   "0.55"))
IMGSZ     = int(os.environ.get("YOLO_IMGSZ",   "896"))
MAX_DET   = int(os.environ.get("YOLO_MAXDET",  "300"))
MIN_AREA  = int(os.environ.get("MIN_AREA",     "40"))   # pixel area filter after model
POST_IOU  = float(os.environ.get("POST_IOU",   "0.90")) # if you add a second NMS later
AUGMENT   = os.environ.get("AUGMENT", "1") not in ("0","false","False")
DRAW_ON   = os.environ.get("DRAW_ON", "enhanced")       # "enhanced" or "original"

# -------------- Preprocessor (your training-time recipe) --------------
class AdvancedPreprocessor:
    def __init__(self, iterations=5, delta=0.14, kappa=15, clahe_clip=2.0):
        self.iterations = iterations
        self.delta = delta
        self.kappa = kappa
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _perona_malik(self, image: np.ndarray) -> np.ndarray:
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
            img += self.delta * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
        return np.clip(img, 0, 255).astype(np.uint8)

    def enhance_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None:
            return None
        if len(bgr.shape) == 3 and bgr.shape[2] == 3:
            enhanced = np.zeros_like(bgr)
            for i in range(3):
                enhanced[:, :, i] = self._perona_malik(bgr[:, :, i])
        else:
            enhanced = self._perona_malik(bgr)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, self.kernel)
        return enhanced

# -------------- Utility --------------
def _draw_boxes(canvas: np.ndarray, boxes_xyxy, confs, clss, names: dict):
    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{names.get(int(k), 'obj')} {float(c):.2f}"
        cv2.putText(canvas, label, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

# -------------- Main --------------
def main(model_path: str, image_path: str):
    model_path = Path(model_path)
    image_path = Path(image_path)
    assert model_path.exists(), f"Weights not found: {model_path}"
    assert image_path.exists(), f"Image not found: {image_path}"

    # Read
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Preprocess (your pipeline)
    pre = AdvancedPreprocessor()
    bgr_enh = pre.enhance_bgr(bgr)

    # Choose drawing background
    canvas = bgr_enh.copy() if DRAW_ON.lower() == "enhanced" else bgr.copy()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    # Predict (Ultralytics handles letterbox)
    results = model.predict(
        source=bgr_enh,
        conf=CONF,
        iou=NMS_IOU,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        agnostic_nms=True,
        device=device,
        verbose=False,
        augment=AUGMENT,
    )
    r0 = results[0]
    boxes = r0.boxes

    kept_xyxy, kept_conf, kept_cls = [], [], []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss  = boxes.cls.detach().cpu().numpy().astype(int)

        # area filter
        tmp_xyxy, tmp_conf, tmp_cls = [], [], []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w * h >= MIN_AREA:
                tmp_xyxy.append((x1, y1, x2, y2))
                tmp_conf.append(c)
                tmp_cls.append(k)

        # Guarantee at least one box (force top-1 by confidence)
        if len(tmp_xyxy) == 0 and len(confs) > 0:
            best = int(np.argmax(confs))
            tmp_xyxy = [xyxy[best]]
            tmp_conf = [confs[best]]
            tmp_cls  = [clss[best]]

        kept_xyxy, kept_conf, kept_cls = tmp_xyxy, tmp_conf, tmp_cls

    # Draw
    names = getattr(r0, "names", {}) or {0: "AB"}
    if len(kept_xyxy) > 0:
        _draw_boxes(canvas, kept_xyxy, kept_conf, kept_cls, names)

    # Save next to input
    out_path = image_path.with_name(f"pred_{image_path.name}")
    cv2.imwrite(str(out_path), canvas)
    print(f"✅ Saved: {out_path}")
    print(f"   Detections drawn: {len(kept_xyxy)} | conf={CONF}, iou={NMS_IOU}, "
          f"min_area={MIN_AREA}, imgsz={IMGSZ}, augment={AUGMENT}")

if __name__ == "__main__":
    main(MODEL_PATH, IMAGE_PATH)
