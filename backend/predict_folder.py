# backend/predict_folder.py
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ---------------- Globals (cached) ----------------
_YOLO_MODEL = None
_YOLO_DEVICE = "cpu"

# ---------------- Preprocessor ----------------
class AdvancedPreprocessor:
    def __init__(self, iterations=5, delta=0.14, kappa=15, clahe_clip=2.0):
        self.iter = iterations
        self.delta = delta
        self.kappa = kappa
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _perona_malik(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        for _ in range(self.iter):
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

# ---------------- Model init/cache ----------------
def init_model(model_path: str, device: Optional[str] = None):
    """
    Load once and cache the Ultralytics YOLOv8 model.
    """
    global _YOLO_MODEL, _YOLO_DEVICE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(model_path)
        _YOLO_DEVICE = device
        try:
            _YOLO_MODEL.fuse()
        except Exception:
            pass
        print(f"[predict_folder] Model loaded on {device}: {model_path}")
    return _YOLO_MODEL

# ---------------- Helpers ----------------
def _decode_bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _draw_boxes(canvas: np.ndarray, boxes_xyxy, confs, clss, names: dict):
    for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{names.get(int(k), 'obj')} {float(c):.2f}"
        cv2.putText(canvas, label, (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

# ---------------- Core single-image inference ----------------
def predict_bytes(
    image_bytes: bytes,
    *,
    use_preprocess: bool = True,
    draw_on: str = "enhanced",     # "enhanced" or "original"
    conf: float = 0.10,
    iou: float = 0.50,
    imgsz: int = 896,
    max_det: int = 300,
    min_area: int = 30,
    augment: bool = True,
    guarantee_one: bool = True,
) -> Dict:
    """
    Flask-friendly: take image BYTES, return annotated PNG BYTES and metrics.
    Guarantees drawing at least one box if the model produced any raw detections.
    """
    assert _YOLO_MODEL is not None, "Call init_model(model_path) once before predict_bytes()."

    # decode
    bgr = _decode_bytes_to_bgr(image_bytes)
    if bgr is None:
        raise RuntimeError("Could not decode image bytes")

    # preprocess
    if use_preprocess:
        pre = AdvancedPreprocessor()
        bgr_enh = pre.enhance_bgr(bgr)
    else:
        bgr_enh = bgr

    canvas = bgr_enh.copy() if draw_on.lower() == "enhanced" else bgr.copy()

    # predict
    results = _YOLO_MODEL.predict(
        source=bgr_enh,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        agnostic_nms=True,
        device=_YOLO_DEVICE,
        verbose=False,
        augment=augment,
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
            if w * h >= min_area:
                tmp_xyxy.append((x1, y1, x2, y2))
                tmp_conf.append(float(c))
                tmp_cls.append(int(k))

        # Guarantee at least one (if model had any raw detections)
        if len(tmp_xyxy) == 0 and len(confs) > 0 and guarantee_one:
            import numpy as np
            best = int(np.argmax(confs))
            tmp_xyxy = [xyxy[best]]
            tmp_conf = [float(confs[best])]
            tmp_cls  = [int(clss[best])]

        kept_xyxy, kept_conf, kept_cls = tmp_xyxy, tmp_conf, tmp_cls

    names = getattr(r0, "names", {}) or {0: "AB"}
    if len(kept_xyxy) > 0:
        _draw_boxes(canvas, kept_xyxy, kept_conf, kept_cls, names)

    # encode annotated PNG
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode annotated image.")
    annotated_png = buf.tobytes()

    total = len(kept_xyxy)
    return {
        "annotated_png": annotated_png,
        "total_cells": total,
        "abnormal_cells": total,            # single-class assumption
        "confidence": (max(kept_conf) if kept_conf else 0.0),
        "boxes": kept_xyxy,
        "classes": kept_cls,
        "scores": kept_conf,
    }

# ---------------- Batch folder CLI ----------------
def predict_folder(
    model_path: str,
    input_dir: str,
    output_dir: str,
    patterns: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    **kwargs
):
    """
    Run prediction over a folder. Saves annotated PNGs to output_dir.
    kwargs are passed to predict_bytes (conf, iou, imgsz, etc.)
    """
    init_model(model_path)  # ensure loaded
    in_dir = Path(input_dir); out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in patterns]
    print(f"[predict_folder] Found {len(files)} images in {input_dir}")
    for p in files:
        b = p.read_bytes()
        res = predict_bytes(b, **kwargs)
        out_path = out_dir / f"pred_{p.stem}.png"
        with open(out_path, "wb") as f:
            f.write(res["annotated_png"])
        print(f"  ✓ {p.name} -> {out_path.name} | boxes: {res['total_cells']}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch predict a folder and save annotated PNGs.")
    ap.add_argument("--weights", required=True, help="Path to YOLOv8 .pt")
    ap.add_argument("--input",   required=True, help="Input folder with images")
    ap.add_argument("--output",  required=True, help="Output folder for annotated images")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-pre", action="store_true", help="Disable preprocessing")
    ap.add_argument("--draw-on", choices=["enhanced","original"], default="enhanced")
    args = ap.parse_args()

    predict_folder(
        model_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        min_area=args.min_area,
        augment=args.augment,
        use_preprocess=(not args.no_pre),
        draw_on=args.draw_on,
        guarantee_one=True,
    )
