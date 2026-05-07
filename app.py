


import torch
import cv2
import numpy as np
import base64
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms.v2 as T  # v2 API — matches inference.py exactly

app = Flask(__name__)
CORS(app)


# ── SERVE FRONTEND ──────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(".", "style.css")

@app.route("/<page>.html")
def serve_page(page):
    return send_from_directory(".", f"{page}.html")

# ── Globals ─────────────────────────────────────────────────────
latest_image   = None
latest_results = None

# ── LOAD YOLO ───────────────────────────────────────────────────
print("Loading YOLO...")
try:
    yolo_model = YOLO("Best.pt")
    print("✅ YOLO loaded")
except Exception as e:
    print("❌ YOLO ERROR:", e)
    yolo_model = None

# ── LOAD EFFICIENTAD ────────────────────────────────────────────
print("\nLoading EfficientAD...")

anomaly_model     = None
anomaly_threshold = 0.5   # compared against raw pred_score (same scale as inference.py)

CKPT_PATH = "efficientad_screw.pt"
IMG_SIZE  = (256, 256)   # must match training resolution
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Transform: mirrors inference.py exactly.
# EfficientAD expects [0,1] floats — no ImageNet normalisation.
infer_transform = T.Compose([
    T.Resize(IMG_SIZE, antialias=True),        # antialias matches inference.py
    T.ToImage(),                               # PIL/ndarray → uint8 tensor
    T.ToDtype(torch.float32, scale=True),      # → float32 [0, 1]
])


def _load_efficientad(path):
    print(f"📂 Loading EfficientAD from: {path}")
    try:
        from anomalib.models.image.efficient_ad.lightning_model import (
            EfficientAd, EfficientAdModelSize
        )
    except ImportError:
        raise RuntimeError("anomalib not installed. Install with: pip install anomalib")

    model = EfficientAd(
        model_size           = EfficientAdModelSize.M,
        teacher_out_channels = 384,
        lr                   = 1e-4,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"✅ EfficientAD loaded on {DEVICE.upper()}")
    return model


try:
    anomaly_model = _load_efficientad(CKPT_PATH)
    print(f"✅ EfficientAD ready  (threshold={anomaly_threshold:.4f})\n")
except Exception as e:
    print(f"❌ Failed on {CKPT_PATH}: {e}\n")
    anomaly_model = None

if anomaly_model is None:
    print("❌ EFFICIENTAD LOAD FAILED — /anomaly will return 500")
    print("   Make sure efficientad_screw.pt is in the working directory.\n")


# ── HELPERS ─────────────────────────────────────────────────────
def encode_b64(img_bgr, fmt=".jpg"):
    _, buf = cv2.imencode(fmt, img_bgr)
    return base64.b64encode(buf).decode("utf-8")

def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── FULL-IMAGE EFFICIENTAD INFERENCE ────────────────────────────
def run_efficientad_full(image_bgr):
    """
    Run EfficientAD on the FULL image (exactly as it was trained).
    Returns:
        raw_score   : float  — image-level anomaly score
        amap_full   : ndarray (H_orig, W_orig) float32
                      anomaly map resized back to original image dimensions
    """
    orig_h, orig_w = image_bgr.shape[:2]
    pil_img = bgr_to_pil(image_bgr)

    tensor = infer_transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,256,256)

    # Sanity checks — catch silent dtype/shape bugs immediately
    assert tensor.dtype == torch.float32,      f"Bad dtype: {tensor.dtype}"
    assert tensor.shape == (1, 3, *IMG_SIZE),  f"Bad shape: {tensor.shape}"
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0, \
        f"Pixel range outside [0,1]: [{tensor.min():.3f}, {tensor.max():.3f}]"

    with torch.no_grad():
        output = anomaly_model(tensor)

    raw_score   = float(output.pred_score.cpu())
    amap_tensor = output.anomaly_map[0, 0].cpu().numpy()   # (256, 256)

    # Resize anomaly map back to original image size
    amap_full = cv2.resize(amap_tensor, (orig_w, orig_h),
                           interpolation=cv2.INTER_LINEAR)
    return raw_score, amap_full


def make_heatmap_b64(amap_2d):
    """Normalise a 2-D float anomaly map → JET-coloured PNG → base64."""
    mn, mx = amap_2d.min(), amap_2d.max()
    norm    = ((amap_2d - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return encode_b64(colored, ".png")


# ════════════════════════════════════════════════════════════════
# ROUTE 1: /detect
# ════════════════════════════════════════════════════════════════
@app.route("/detect", methods=["POST"])
def detect():
    global latest_image, latest_results

    if not yolo_model:
        return jsonify({"error": "YOLO not loaded"}), 500

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    img_arr = np.frombuffer(file.read(), np.uint8)
    img     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    results = yolo_model(img)[0]

    latest_image   = img.copy()
    latest_results = results

    detections = []
    screws = nuts = bolts = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id].lower()
        conf   = float(box.conf[0])
        coords = box.xyxy[0].tolist()

        detections.append({
            "id":    len(detections) + 1,
            "box":   [int(x) for x in coords],
            "conf":  round(conf, 4),
            "label": label
        })
        if   "screw" in label: screws += 1
        elif "nut"   in label: nuts   += 1
        elif "bolt"  in label: bolts  += 1

    return jsonify({
        "total": len(detections), "screws": screws,
        "nuts": nuts, "bolts": bolts, "detections": detections
    })


# ════════════════════════════════════════════════════════════════
# ROUTE 2: /anomaly
# ════════════════════════════════════════════════════════════════
@app.route("/anomaly", methods=["POST"])
def anomaly():
    global latest_image, latest_results

    if anomaly_model is None:
        return jsonify({
            "error":    "EfficientAD not loaded",
            "solution": "Check terminal — ensure efficientad_screw.pt is present"
        }), 500

    if latest_image is None or latest_results is None:
        return jsonify({"error": "No detection data — call /detect first"}), 400

    img_h, img_w = latest_image.shape[:2]

    try:
        raw_score, amap_full = run_efficientad_full(latest_image)
        print(f"  [FULL-IMG] raw_score={raw_score:.4f}")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"EfficientAD inference failed: {e}"}), 500

    report  = []
    defects = 0
    heatmap_overlay_b64 = None

    for i, box in enumerate(latest_results.boxes):
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id] if yolo_model else "unknown"
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]

        pad  = 10
        cx1  = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2  = min(img_w, x2 + pad); cy2 = min(img_h, y2 + pad)
        crop_bgr = latest_image[cy1:cy2, cx1:cx2]

        obj_score = raw_score
        obj_amap  = amap_full[cy1:cy2, cx1:cx2]

        is_defect = obj_score >= anomaly_threshold
        if is_defect:
            defects += 1

        print(f"  [OBJ {i+1}] score={obj_score:.4f}  "
              f"defect={is_defect}  threshold={anomaly_threshold:.4f}")

        # Thumbnail of the crop
        visual_b64 = encode_b64(cv2.resize(crop_bgr, (128, 128)), ".jpg") \
                     if crop_bgr.size else None

        # Heatmap: normalise the per-crop anomaly map for visualisation only
        heatmap_b64 = make_heatmap_b64(obj_amap) if obj_amap.size else None

        report.append({
            "id":           i + 1,
            "label":        label,
            "det_conf":     round(conf, 4),
            "score":        round(obj_score, 4),
            "status":       "DEFECTIVE" if is_defect else "OK",
            "visual_data":  visual_b64,
            "heatmap_data": heatmap_b64,
        })

        if heatmap_b64 and (is_defect or heatmap_overlay_b64 is None):
            heatmap_overlay_b64 = heatmap_b64

    full_overlay_b64 = make_heatmap_b64(amap_full)

    total = len(report)
    return jsonify({
        "summary":         {"total": total, "defective": defects, "ok": total - defects},
        "report":          report,
        "heatmap_overlay": full_overlay_b64,
    })


# ════════════════════════════════════════════════════════════════
# ROUTE 3: /health
# ════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "device":    DEVICE.upper(),
        "yolo":      {"loaded": yolo_model is not None,  "checkpoint": "Best.pt",            "arch": "YOLOv8n"},
        "efficientad":{"loaded": anomaly_model is not None,"checkpoint": CKPT_PATH,          "arch": "EfficientAD-M",
                       "dataset": "MVTec AD (Screw)",     "threshold": anomaly_threshold},
    })


# ════════════════════════════════════════════════════════════════
# ROUTE 4: /batch
# ════════════════════════════════════════════════════════════════
@app.route("/batch", methods=["POST"])
def batch():
    if not yolo_model:
        return jsonify({"error": "YOLO not loaded"}), 500
    if anomaly_model is None:
        return jsonify({"error": "EfficientAD not loaded"}), 500

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    total_defects = 0

    for f in files:
        img_arr = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            results.append({"filename": f.filename, "error": "Could not decode image"})
            continue

        # YOLO detection
        det_results = yolo_model(img_bgr)[0]
        det_count   = len(det_results.boxes)
        det_boxes   = []
        for box in det_results.boxes:
            cls_id = int(box.cls[0])
            det_boxes.append({
                "label": yolo_model.names[cls_id].lower(),
                "conf":  round(float(box.conf[0]), 4),
                "box":   [int(x) for x in box.xyxy[0].tolist()],
            })

        # EfficientAD on full image
        try:
            raw_score, amap_full = run_efficientad_full(img_bgr)
        except Exception as e:
            results.append({"filename": f.filename, "error": f"Anomaly inference failed: {e}"})
            continue

        is_defect = raw_score >= anomaly_threshold
        if is_defect:
            total_defects += 1

        # Thumbnail
        thumb_h = 120
        scale   = thumb_h / img_bgr.shape[0]
        thumb   = cv2.resize(img_bgr, (int(img_bgr.shape[1] * scale), thumb_h))
        _, buf  = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        thumb_b64 = base64.b64encode(buf).decode("utf-8")

        heatmap_b64 = make_heatmap_b64(amap_full)

        results.append({
            "filename":   f.filename,
            "objects":    det_count,
            "detections": det_boxes,
            "score":      round(raw_score, 4),
            "status":     "DEFECTIVE" if is_defect else "OK",
            "thumb":      thumb_b64,
            "heatmap":    heatmap_b64,
        })
        print(f"  [BATCH] {f.filename}  score={raw_score:.4f}  status={'DEFECTIVE' if is_defect else 'OK'}")

    total = len(results)
    return jsonify({
        "summary": {"total": total, "defective": total_defects, "ok": total - total_defects},
        "results": results,
    })


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 SERVER RUNNING ON http://0.0.0.0:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)