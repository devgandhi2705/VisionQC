"""
EfficientAD Anomaly Detection - Inference Script
Usage:
    python inference.py --model efficientad_screw.pt --image your_image.jpg
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
import os

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE        = (256, 256)       # EfficientAD default input resolution
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
ANOMALY_THRESHOLD = 0.5              # ← tune on your validation set


# ── Pre-processing ─────────────────────────────────────────────────────────────
# EfficientAD expects [0, 1] floats — no ImageNet normalisation
transform = T.Compose([
    T.Resize(IMAGE_SIZE, antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),   # → [0, 1]
])


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load an EfficientAD checkpoint (.pt with state_dict)."""
    print(f"[INFO] Loading model from: {model_path}")

    try:
        from anomalib.models.image.efficient_ad.lightning_model import (
            EfficientAd, EfficientAdModelSize
        )
    except ImportError:
        raise RuntimeError(
            "anomalib is not installed. Install it with: pip install anomalib"
        )

    try:
        model = EfficientAd(
            model_size           = EfficientAdModelSize.M,
            teacher_out_channels = 384,
            lr                   = 1e-4,
        )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Support both plain state_dict and Lightning-style checkpoints
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(DEVICE)
        print(f"[INFO] Model loaded successfully on {DEVICE.upper()}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e


# ── Image loading ──────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> tuple:
    """Load and preprocess an image. Returns (tensor, original PIL image)."""
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)
    return tensor, img


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, image_tensor: torch.Tensor) -> dict:
    """
    Run EfficientAD inference.
    anomalib's EfficientAD returns an ImageResult with:
      • .pred_score   — scalar anomaly score
      • .anomaly_map  — (1, 1, H, W) heatmap tensor
    """
    with torch.no_grad():
        output = model(image_tensor)

    pred_score  = float(output.pred_score.cpu())
    anomaly_map = output.anomaly_map[0, 0].cpu().numpy()   # (H, W)

    return {"score": pred_score, "anomaly_map": anomaly_map}


# ── Visualisation ──────────────────────────────────────────────────────────────
def visualise(original_img: Image.Image, result: dict, output_path: str):
    """Save a 3-panel visualisation: original | heatmap | blended overlay."""
    score       = result["score"]
    anomaly_map = result["anomaly_map"]
    is_anomaly  = score > ANOMALY_THRESHOLD

    # Normalise heatmap to [0, 1]
    heat_norm = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    # Resize original for overlay
    img_np = np.array(original_img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0])))  # (H, W, 3)

    label = (
        f"ANOMALY  (score: {score:.4f})"
        if is_anomaly
        else f"NORMAL  (score: {score:.4f})"
    )
    title_color = "red" if is_anomaly else "green"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"EfficientAD  |  Score: {score:.4f}  |  "
        f"Threshold: {ANOMALY_THRESHOLD:.4f}  |  {label}",
        fontsize=12, fontweight="bold", color=title_color,
    )

    # Panel 1 — Original
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — Heatmap
    im = axes[1].imshow(heat_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Anomaly Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — Blended overlay
    overlay  = img_np.astype(float) / 255.0
    heat_rgb = plt.cm.jet(heat_norm)[..., :3]
    blended  = 0.55 * overlay + 0.45 * heat_rgb
    axes[2].imshow(np.clip(blended, 0, 1))
    axes[2].set_title("Heatmap Overlay", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Visualisation saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global ANOMALY_THRESHOLD

    parser = argparse.ArgumentParser(description="EfficientAD inference on a single image")
    parser.add_argument("--model",     required=True,              help="Path to efficientad checkpoint (.pt)")
    parser.add_argument("--image",     required=True,              help="Path to input image")
    parser.add_argument("--threshold", type=float, default=None,   help=f"Anomaly score threshold (default: {ANOMALY_THRESHOLD})")
    parser.add_argument("--output",    default="result.png",       help="Output visualisation path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.threshold is not None:
        ANOMALY_THRESHOLD = args.threshold

    model              = load_model(args.model)
    image_tensor, orig = preprocess_image(args.image)
    result             = run_inference(model, image_tensor)

    # Pixel-level defect stats
    heat_norm  = result["anomaly_map"]
    heat_norm  = (heat_norm - heat_norm.min()) / (heat_norm.max() - heat_norm.min() + 1e-8)
    pred_mask  = (heat_norm >= ANOMALY_THRESHOLD).astype(np.uint8)

    print(f"\n{'='*45}")
    print(f"  Anomaly Score  : {result['score']:.6f}")
    print(f"  Threshold      : {ANOMALY_THRESHOLD}")
    print(f"  Decision       : {'⚠  ANOMALY DETECTED' if result['score'] > ANOMALY_THRESHOLD else '✓  NORMAL — no defect'}")
    print(f"  Defect pixels  : {pred_mask.sum()} / {pred_mask.size} ({100 * pred_mask.sum() / pred_mask.size:.1f}%)")
    print(f"{'='*45}\n")

    visualise(orig, result, args.output)


if __name__ == "__main__":
    main()