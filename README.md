---
title: VisionQC
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# VisionQC — Industrial Defect Detection AI

> A two-stage deep learning pipeline for real-time industrial hardware inspection.  
> Combines **YOLOv8** object detection with **EfficientAD** anomaly analysis to locate and score defects in screws, nuts, and bolts.

**Live Demo Frontend:** Open `index.html` in your browser  
**Inspector App:** Start the Flask server → open `dashboard.html`

---

## Overview

VisionQC is a full-stack AI quality control system built as a final-year project at **SIT Pune (AI & ML Department)**. It runs a two-stage pipeline:

1. **Stage 1 — Detection:** YOLOv8n detects and localises hardware components (screws, nuts, bolts) with bounding boxes and confidence scores.
2. **Stage 2 — Anomaly Scoring:** EfficientAD-M generates a pixel-level anomaly heatmap and an image-level defect score, trained on the MVTec AD screw dataset — no defect labels needed at inference.

---

## Features

| Feature | Description |
|---|---|
| Single Scan | Upload one image → full detect + anomaly report |
| Batch Processing | Upload 2+ images → batch pipeline → unified report |
| Live Camera | Capture from webcam, detect and analyze in one click |
| Anomaly Heatmap | JET-coloured pixel-level defect map overlaid on image |
| Inspection Report | Pass/fail verdict, bounding boxes, scores, thumbnails |
| CSV Export | One row per detected object with YOLO coords + score |
| PDF/Print Report | Print-ready audit trail via browser print dialog |
| Dataset Explorer | Dataset page with 5-source breakdown and bounding box gallery |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Anomaly Detection | EfficientAD-M (anomalib) |
| Deep Learning | PyTorch + torchvision |
| Backend API | Flask + Flask-CORS |
| Computer Vision | OpenCV |
| Frontend | Vanilla HTML/CSS/JS (no framework) |

---

## Project Structure

```
VisionQC/
├── app.py                        # Flask backend (4 API routes)
├── index.html                    # Landing page
├── dashboard.html                # Inspector app (single + batch)
├── report.html                   # Inspection report viewer
├── dataset.html                  # Training dataset explorer
├── style.css                     # Shared design system
├── requirements.txt              # Python dependencies
│
├── Best.pt                       # YOLOv8n trained checkpoint (50 MB)
├── efficientad_screw.pt          # EfficientAD-M checkpoint (80 MB)
│
├── static/
│   ├── dataset/                  # Sample images used in dataset.html
│   │   ├── visualized/           # Bounding box annotation previews
│   │   ├── mvtec/                # MVTec-AD sample images
│   │   ├── synthetic/            # Synthetic collage samples
│   │   ├── own/                  # Custom lab-captured samples
│   │   ├── negatives/            # Hard negative samples
│   │   └── roboflow/             # Roboflow dataset samples
│   ├── output/                   # Sample reports & CSV for landing page
│   └── efficientAD_result.jpeg   # EfficientAD inference visualization
│
├── Images/
│   ├── dataset_processed/        # Full processed training set
│   │   ├── images/               # Raw training images
│   │   ├── labels/               # YOLO .txt annotation files
│   │   └── visualized/           # Annotated bounding box images
│   ├── yolo_train_img/           # Sample images per source category
│   └── demo_images/              # Good/defected demo images
│
└── other/
    ├── inference.py              # Standalone EfficientAD inference script
    └── verify_gpu.py             # GPU availability checker
```

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `POST` | `/detect` | Run YOLOv8 on uploaded image |
| `POST` | `/anomaly` | Run EfficientAD on last detected image |
| `POST` | `/batch` | Run full pipeline on multiple images |
| `GET` | `/health` | Check model load status and device |

---

## Setup & Running

### 1. Clone the repository

```bash
git clone https://github.com/devgandhi2705/VisionQC.git
cd VisionQC
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (recommended):** Install the CUDA-enabled PyTorch build for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

### 4. Start the Flask server

```bash
python app.py
```

Server starts at `http://127.0.0.1:5000`

### 5. Open the app

Open `index.html` in your browser (double-click or use Live Server in VS Code).  
Click **Launch Inspector** → the dashboard connects to the Flask backend automatically.

---

## Model Checkpoints

| File | Architecture | Dataset | Size |
|---|---|---|---|
| `Best.pt` | YOLOv8n | Custom 3,500+ image corpus | 50 MB |
| `efficientad_screw.pt` | EfficientAD-M | MVTec AD — Screw (320 normal) | 80 MB |

---

## Training Dataset

The YOLOv8 model was trained on a 3,500+ image corpus from **5 sources**:

| Source | Description | Count |
|---|---|---|
| S1 — MVTec-AD | Studio-lit normal screw images (benchmark) | 320 |
| S2 — Custom | Lab-captured photos + Roboflow augmentation | 60+ |
| S3 — Roboflow | Community-labeled hardware images | ~200 |
| S4 — Synthetic | Programmatic multi-object collages | 250+ |
| S5 — Negatives | Hard negatives (bolts, nails, pins) | ~100 |

**YOLOv8 Results:**
- Precision: **94.30%**
- Recall: **89.40%**
- mAP@50: **94.46%**
- Inference: **4.3 ms/image** (GPU)

**EfficientAD Results (MVTec AD — Screw):**
- Image AUROC: **96.98%**
- Pixel AUROC: **97.89%**

---

## Deployment

### Recommended: Hugging Face Spaces (Docker)

Hugging Face Spaces supports Flask apps via Docker and is the best free platform for ML inference. GPU Spaces (NVIDIA T4) are available at ~$0.60/hr when needed.

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) → select **Docker** SDK
2. Add a `Dockerfile` in the repo root:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

3. Update `app.py` to bind on port `7860`:
```python
app.run(host="0.0.0.0", port=7860)
```

4. Push to your HF Space repo — it auto-deploys.

> For GPU inference on HF Spaces, upgrade the Space hardware to **T4 Small** ($0.60/hr) in Space Settings.

### Alternative: Modal (Serverless GPU)

[Modal.com](https://modal.com) offers $30/month free GPU credit (T4/A10G). Deploy Flask as a Modal web endpoint — no Docker setup required. Ideal for demo/testing.

---

## Screenshots

| Landing Page | Inspector Dashboard |
|---|---|
| ![Landing](static/output/single_report.png) | ![Dashboard](static/output/batch_report.png) |

---

## Built By

**SIT Pune — Department of AI & ML**  
Final Year Project · 2025–26

---

## License

This project is for academic and educational purposes.
