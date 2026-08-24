---
title: DriveSafe AI
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🛡️ DriveSafe AI — Real-Time Driver Drowsiness Detection

An enterprise-grade driver drowsiness detection system featuring a high-performance **MobileNetV3Large** backbone, optimized **TFLite** inference, and a zero-latency **FastAPI WebSocket** dashboard. Engineered for real-time CPU execution and seamless deployment on **Hugging Face Spaces**.

> 🔗 **Live Demo**: https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| **97.24% Accuracy** | MobileNetV3Large backbone with Binary Focal Crossentropy loss and Cosine Decay fine-tuning |
| **4.3 MB TFLite Model** | Dynamic range quantization — 98.1% smaller than the original 229MB InceptionResNetV2 |
| **Zero-Latency WebSocket** | FastAPI + raw WebSocket streaming with client-side canvas rendering (no HTTP overhead) |
| **Robust Face Detection** | OpenCV DNN SSD (ResNet-10) — accurate in all lighting conditions |
| **Blink-Proof Logic** | 3-second temporal buffer distinguishes real drowsiness from natural blinking |
| **Synthetic Audio Alarm** | Browser-native Web Audio API siren — no external files required |
| **Free Tier Optimized** | Thread-capped (2 cores), frame-decimated inference (~8 FPS) for Hugging Face Spaces |

---

## 🏗️ System Architecture

### High-Level Data Flow

```
┌──────────────────┐       WebSocket (ws://)        ┌───────────────────────────────────────┐
│     BROWSER      │◄──────────────────────────────►│        FastAPI SERVER (app.py)         │
│                  │                                 │                                       │
│  ┌────────────┐  │   320×240 JPEG binary blobs     │  ┌───────────────────────────────┐    │
│  │  Webcam    │──┼──────────────────────────────►  │  │  OpenCV DNN SSD (ResNet-10)   │    │
│  └────────────┘  │                                 │  │  Face Detection (every 15th)  │    │
│                  │                                 │  └──────────┬────────────────────┘    │
│  ┌────────────┐  │   JSON {status, conf, box}      │             │ face ROI                │
│  │  Canvas    │◄─┼──────────────────────────────── │  ┌──────────▼────────────────────┐    │
│  │  Overlay   │  │                                 │  │  TFLite MobileNetV3Large      │    │
│  └────────────┘  │                                 │  │  Drowsiness (every 3rd frame) │    │
│                  │                                 │  └──────────┬────────────────────┘    │
│  ┌────────────┐  │   alarm: "True" / "False"       │             │ sigmoid probability     │
│  │  Web Audio │◄─┼──────────────────────────────── │  ┌──────────▼────────────────────┐    │
│  │  API Siren │  │                                 │  │  Temporal Buffer (3s confirm)  │    │
│  └────────────┘  │                                 │  └───────────────────────────────┘    │
└──────────────────┘                                 └───────────────────────────────────────┘
```

### Inference Pipeline (Step-by-Step)

| Step | Operation | Frequency | Details |
|------|-----------|-----------|---------|
| 1 | **Frame Capture** | Every frame | Browser captures 320×240 from webcam via `<canvas>`, encodes as JPEG, sends as binary blob over WebSocket |
| 2 | **Face Detection** | Every 15th frame | OpenCV DNN SSD (`deploy.prototxt` + `res10_300x300_ssd.caffemodel`) detects face bounding box. Between detections, the previous box is reused (faces don't teleport) |
| 3 | **Face Preprocessing** | Every 3rd frame | Face ROI is **square-cropped** (preserving aspect ratio) → resized to **96×96** → converted **BGR→RGB** → cast to `float32` |
| 4 | **AI Inference** | Every 3rd frame | TFLite interpreter runs MobileNetV3Large forward pass. Output is a single **sigmoid** neuron: `0.0` = Active, `1.0` = Fatigue |
| 5 | **Temporal Filtering** | Continuous | If `probability > 0.5` persists for **≥3 consecutive seconds**, the system escalates from "DROWSY" → "**ALERT!!!**" and triggers the audio alarm |
| 6 | **Result Delivery** | Every frame | Server sends JSON `{status, confidence, alarm, bounding_box}` back. Client renders overlay on canvas — **no image is ever sent back from server** |

### Why WebSockets Instead of REST/Gradio?

Traditional web frameworks (Flask, Gradio) use HTTP request-response cycles for each frame. This introduces:
- **TCP handshake overhead** per frame (~50ms on slow connections)
- **Base64 image serialization** in both directions (doubles bandwidth)
- **Queue buildup** when the server is slower than the client's frame rate

WebSockets maintain a **single persistent TCP connection** with full-duplex streaming. The client sends raw binary JPEG blobs (no Base64 encoding), and the server returns lightweight JSON (no image return). Combined with our **Send-Acknowledge backpressure** lock (the client waits for the server's response before sending the next frame), this eliminates buffer bloat entirely.

---

## 🧠 Model Architecture & Training

### MobileNetV3Large — Why This Architecture?

MobileNetV3 was selected over alternatives for specific technical reasons:

| Architecture | Size | Accuracy | Why Not? |
|---|---|---|---|
| InceptionResNetV2 | 229 MB | ~85% | Massively over-parameterized for binary classification. Too large for web deployment |
| EfficientNetV2-B0 | ~21 MB | ~95.88% | Good accuracy but required full TensorFlow runtime on server |
| **MobileNetV3Large** | **4.3 MB** | **97.24%** | **Purpose-built for mobile/edge. Best accuracy-to-size ratio. Native TFLite support** |

MobileNetV3 uses three key innovations:
1. **Inverted Residual Blocks** — expand channels, apply depthwise convolution, then project back. 6× fewer parameters than standard convolutions.
2. **Squeeze-and-Excite Attention** — channel-wise attention that helps the model focus on discriminative eye/mouth features for drowsiness.
3. **Hard-Swish Activation** — computationally cheaper than standard Swish while maintaining accuracy.

### Two-Phase Training Strategy (`drowsiness.py`)

#### Phase 1: Neural Head Warmup (Epochs 1–20)
```
Backbone: FROZEN (ImageNet weights preserved)
Learning Rate: 1e-3 (Adam)
Training: Only the custom classification head
Head: GlobalAvgPool → BatchNorm → Dense(1024, swish) → Dropout(0.5)
       → Dense(256, swish) → Dropout(0.3) → Dense(1, sigmoid)
```
This phase teaches the classification layers to interpret MobileNetV3's feature maps for drowsiness-specific patterns (drooping eyelids, yawning, head tilt) without corrupting the pretrained backbone.

#### Phase 2: Surgical Fine-Tuning (Epochs 21–100)
```
Backbone: Last 100 layers UNFROZEN
Learning Rate: Cosine Decay (5e-5 → 5e-7)
BatchNorm Layers: FROZEN (preserve running statistics)
Early Stopping: patience=12, restore_best_weights=True
```
Unfreezing the deeper layers allows the model to adapt low-level feature detectors (edge detection, texture recognition) specifically for facial fatigue patterns, while the cosine decay prevents catastrophic forgetting.

#### Loss Function: Binary Focal Crossentropy (γ=2.5)

Standard binary crossentropy treats all misclassifications equally. Focal Loss adds an exponential focusing parameter that **down-weights easy examples** and forces the model to focus on the hard-to-classify 2.76% of images it's currently missing. With `gamma=2.5`, a correctly classified example at 90% confidence contributes 100× less to the loss than a misclassified example at 50% confidence.

#### Data Augmentation Pipeline

| Augmentation | Range | Purpose |
|---|---|---|
| `RandomFlip` | Horizontal | Driver can be on either side of camera |
| `RandomRotation` | ±18° | Simulates head tilt during drowsiness |
| `RandomTranslation` | ±10% | Robustness to face position jitter |
| `RandomZoom` | ±10% | Handles varying camera distances |
| `RandomBrightness` | ±15% | Day/night driving conditions |
| `RandomContrast` | ±15% | Compensates for windshield glare/shadows |

---

## 📊 Benchmark Results

Independently validated against **2,976 images** (1,782 Active + 1,192 Fatigue) using `evaluate_model.py`:

| Metric | Value |
|---|---|
| **Final Accuracy** | 97.24% |
| **Turbo Latency** | 104.47 ms / image |
| **Throughput** | 9 images / second |
| **Active Precision** | 97.3% (1748 / 1796 predictions) |
| **Fatigue Recall** | 97.1% (1144 / 1192 actual fatigue cases) |
| **False Alarm Rate** | 1.9% (34 active faces misclassified as drowsy) |
| **Miss Rate** | 4.0% (48 drowsy faces misclassified as active) |

```text
--- CONFUSION MATRIX ---
                         | Pred Active Subjects | Pred Fatigue Subjects
Actual Active Subjects   |           1748       |            34
Actual Fatigue Subjects  |             48       |          1144
```

To reproduce:
```bash
# Place test images in dataset/test/Active Subjects/ and dataset/test/Fatigue Subjects/
python evaluate_model.py
```

---

## ⚡ Hugging Face Free Tier Optimizations

The system is specifically engineered for Hugging Face Spaces **Free Tier** containers (2 vCPUs, 16 GB RAM):

| Optimization | Implementation | Impact |
|---|---|---|
| **Thread Capping** | `cv2.setNumThreads(2)` + TFLite `num_threads=2` | Prevents OpenCV from spawning 16 threads on a 2-core container, eliminating context-switching overhead |
| **Face Detection Decimation** | SSD runs every **15th** frame | ~93% reduction in face detection compute. Faces don't move significantly between frames |
| **Inference Decimation** | Drowsiness model runs every **3rd** frame (~8 FPS) | ~66% reduction in AI compute. Drowsiness develops over seconds, not milliseconds |
| **WebSocket Backpressure** | Client waits for server ACK before sending next frame | Eliminates queue buildup that causes growing latency over time |
| **Aspect Ratio Preservation** | 320×240 capture + square crop before 96×96 resize | Prevents facial distortion without additional compute |
| **JSON-Only Response** | Server returns `{status, conf, box}` — no image data | ~95% bandwidth reduction vs. sending processed frames back |

---

## 🔄 Evolution from Original Project

| Aspect | Original Version | Current Version |
|---|---|---|
| **Model** | InceptionResNetV2 (229 MB `.h5`) | MobileNetV3Large TFLite (4.3 MB `.tflite`) |
| **Accuracy** | ~85% (biased toward fatigue class) | 97.24% (balanced, independently verified) |
| **Face Detection** | Haar Cascades (fails in low light) | OpenCV DNN SSD ResNet-10 (all conditions) |
| **Inference Runtime** | Full TensorFlow (~500ms/frame) | TFLite via LiteRT (~104ms/frame) |
| **Web Framework** | None / basic Gradio | FastAPI + WebSocket (full-duplex) |
| **Input Resolution** | 224×224 (mismatched with model) | 96×96 (exact match to training) |
| **Color Space** | BGR fed directly (incorrect) | BGR→RGB conversion before inference |
| **Classification** | `argmax` on sigmoid (always wrong) | Probability threshold > 0.5 |
| **Alarm** | External `.wav` file (404 on deploy) | Synthetic Web Audio API siren |
| **Deployment** | Local only | Hugging Face Docker Space (live URL) |
| **Model Size Reduction** | — | **98.1% smaller** (229 MB → 4.3 MB) |
| **Latency Improvement** | — | **~5× faster** (500ms → 104ms) |

---

## 💻 Local Installation

```bash
# 1. Clone
git clone https://github.com/Ankush1461/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# 2. Virtual Environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Web Application
python app.py

# 5. Or run the standalone OpenCV test (no browser needed)
python drowsiness_cv.py
```

Open your browser at **http://127.0.0.1:7860** (webcam required).

### Dependencies (`requirements.txt`)
```
ai-edge-litert          # Google LiteRT (TFLite Runtime)
opencv-python-headless  # OpenCV without GUI (server-safe)
numpy                   # Numerical operations
fastapi                 # Web server framework
uvicorn[standard]       # ASGI server
websockets              # WebSocket protocol support
scikit-learn            # Evaluation metrics
```

---

## 🚀 Deployment Guide

### GitHub Repository Setup

**Prerequisites:** Git LFS must be installed for large model files.

```bash
# 1. Install Git LFS (one-time setup)
git lfs install

# 2. Navigate to the repository
cd driver-drowsiness-detection

# 3. Stage all files (models tracked by LFS automatically)
git add .

# 4. Commit
git commit -m "Initial deployment: DriveSafe AI with MobileNetV3Large"

# 5. Push to GitHub
git push origin main
```

**Verify LFS upload:**

```bash
# Confirm large files are tracked by LFS
git lfs ls-files
# Should show: drowsiness.tflite, drowsiness.keras, res10_*.caffemodel
```

> ⚠️ **GitHub LFS Quota:** Free tier includes 1 GB storage + 1 GB/month bandwidth. Your models total ~150 MB. Monitor usage at https://github.com/settings/billing

### Hugging Face Spaces Deployment

Your repository is pre-configured for HF Spaces via the `README.md` YAML frontmatter (`sdk: docker`).

**Option A: Automatic Deployment (Recommended)**

```bash
# Push to main triggers automatic Docker build
git push origin main
```

Then visit https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2 to monitor the build.

**Option B: Manual Space Creation**

If the Space doesn't exist yet:

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Name:** drivesafe-ai-v2
   - **SDK:** Docker
   - **License:** CC BY-NC 4.0
3. Link your repository: `Ankush1461/driver-drowsiness-detection`
4. The Space will auto-build from your Dockerfile

**Option C: Rebuild via API**

```bash
# Restart the Space (triggers rebuild)
curl -X POST https://huggingface.co/api/spaces/ankushkarmakar/drivesafe-ai-v2/restart
```

### Docker Build (Local Testing)

Test the Docker image locally before pushing:

```bash
# Build the image
docker build -t drivesafe-ai .

# Run the container
docker run -p 7860:7860 drivesafe-ai

# Open http://localhost:7860 in your browser
```

**Expected Docker image contents:**

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~15 KB | FastAPI WebSocket server |
| `drowsiness.tflite` | 4.3 MB | Quantized inference model |
| `res10_300x300_ssd_iter_140000.caffemodel` | 11 MB | Face detector weights |
| `deploy.prototxt` | 30 KB | SSD network architecture |
| `face_landmarker.task` | 3.6 MB | MediaPipe (legacy) |
| `requirements.txt` | ~1 KB | Python dependencies |

Total image size: ~250 MB (slim Python base + OpenCV deps + models)

### Deployment Checklist

```text
[✓] .gitignore configured (excludes datasets, IDE files, __pycache__)
[✓] .dockerignore configured (excludes datasets, experiments, model variants)
[✓] .gitattributes tracks large files via LFS (.keras, .tflite, .caffemodel)
[✓] Dockerfile uses python:3.10-slim (minimal attack surface)
[✓] README.md has HF Spaces YAML frontmatter (sdk: docker)
[✓] requirements.txt has all runtime dependencies
[✓] app.py runs on port 7860 (HF Spaces default)
[✓] Models are quantized (4.3 MB TFLite, fits in free tier)
[✓] Thread count capped for 2-core free tier containers
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| **Build fails** | LFS files not downloaded | Add `RUN git lfs pull` to Dockerfile |
| **Out of memory** | Too many threads | Verify `cv2.setNumThreads(2)` in app.py |
| **WebSocket timeout** | HF proxy idle disconnect | Send-Ack backpressure prevents this |
| **Camera blocked** | HTTP instead of HTTPS | HF Spaces serves over HTTPS by default |
| **Slow first inference** | Cold start model loading | Expected on free tier; first frame takes ~2s |


---

## 📁 Project Structure

```
├── dataset
│   └── train
│       ├── Active Subjects          # Training images — alert drivers
│       └── Fatigue Subjects         # Training images — drowsy drivers
├── app.py                           # FastAPI WebSocket server + embedded HTML/CSS/JS dashboard
├── drowsiness.py                    # MobileNetV3Large 2-phase training pipeline
├── drowsiness.tflite                # Quantized TFLite model (4.3 MB)
├── drowsiness.keras                 # Full Keras model checkpoint (50 MB)
├── drowsiness_cv.py                 # Standalone local OpenCV testing script
├── evaluate_model.py                # Independent accuracy benchmarking tool
├── clean_dataset.py                 # Dataset sanitization & deduplication utility
├── deploy.prototxt                  # SSD face detector network architecture definition
├── res10_300x300_ssd_iter_140000    # SSD face detector pretrained weights (10 MB)
│   .caffemodel
├── experiments/                     # Ablation studies and training scripts
├── results/                         # CSV and TXT experiment results
├── face_landmarker.task             # MediaPipe face landmark model (legacy, unused)
├── requirements.txt                 # Python package dependencies
├── Dockerfile                       # Hugging Face Spaces Docker container config
├── .dockerignore                    # Docker build exclusions
├── .gitignore                       # Git ignore rules
├── .gitattributes                   # Git LFS tracking for model files
├── LICENSE                          # CC BY-NC 4.0 license
└── README.md                        # This file (includes HF Space YAML metadata)
```

---

## 🛠️ Retraining the Model

If you want to retrain with your own dataset:

1. **Prepare your dataset** — organize images into `dataset/train/Active Subjects/` and `dataset/train/Fatigue Subjects/`.
2. **(Optional) Clean the dataset** — run `python clean_dataset.py` to remove corrupt/duplicate images.
3. **Train** — run `python drowsiness.py`. This will:
   - Phase 1: Train the classification head for 20 epochs (frozen backbone)
   - Phase 2: Fine-tune the last 100 backbone layers for up to 80 more epochs
   - Auto-save the best model as `drowsiness.keras`
   - Auto-convert to `drowsiness.tflite` for deployment
4. **Evaluate** — run `python evaluate_model.py` to benchmark against your test set.

---

## ⚖️ License & Copyright

**© 2026 DriveSafe AI. All rights reserved.**

Licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- ✅ **Permitted**: Personal use, academic research, learning from the code.
- ❌ **Prohibited**: Commercial use, integration into paid products, monetization without explicit written permission.
