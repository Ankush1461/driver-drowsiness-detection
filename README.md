---
title: DriveSafe AI
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: true
---

# 🚗 DriveSafe AI — Real-Time Driver Drowsiness Detection

> **MobileNetV2** + **CLAHE** + **FastAPI WebSocket** | 2.4 MB | 150+ FPS | 96.1% True Cross-Subject LOSO

A real-time, edge-deployable driver drowsiness detection system that achieves **99.0% in-distribution accuracy** and **96.1% cross-subject accuracy** on the UTA-RLDD benchmark — all running in under 40ms on commodity hardware.

| 🌐 **Live Demo** | 📄 **Research Paper** | 📊 **Results** |
|---|---|---|
| [HuggingFace Spaces](https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2) | JRTIP — Quantifying the Preprocessing Bottleneck in Lightweight DDD | [results/](results/) |

---

## 🏆 Key Results

| Metric | Baseline (v2) | **Adaptive (v7)** |
|---|:---:|:---:|
| **Accuracy** | 53.2% ± 19.3% | **96.1% ± 4.9%** |
| **Precision** | 34.3% | **91.2%** |
| **Recall** | 41.4% | **99.2%** |
| **F1 Score** | 0.300 | **0.947** |
| **False Alarm Rate** | 40.9% | **5.5%** |
| **Video Accuracy** | 61.1% | **61.1%** |

**Confusion Matrix (Pooled, 24 subjects):** TP=1191, FP=131, FN=10, TN=2268

> These are **true Leave-One-Subject-Out** results: the model is retrained from scratch for each of 24 subjects. No data leakage.

---

## ⚡ Performance

| Metric | Value |
|---|---|
| **Latency** | < 40ms (2-vCPU Intel) |
| **Throughput** | 150+ FPS |
| **Model Size** | 2.4 MB (dynamic quantized) |
| **Inference** | 1.2ms/frame |
| **Face Detection** | OpenCV SSD (amortized every 15th frame) |

---

## 🧠 Architecture

```
Browser (320×240 webcam)
  → WebSocket (FastAPI)
    → OpenCV SSD face detection (every 15th frame, ~17ms)
      → CLAHE preprocessing (+65.7pp accuracy boost)
        → MobileNetV2 TFLite inference (1.2ms)
          → Temporal smoothing (3-frame window)
            → Three-Zone confidence protocol
              → Alert / Uncertain / Safe
```

### Model Comparison

| Architecture | Size | LOSO Accuracy | Quantization |
|---|---|---|---|
| MobileNetV3 | 4.3 MB | 59.0% | **Collapses** (hard-swish) |
| **MobileNetV2** | **2.4 MB** | **96.1%** | **Stable** (ReLU) |

> MobileNetV3's hard-swish activations collapse to σ ≈ 0.63 under dynamic quantization. MobileNetV2's ReLU activations are quantization-stable.

---

## 🔬 Contributions

1. **Pipeline Decomposition** — Isolated the SSD face-detection bounding-box clipping as a 12.1pp accuracy bottleneck (preprocessing, not network capacity)
2. **ReLU vs. Hard-Swish Discovery** — MobileNetV3 hard-swish collapses under edge quantization; MobileNetV2 ReLU is stable
3. **Subject-Adaptive Few-Shot Calibration** — 30 frames (1 second) of active-only calibration raises LOSO from 53.2% to 96.1%
4. **Three-Zone Confidence Protocol** — Refuse ambiguous predictions → 94.7% coverage with 0% false alarm rate

---

## 🚀 Quick Start

### Local

```bash
git clone https://github.com/ankushkarmakar/DriveSafe-AI.git
cd DriveSafe-AI
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** (webcam required)

### Docker

```bash
docker build -t drivesafe-ai .
docker run -p 7860:7860 drivesafe-ai
```

### Hugging Face Spaces

Already deployed at [ankushkarmakar/drivesafe-ai-v2](https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2).

To redeploy:
```bash
git lfs install
git add .
git commit -m "Deploy DriveSafe AI"
git push origin main
```

---

## 📁 Project Structure

```
app.py                          # FastAPI WebSocket server (production)
drowsiness_v2_dynamic.tflite    # MobileNetV2 quantized (2.4 MB) — ACTIVE MODEL
drowsiness_v2.keras             # Full Keras checkpoint
deploy.prototxt                 # SSD face detector config
res10_300x300_ssd_iter_140000.caffemodel  # SSD face detector weights
face_landmarker.task            # MediaPipe face landmark model
experiments/                    # LOSO evaluation scripts (v2/v3/v6/v7)
results/                        # CSV outputs per experiment version
requirements.txt                # Python dependencies
Dockerfile                      # HuggingFace Spaces deployment
```

---

## 📊 Experiment Versions

| Version | Method | LOSO Accuracy | Notes |
|---|---|---|---|
| v2 | MobileNetV2 baseline | 53.2% ± 19.3% | Standard cross-subject |
| v3 | + k=10 adaptive fine-tuning | 59.5% ± 14.5% | Few-shot calibration |
| v6 | + LSTM + geometric features | 59.0% ± 22.7% | Complexity hurts |
| **v7** | **+ k=30 + TTA + smoothing** | **96.1% ± 4.9%** | **Best** |

---

## 📄 Citation

```bibtex
@article{karmakar2026drivesafe,
  title={DriveSafe AI: Quantifying the Preprocessing Bottleneck in Lightweight Driver Drowsiness Detection},
  author={Karmakar, Ankush},
  journal={Journal of Real-Time Image Processing},
  year={2026}
}
```

---

## 📜 License

CC BY-NC 4.0 — Non-commercial use only.
