---
title: DriveSafe AI
emoji: car
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: true
---

# DriveSafe AI - Real-Time Driver Drowsiness Detection

> **99.0% accuracy** | **2.4 MB** | **150+ FPS** | **Sub-40ms latency**
> A complete edge-deployed drowsiness detection pipeline: MobileNetV2 + CLAHE + FastAPI WebSocket

DriveSafe AI is a lightweight, real-time driver drowsiness detection system designed for **edge deployment without GPU**. It runs a full inference pipeline - face detection, lighting normalization, neural classification, temporal smoothing, and uncertainty-aware alerting - in under 40 milliseconds on a standard 2-vCPU Intel laptop.

**[Try the Live Demo](https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2)** - Grant camera access and the system monitors you in real time.

**[Read the Paper](DriveSafe_AI_JRTIP.tex)** - Published in the *Journal of Real-Time Image Processing (JRTIP)*.

---

## Results Across Evaluation Scenarios

This system was evaluated under three increasingly realistic conditions. The accuracy gap between them tells an important story about what 99% accuracy really means in the real world.

### Scenario 1: In-Distribution (Single Dataset Split)

The standard train/test split on the same dataset evaluation. This is what most papers report.

| Metric | MobileNetV2 + CLAHE |
|---|:---:|
| **Accuracy** | **99.0%** |
| **Precision** | 98.8% |
| **Recall** | 99.2% |
| **F1 Score** | 0.990 |
| **Model Size** | 2.4 MB (dynamic quantized) |
| **Inference Time** | 1.2ms per frame |

> **This is the live demo accuracy.** The webcam demo uses this exact pipeline - MobileNetV2 with CLAHE preprocessing, running on a TFLite quantized model.

### Scenario 2: Cross-Subject Baseline (True LOSO, No Adaptation)

The hardest realistic test: train on 23 subjects, test on the 24th, repeated for all 24 subjects. The model has **never seen the test subject face**.

| Metric | Baseline (v2) |
|---|:---:|
| **Accuracy** | 53.2% +/- 19.3% |
| **Precision** | 34.3% |
| **Recall** | 41.4% |
| **F1 Score** | 0.300 |
| **False Alarm Rate** | 40.9% |

> **This is the honest truth most papers hide.** Without subject-specific calibration, a drowsiness detector that works at 99% on known drivers drops to barely above random chance on new drivers. This is the core problem this research addresses.

### Scenario 3: Cross-Subject with Personalization (True LOSO + Adaptive Calibration)

The same true LOSO protocol, but with a **30-frame (1 second) calibration step** at vehicle startup. The driver just sits normally for 1 second - no labeling needed, just active (awake) frames.

| Metric | Baseline (v2) | **Adaptive (v7)** | Improvement |
|---|:---:|:---:|:---:|
| **Accuracy** | 53.2% +/- 19.3% | **96.1% +/- 4.9%** | +42.9pp |
| **Precision** | 34.3% | **91.2%** | +56.9pp |
| **Recall** | 41.4% | **99.2%** | +57.8pp |
| **F1 Score** | 0.300 | **0.947** | +0.647 |
| **False Alarm Rate** | 40.9% | **5.5%** | -35.4pp |
| **Video Accuracy** | 61.1% | **61.1%** | - |

**Confusion Matrix (Pooled, 24 subjects):** TP=1,191 / FP=131 / FN=10 / TN=2,268

> **The solution: personalization, not bigger models.** Just 30 frames of calibration (1 second of video) lifts cross-subject accuracy from 53% to 96%. The model is retrained from scratch for each subject - no data leakage.

### Per-Subject Breakdown (Adaptive, v7)

| Range | Subjects | Count |
|---|---|:---:|
| 95-100% | S02, S03, S07, S09, S10, S11, S13, S15, S16, S18, S19, S20, S21, S22, S24 | 15 |
| 90-95% | S01, S04, S05, S06, S08, S12, S14, S17, S23 | 9 |
| < 90% | - | 0 |

> All 24 subjects are above 90% accuracy. 15 subjects are above 95%.

---

## Performance Profile

| Metric | Value |
|---|---|
| **End-to-End Latency** | < 40ms (2-vCPU Intel) |
| **Throughput** | 150+ FPS |
| **Model Size** | 2.4 MB (dynamic quantized, 72% compression from 8.6 MB Float32) |
| **Inference Time** | 1.2ms per frame (MobileNetV2 TFLite) |
| **Face Detection** | OpenCV SSD, amortized every 15th frame (~17ms) |
| **Total Pipeline** | Face detect -> CLAHE -> TFLite inference -> Temporal smoothing -> Alert |

---

## How It Works

Browser captures 320x240 webcam -> WebSocket -> OpenCV SSD face detection (every 15th frame) -> CLAHE -> MobileNetV2 TFLite (every 3rd frame) -> Temporal buffer (3s) -> Alert

### Three-Zone Confidence Protocol

Instead of forcing binary decisions on ambiguous frames, the system refuses to classify uncertain predictions:

| Zone | Threshold | Behavior | Coverage |
|---|---|---|:---:|
| **Active (Safe)** | sigma < 0.05 | Driver is alert | 82.1% |
| **Uncertain** | 0.05 <= sigma <= 0.95 | Refuse - trigger secondary monitors | 5.3% |
| **Fatigue (Alert)** | sigma > 0.95 | Sound alarm + visual alert | 12.6% |

> **94.7% high-confidence coverage with 0% false alarm rate.** The 5.3% uncertain zone eliminates ambiguous predictions that would otherwise cause false alarms.

---

## Key Research Contributions

### 1. Pipeline Decomposition - The SSD Bottleneck

By systematically profiling each stage of the inference pipeline, we discovered that **preprocessing, not network capacity, is the dominant accuracy bottleneck**:

- The OpenCV SSD face detector uses a tight bounding box that **clips peripheral facial features** (forehead, jawline, ears)
- This introduces a **domain shift** between training (full faces) and inference (clipped faces)
- Quantified impact: **-12.1 percentage points** of accuracy lost purely from the face detection handoff

### 2. ReLU vs. Hard-Swish Quantization Collapse

MobileNetV3 uses hard-swish activations optimized for GPU inference. Under standard edge quantization:

| Architecture | Activation | Quantized Size | Accuracy | Verdict |
|---|---|---|---|---|
| MobileNetV2 | ReLU | 2.4 MB | **99.0%** | Stable |
| MobileNetV3 | Hard-Swish | 2.3 MB | ~50% | Collapses |

> MobileNetV3 hard-swish activations produce near-random probability distributions (sigma ~ 0.63 for both classes) under dynamic quantization. **ReLU is the correct choice for edge deployment.**

### 3. CLAHE - The +65.7pp Preprocessing Win

| Preprocessing | Accuracy | Notes |
|---|---|---|
| Raw RGB | 33.3% | Fails under mixed lighting |
| **CLAHE** | **99.0%** | Local histogram equalization bridges illumination gap |

> CLAHE normalizes lighting at the local tile level, making the classifier robust to headlights, shadows, and time-of-day variations.

### 4. Subject-Adaptive Few-Shot Calibration

30 frames (1 second) of the driver sitting normally at vehicle startup:

1. Collect 30 active (awake) frames during the first minute of driving
2. Fine-tune the final 5 layers of MobileNetV2 on these frames
3. Calibrate the classification threshold using the model own uncertainty estimates
4. **Zero manual labeling required** - the system assumes the driver is awake at startup

### 5. Test-Time Augmentation (TTA)

Averaging predictions over 3 augmented versions of the same face:

| Method | Accuracy | Stability |
|---|---|---|
| Single prediction | 94.2% | Moderate variance |
| **TTA (3 augmented)** | **96.1%** | Reduced variance |

---

## Quick Start

### Local (Python)

git clone https://github.com/ankushkarmakar/DriveSafe-AI.git
cd DriveSafe-AI
pip install -r requirements.txt
python app.py

Open **http://127.0.0.1:7860** - grant camera access and the system starts monitoring.

### Docker

docker build -t drivesafe-ai .
docker run -p 7860:7860 drivesafe-ai

### Requirements

numpy, ai-edge-litert, opencv-contrib-python-headless<5, fastapi, uvicorn[standard], websockets, scikit-learn

---

## Project Structure

app.py - FastAPI WebSocket server (production)
drowsiness_v2_dynamic.tflite - MobileNetV2 quantized (2.4 MB) - PRODUCTION MODEL
drowsiness_v2.keras - Full Keras checkpoint (Float32, 8.6 MB)
deploy.prototxt - SSD face detector configuration
res10_300x300_ssd_iter_140000.caffemodel - SSD face detector weights
experiments/ - LOSO evaluation scripts (v2/v3/v6/v7)
results/ - CSV outputs per experiment version
DriveSafe_AI_JRTIP.tex - Research paper (LaTeX)

---

## Model Comparison

| Model | Size | In-Distribution | Cross-Subject (LOSO) | Quantization |
|---|---|---|---|---|
| MobileNetV3 | 4.3 MB | ~98% | 59.0% | Collapses (hard-swish) |
| **MobileNetV2** | **2.4 MB** | **99.0%** | **96.1%** (adaptive) | Stable (ReLU) |
| MobileNetV2 (Float32) | 8.6 MB | 98.3% | 96.1% | N/A (no quantization) |

> MobileNetV2 dynamic quantization **compresses by 72%** (8.6 MB -> 2.4 MB) while **improving** validation accuracy to 99.0%.

---

## Citation

@article{karmakar2026drivesafe,
  title={DriveSafe AI: Quantifying the Preprocessing Bottleneck in Lightweight Driver Drowsiness Detection},
  author={Karmakar, Ankush},
  journal={Journal of Real-Time Image Processing},
  year={2026}
}

---

## License

CC BY-NC 4.0 - Non-commercial use only.
