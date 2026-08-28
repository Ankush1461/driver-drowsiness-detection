---
title: DriveSafe AI
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: true
---

# 🚗 DriveSafe AI: Real-Time Driver Drowsiness Detection

[![GitHub Repo](https://img.shields.io/badge/GitHub-Ankush1461%2Fdriver--drowsiness--detection-181717?style=for-the-badge&logo=github)](https://github.com/Ankush1461/driver-drowsiness-detection)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Live%20Demo-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.style=for-the-badge)](LICENSE)

> **99.0% In-Distribution Accuracy** | **2.4 MB Quantized Model** | **150+ FPS** | **Sub-40ms End-to-End Latency**

**DriveSafe AI** is a lightweight, edge-optimized driver drowsiness detection system engineered to run real-time inference **without dedicated GPU hardware**. Operating efficiently on standard consumer CPU architectures (e.g., 2-vCPU Intel), the pipeline incorporates face detection, localized lighting normalization (CLAHE), neural classification via dynamic quantized MobileNetV2, temporal smoothing, and uncertainty-aware three-zone alerting.

---

## 📋 Table of Contents
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Evaluation Results Across Scenarios](#-evaluation-results-across-scenarios)
  - [Scenario 1: In-Distribution Evaluation](#scenario-1-in-distribution-single-dataset-split)
  - [Scenario 2: Cross-Subject Baseline (True LOSO)](#scenario-2-cross-subject-baseline-true-loso-no-adaptation)
  - [Scenario 3: Cross-Subject with Personalization (LOSO + Adaptive Calibration)](#scenario-3-cross-subject-with-personalization-true-loso--adaptive-calibration)
  - [Per-Subject Performance Breakdown](#per-subject-breakdown-adaptive-v7)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Three-Zone Confidence Protocol](#-three-zone-confidence-protocol)
- [Key Engineering Insights](#-key-engineering-insights)
- [Performance Profile](#-performance-profile)
- [Model Benchmark & Comparison](#-model-benchmark--comparison)
- [Quick Start & Deployment](#-quick-start--deployment)
  - [Local Installation](#1-local-python-setup)
  - [Docker Containerization](#2-docker-deployment)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🌐 Live Demo

- ⚡ **[Try the Live Interactive Demo](https://huggingface.co/spaces/ankushkarmakar/drivesafe-ai-v2)**: Grant camera access to monitor drowsiness in real time in your web browser.

---

## ✨ Key Features

- 🏎️ **Ultra-Low Latency & High FPS**: Sub-40ms end-to-end pipeline execution delivering 150+ FPS on edge CPUs.
- 📱 **Compact Edge Deployment**: Dynamic quantized MobileNetV2 reduced to **2.4 MB** (72% size reduction from 8.6 MB Float32).
- 💡 **Illumination-Robust (CLAHE)**: +65.7 percentage points accuracy boost under harsh lighting variations and shadows.
- 🎯 **Few-Shot Personalization**: 30-frame (1 second) zero-label startup calibration lifts Leave-One-Subject-Out (LOSO) cross-subject accuracy from **53.2% to 96.1%**.
- 🛡️ **Uncertainty-Aware Alerting**: Three-zone confidence filtering delivers 94.7% high-confidence coverage with zero false alarms on ambiguous frames.

---

## 📊 Evaluation Results Across Scenarios

DriveSafe AI was evaluated across three rigorous evaluation scenarios to benchmark real-world cross-subject generalization.

### Scenario 1: In-Distribution (Single Dataset Split)

Standard train/test split on homogeneous subject data.

| Metric | MobileNetV2 + CLAHE |
| :--- | :---: |
| **Accuracy** | **99.0%** |
| **Precision** | **98.8%** |
| **Recall** | **99.2%** |
| **F1 Score** | **0.990** |
| **Model Size** | **2.4 MB** *(Dynamic Quantized TFLite)* |
| **Inference Time** | **1.2ms / frame** |

> *Note: This represents the model running in the live interactive web demo.*

---

### Scenario 2: Cross-Subject Baseline (True LOSO, No Adaptation)

Leave-One-Subject-Out (LOSO) cross-validation across 24 distinct subjects. **The model never saw the target subject during training.**

| Metric | Baseline (v2) |
| :--- | :---: |
| **Accuracy** | **53.2% ± 19.3%** |
| **Precision** | **34.3%** |
| **Recall** | **41.4%** |
| **F1 Score** | **0.300** |
| **False Alarm Rate** | **40.9%** |

> *Insight: Uncalibrated deep neural networks suffer severe domain shifts across different subject face structures.*

---

### Scenario 3: Cross-Subject with Personalization (True LOSO + Adaptive Calibration)

Same true LOSO protocol paired with a **30-frame (1-second) passive calibration** step at startup (active driver state assumed).

| Metric | Baseline (v2) | Adaptive (v7) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 53.2% ± 19.3% | **96.1% ± 4.9%** | **+42.9 pp** |
| **Precision** | 34.3% | **91.2%** | **+56.9 pp** |
| **Recall** | 41.4% | **99.2%** | **+57.8 pp** |
| **F1 Score** | 0.300 | **0.947** | **+0.647** |
| **False Alarm Rate** | 40.9% | **5.5%** | **-35.4 pp** |
| **Video Accuracy** | 61.1% | **61.1%** | — |

**Pooled Confusion Matrix (24 subjects):**
- True Positives (TP): `1,191` | False Positives (FP): `131`
- False Negatives (FN): `10` | True Negatives (TN): `2,268`

---

### Per-Subject Breakdown (Adaptive, v7)

| Accuracy Range | Subject Identifiers | Subject Count |
| :---: | :--- | :---: |
| **95% – 100%** | S02, S03, S07, S09, S10, S11, S13, S15, S16, S18, S19, S20, S21, S22, S24 | **15** |
| **90% – 95%** | S01, S04, S05, S06, S08, S12, S14, S17, S23 | **9** |
| **< 90%** | None | **0** |

> *All 24 test subjects achieve >90.0% individual accuracy, with 15 achieving >95.0%.*

---

## 🏗️ System Architecture & Pipeline

```
[ Webcam Feed (320x240) ] 
          │
          ▼  (WebSocket stream)
[ OpenCV SSD Face Detection ] ──► (Amortized: Execution every 15th frame ~17ms)
          │
          ▼
[ CLAHE Preprocessing ] ────────► (Contrast Limited Adaptive Histogram Equalization)
          │
          ▼
[ MobileNetV2 TFLite Engine ] ──► (Inference every 3rd frame, 1.2ms latency)
          │
          ▼
[ Temporal Buffer (3 Sec) ] ────► (Sliding window noise filtering)
          │
          ▼
[ Uncertainty-Aware Alerting ] ─► (Active / Uncertain / Fatigue Alarm)
```

---

## 🛡️ Three-Zone Confidence Protocol

To mitigate dangerous false alarms and handle ambiguous frames, predictions are divided into three distinct operational zones:

| Operational Zone | Threshold ($\sigma$) | System Action | Target Coverage |
| :--- | :---: | :--- | :---: |
| 🟢 **Active (Safe)** | $\sigma < 0.05$ | Driver alert and responsive | **82.1%** |
| 🟡 **Uncertain** | $0.05 \le \sigma \le 0.95$ | Defer classification; request secondary monitoring | **5.3%** |
| 🔴 **Fatigue (Alert)** | $\sigma > 0.95$ | Trigger visual warning & auditory alarm | **12.6%** |

---

## 💡 Key Engineering Insights

1. **Quantifying the SSD Detection Bottleneck**:
   - OpenCV SSD face bounding box tightly clips peripheral features (forehead, ears, jawline).
   - Induces an unintended domain mismatch between full-face training and clipped-face inference.
   - Identified a **12.1 percentage point loss** attributable solely to face detection handoffs.

2. **ReLU vs. Hard-Swish Quantization Collapse**:
   - MobileNetV3 hard-swish activation functions suffer severe degradation under dynamic 8-bit quantization ($\approx 50\%$ accuracy / near-random output variance).
   - MobileNetV2 with **ReLU** remains stable and maintains **99.0%** accuracy post-quantization.

3. **CLAHE Illumination Robustness**:
   - Raw RGB input drops to **33.3% accuracy** under poor lighting.
   - Local histogram equalization via **CLAHE** yields a **+65.7 pp increase** to achieve **99.0% accuracy**.

4. **Subject-Adaptive Few-Shot Calibration**:
   - Fine-tunes top 5 layers of MobileNetV2 using 30 frames captured during first minute of driving.
   - Fully automated zero-label calibration.

5. **Test-Time Augmentation (TTA)**:
   - Evaluates multi-crop/augmented frames per face to increase stability ($94.2\% \rightarrow 96.1\%$).

---

## ⏱️ Performance Profile

| Operational Metric | Measured Value |
| :--- | :--- |
| **End-to-End Latency** | **< 40ms** (2-vCPU Intel Laptop CPU) |
| **Throughput** | **150+ FPS** |
| **Quantized Model Size** | **2.4 MB** *(72% compression from 8.6 MB Float32)* |
| **Classification Inference** | **1.2ms / frame** |
| **Face Detection Cost** | **~17ms** *(Amortized every 15th frame)* |

---

## 🔬 Model Benchmark & Comparison

| Architecture | Model Size | In-Distribution Accuracy | Cross-Subject (LOSO) Accuracy | Quantization Stability |
| :--- | :---: | :---: | :---: | :---: |
| **MobileNetV3** | 4.3 MB | ~98.0% | 59.0% | Collapses (Hard-Swish) |
| **MobileNetV2 (Float32)** | 8.6 MB | 98.3% | 96.1% | N/A |
| **MobileNetV2 (Dynamic Quantized)** | **2.4 MB** | **99.0%** | **96.1% (Adaptive)** | **Stable (ReLU)** |

---

## 🚀 Quick Start & Deployment

### Prerequisites

- **Python**: Python 3.10 or higher recommended.
- **System Libraries** (Linux only): `libgl1`, `libglib2.0-0` (for OpenCV rendering).
- **Webcam**: Standard USB or integrated laptop webcam.

---

### 1. Local Python Setup

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Ankush1461/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

#### Step 2: Create & Activate Virtual Environment

- **On Windows (PowerShell / CMD)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Important Dependency Note**: OpenCV 5.0+ dropped legacy Caffe model support (`cv2.dnn.readNetFromCaffe`). The `requirements.txt` pins `opencv-contrib-python-headless<5` to ensure SSD face detection models load correctly.

#### Step 4: Launch the Production Server

You can launch the server using either method below:

- **Method A: Direct Script Execution**
  ```bash
  python app.py
  ```

- **Method B: Uvicorn ASGI Server**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 7860
  ```

#### Step 5: Open Web Interface
Open your web browser and navigate to:
```text
http://127.0.0.1:7860  or  http://localhost:7860
```
Grant webcam access when prompted to begin real-time drowsiness monitoring via WebSocket.

---

### 2. Docker Deployment

The included `Dockerfile` builds a lightweight containerized environment with all system dependencies and OpenCV/TFLite bindings pre-configured.

#### Step 1: Build the Container Image
```bash
docker build -t drivesafe-ai .
```

#### Step 2: Run the Container

- **Foreground (Interactive Logs)**:
  ```bash
  docker run -p 7860:7860 drivesafe-ai
  ```

- **Background (Detached Mode)**:
  ```bash
  docker run -d -p 7860:7860 --name drivesafe-app drivesafe-ai
  ```

#### Step 3: Access & Manage Container
- **Access Web Interface**: Visit `http://localhost:7860` in your web browser.
- **View Container Logs** (Background mode): `docker logs -f drivesafe-app`
- **Stop Container** (Background mode): `docker stop drivesafe-app`

---



## 📁 Project Directory Structure

```
driver-drowsiness-detection/
├── app.py                                   # FastAPI WebSocket production server
├── drowsiness_v2_dynamic.tflite             # Production quantized model (2.4 MB)
├── drowsiness_v2.keras                      # Float32 model checkpoint (8.6 MB)
├── deploy.prototxt                          # OpenCV SSD face detector config
├── res10_300x300_ssd_iter_140000.caffemodel # OpenCV SSD face detector weights
├── clean_dataset.py                         # Dataset cleaning utility
├── evaluate_model.py                        # Model evaluation harness
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Container build configuration
├── experiments/                             # LOSO evaluation scripts (v2, v3, v6, v7)
└── results/                                 # Experiment outputs & benchmark metrics
```

---

## 📄 License

This project is distributed under the **CC BY-NC 4.0 License** (Creative Commons Attribution-NonCommercial 4.0 International). Free for personal, academic, and non-commercial usage.


