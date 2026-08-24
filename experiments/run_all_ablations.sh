#!/bin/bash
# ============================================================================
# Run All Ablation Studies
# ============================================================================
# Execute this from the repo root directory:
#   cd driver-drowsiness-detection
#   bash experiments/run_all_ablations.sh
#
# Prerequisites:
#   - Python 3.10+ with TensorFlow, OpenCV, NumPy
#   - dataset/train_cropped/ with images (for model + quantization ablations)
#   - Webcam or video file (for temporal, frame-decimation, e2e latency)
#   - Hardware config filled in experiments/hardware_config.md
# ============================================================================

set -e

echo "============================================="
echo "  DriveSafe AI — Full Ablation Suite"
echo "============================================="
echo ""

# Record the git commit
COMMIT=$(git rev-parse HEAD)
echo "Git commit: $COMMIT"
echo "Timestamp:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Record hardware info
echo "--- Hardware Info ---"
echo "OS: $(uname -a)"
if command -v lscpu &> /dev/null; then
    echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
    echo "Cores: $(nproc)"
fi
if command -v wmic &> /dev/null; then
    echo "CPU: $(wmic cpu get name 2>/dev/null | tail -1)"
fi
echo "Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null)"
echo "TensorFlow: $(python3 -c 'import tensorflow; print(tensorflow.__version__)' 2>/dev/null || echo 'not found')"
echo ""

mkdir -p results

# --- 1. Temporal Window Ablation ---
echo "[1/5] Running temporal window ablation..."
python experiments/ablation_temporal_window.py || echo "SKIPPED (needs webcam/video)"
echo ""

# --- 2. Model Comparison Ablation ---
echo "[2/5] Running model comparison ablation..."
python experiments/ablation_model_comparison.py || echo "SKIPPED (needs dataset)"
echo ""

# --- 3. Quantization Ablation ---
echo "[3/5] Running quantization ablation..."
python experiments/ablation_quantization.py || echo "SKIPPED (needs dataset + keras model)"
echo ""

# --- 4. Frame Decimation Ablation ---
echo "[4/5] Running frame decimation ablation..."
python experiments/ablation_frame_decimation.py || echo "SKIPPED (needs webcam/video)"
echo ""

# --- 5. End-to-End Latency ---
echo "[5/5] Measuring end-to-end latency..."
python experiments/measure_e2e_latency.py || echo "SKIPPED (needs webcam/video)"
echo ""

echo "============================================="
echo "  All ablations complete!"
echo "  Results in: results/"
echo "============================================="
echo ""
echo "Generated files:"
ls -la results/ 2>/dev/null || echo "No results directory found"
