"""
Temporal Confirmation Window Ablation Study
============================================
Compares different temporal confirmation windows against frame-level decisions.

Usage:
    python experiments/ablation_temporal_window.py

Requirements:
    - drowsiness.tflite in the repo root
    - A test video or webcam feed (set VIDEO_SOURCE below)
    - The deployed app.py server running (for full pipeline test)

Outputs:
    - results/temporal_ablation.csv
    - results/temporal_ablation.tex (LaTeX-ready table)
"""

import csv
import os
import sys
import time
import numpy as np

# --- Configuration ---
VIDEO_SOURCE = 0  # 0 = webcam, or path to video file
FRAME_SKIP_FACE = 15      # Face detection every Nth frame
FRAME_SKIP_INFER = 3      # Classification every Nth frame
FATIGUE_THRESHOLD = 0.5   # Sigmoid threshold
TEST_DURATION_SEC = 60    # How long to run each configuration
OUTPUT_DIR = "results"

# Temporal windows to test (in seconds)
TEMPORAL_WINDOWS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Approximate FPS of the webcam/input stream (adjust to your setup)
STREAM_FPS = 30


def simulate_temporal_logic(predictions, temporal_window_sec, fps=STREAM_FPS):
    """
    Simulate temporal confirmation on a sequence of frame-level predictions.

    Args:
        predictions: list of (timestamp_sec, fatigued: bool)
        temporal_window_sec: seconds of sustained fatigue before alert
        fps: frames per second of the stream

    Returns:
        dict with metrics
    """
    if temporal_window_sec == 0.0:
        # Frame-level: alert on every fatigue frame
        alerts = [p[1] for p in predictions]
        total_frames = len(predictions)
        true_positives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                           if fatigued and alerts[i])
        false_positives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                            if not fatigued and alerts[i])
        false_negatives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                            if fatigued and not alerts[i])
        true_negatives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                           if not fatigued and alerts[i] == False)
    else:
        window_frames = int(temporal_window_sec * fps)
        alerts = []
        consecutive_fatigue = 0
        drowsy_since = None
        alert_active = False

        for ts, fatigued in predictions:
            if fatigued:
                consecutive_fatigue += 1
                if consecutive_fatigue >= window_frames and not alert_active:
                    alert_active = True
                    drowsy_since = ts
            else:
                consecutive_fatigue = 0
                alert_active = False
                drowsy_since = None
            alerts.append(alert_active)

        total_frames = len(predictions)
        true_positives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                           if fatigued and alerts[i])
        false_positives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                            if not fatigued and alerts[i])
        false_negatives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                            if fatigued and not alerts[i])
        true_negatives = sum(1 for i, (ts, fatigued) in enumerate(predictions)
                           if not fatigued and not alerts[i])

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    false_alarm_rate = false_positives / max(false_positives + true_negatives, 1)
    miss_rate = false_negatives / max(false_negatives + true_positives, 1)

    return {
        "temporal_window_sec": temporal_window_sec,
        "total_frames": total_frames,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            import tensorflow as tf
            litert = tf.lite

    import cv2

    model_path = "drowsiness.tflite"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run from repo root.")
        sys.exit(1)

    interpreter = litert.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Face detector
    face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                         "res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {VIDEO_SOURCE}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or STREAM_FPS

    # --- Collect frame-level predictions ---
    print("Collecting frame-level predictions...")
    frame_predictions = []
    frame_count = 0
    start = time.time()

    while time.time() - start < TEST_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        fatigued = False

        # Periodic face detection
        if frame_count % FRAME_SKIP_FACE == 0:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                          1.0, (300, 300),
                                          (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            best_idx = -1
            best_conf = 0
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.5 and conf > best_conf:
                    best_conf = conf
                    best_idx = i
            if best_idx >= 0:
                box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size > 0:
                    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (96, 96)).astype(np.float32)
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_details[0]['index'], img)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    fatigued = float(pred[0][0]) > FATIGUE_THRESHOLD

        frame_predictions.append((frame_count / fps, fatigued))
        frame_count += 1

    cap.release()
    print(f"Collected {len(frame_predictions)} frames over {TEST_DURATION_SEC}s")

    # --- Run ablation ---
    results = []
    for window in TEMPORAL_WINDOWS:
        metrics = simulate_temporal_logic(frame_predictions, window)
        results.append(metrics)
        print(f"  Window {window:4.1f}s: P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
              f"FAR={metrics['false_alarm_rate']:.3f}")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "temporal_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX table ---
    tex_path = os.path.join(OUTPUT_DIR, "temporal_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by ablation_temporal_window.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Temporal Window Ablation Results}\n")
        f.write("\\label{tab:temporal_ablation}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Window (s) & Precision & Recall & F1 & FAR & Miss Rate & Delay (s) \\\\\n\\midrule\n")
        for r in results:
            delay = r["temporal_window_sec"]
            f.write(f"{delay:4.1f} & {r['precision']:.3f} & {r['recall']:.3f} "
                    f"& {r['f1']:.3f} & {r['false_alarm_rate']:.3f} "
                    f"& {r['miss_rate']:.3f} & {delay:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
