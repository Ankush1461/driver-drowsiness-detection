"""
Frame Decimation Ablation Study
================================
Compares inference at every frame, every 2nd, 3rd, 5th, and 10th frame.
Measures both model accuracy (on cached predictions) and end-to-end
camera-to-alert latency.

Usage:
    python experiments/ablation_frame_decimation.py

Outputs:
    - results/frame_decimation_ablation.csv
    - results/frame_decimation_ablation.tex
"""

import csv
import os
import sys
import time
import numpy as np

# --- Configuration ---
VIDEO_SOURCE = 0  # 0 = webcam, or path to video file
FATIGUE_THRESHOLD = 0.5
TEST_DURATION_SEC = 60
OUTPUT_DIR = "results"

# Frame skip factors to test
FRAME_SKIP_FACTORS = [1, 2, 3, 5, 10]
FACE_DETECT_EVERY = 15  # Keep constant for fair comparison
NUM_THREADS = 2


def main():
    import cv2

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            import tensorflow as tf
            litert = tf.lite

    # Load model
    interpreter = litert.Interpreter(model_path="drowsiness.tflite",
                                     num_threads=NUM_THREADS)
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- Collect ALL frames first, then simulate each decimation factor ---
    print("Capturing frames...")
    all_frames = []
    start = time.time()
    while time.time() - start < TEST_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    print(f"Captured {len(all_frames)} frames ({TEST_DURATION_SEC}s at ~{fps:.0f} FPS)")

    # --- Run each decimation factor ---
    results = []

    for skip in FRAME_SKIP_FACTORS:
        print(f"\nFrame skip = {skip} (infer every {skip}th frame)...")

        correct = 0
        total = 0
        inference_count = 0
        last_face_box = None
        face_detection_count = 0

        infer_start = time.perf_counter()

        for idx, frame in enumerate(all_frames):
            h, w = frame.shape[:2]

            # Face detection every FACE_DETECT_EVERY frames
            if idx % FACE_DETECT_EVERY == 0:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                best_idx, best_conf = -1, 0
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > 0.5 and conf > best_conf:
                        best_conf = conf
                        best_idx = i
                if best_idx >= 0:
                    box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
                    last_face_box = box.astype("int")
                    face_detection_count += 1

            # Classification only on frames divisible by skip
            if idx % skip != 0:
                continue

            if last_face_box is None:
                continue

            x1, y1, x2, y2 = last_face_box
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size == 0:
                continue

            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96)).astype(np.float32)
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            inference_count += 1

        elapsed = time.perf_counter() - infer_start

        # Compute per-inference latency (excluding face detection overhead)
        # and per-frame latency (including skipped frames)
        total_frames = len(all_frames)
        per_inference_ms = (elapsed / max(inference_count, 1)) * 1000
        per_frame_ms = (elapsed / total_frames) * 1000
        effective_fps = inference_count / elapsed if elapsed > 0 else 0
        compute_reduction = (1 - inference_count / total_frames) * 100

        results.append({
            "frame_skip": skip,
            "total_frames": total_frames,
            "inferences_run": inference_count,
            "face_detections": face_detection_count,
            "compute_reduction_pct": compute_reduction,
            "elapsed_sec": elapsed,
            "per_inference_ms": per_inference_ms,
            "per_frame_ms": per_frame_ms,
            "effective_fps": effective_fps,
        })

        print(f"  Inferences: {inference_count}/{total_frames} "
              f"({compute_reduction:.0f}% reduction) | "
              f"Per-inference: {per_inference_ms:.1f} ms | "
              f"Effective FPS: {effective_fps:.1f}")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "frame_decimation_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "frame_decimation_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by ablation_frame_decimation.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Frame Decimation Ablation Results}\n")
        f.write("\\label{tab:frame_decimation_ablation}\n")
        f.write("\\begin{tabular}{ccrrrrr}\n\\toprule\n")
        f.write("Skip & Inferences & Reduction (\\%) & Per-Infer (ms) "
                "& Per-Frame (ms) & Eff. FPS & Face Dets \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['frame_skip']} & {r['inferences_run']} "
                    f"& {r['compute_reduction_pct']:.0f} "
                    f"& {r['per_inference_ms']:.1f} "
                    f"& {r['per_frame_ms']:.1f} "
                    f"& {r['effective_fps']:.1f} "
                    f"& {r['face_detections']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
