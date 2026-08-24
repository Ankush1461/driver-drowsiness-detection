"""
End-to-End Latency Measurement
================================
Measures camera-to-alert latency separately from batched model inference.
This script runs the full pipeline (capture → face detect → classify →
temporal check → alert decision) and times each component.

Usage:
    python experiments/measure_e2e_latency.py

Outputs:
    - results/e2e_latency.csv
    - results/e2e_latency.tex
    - Console summary with per-component breakdown
"""

import csv
import os
import sys
import time
import numpy as np

# --- Configuration ---
VIDEO_SOURCE = 0
FACE_DETECT_EVERY = 15
INFER_EVERY = 3
FATIGUE_THRESHOLD = 0.5
TEMPORAL_WINDOW_SEC = 3.0
TEST_DURATION_SEC = 60
NUM_THREADS = 2
OUTPUT_DIR = "results"


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

    interpreter = litert.Interpreter(model_path="drowsiness.tflite",
                                     num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                         "res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {VIDEO_SOURCE}")
        sys.exit(1)

    fps_stream = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Timing accumulators
    face_detect_times = []
    classify_times = []
    e2e_times = []  # Full frame processing time
    total_frames = 0

    last_face_box = None
    consecutive_fatigue = 0
    alert_active = False
    frame_count = 0

    print(f"Measuring latency for {TEST_DURATION_SEC}s at ~{fps_stream:.0f} FPS...")
    print(f"  Face detect every {FACE_DETECT_EVERY} frames")
    print(f"  Classification every {INFER_EVERY} frames")
    print(f"  Temporal window: {TEMPORAL_WINDOW_SEC}s")
    print()

    start_stream = time.perf_counter()

    while True:
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        classified_this_frame = False
        pred_prob = None

        # --- Face Detection ---
        if frame_count % FACE_DETECT_EVERY == 0:
            t_fd_start = time.perf_counter()
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
            face_detect_times.append(time.perf_counter() - t_fd_start)

        # --- Classification ---
        if frame_count % INFER_EVERY == 0 and last_face_box is not None:
            t_cls_start = time.perf_counter()
            x1, y1, x2, y2 = last_face_box
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size > 0:
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (96, 96)).astype(np.float32)
                img = np.expand_dims(img, axis=0)
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                pred_prob = float(pred[0][0])
                classified_this_frame = True
            classify_times.append(time.perf_counter() - t_cls_start)

        # --- Temporal Confirmation ---
        if classified_this_frame and pred_prob is not None:
            if pred_prob > FATIGUE_THRESHOLD:
                consecutive_fatigue += 1
            else:
                consecutive_fatigue = 0

        # --- End-to-End (this frame) ---
        t_e2e = time.perf_counter() - t_frame_start
        e2e_times.append(t_e2e)

        total_frames += 1
        frame_count += 1

        # Stop after duration
        if time.perf_counter() - start_stream > TEST_DURATION_SEC:
            break

    cap.release()
    stream_elapsed = time.perf_counter() - start_stream

    # --- Aggregate ---
    e2e_ms = np.array(e2e_times) * 1000
    fd_ms = np.array(face_detect_times) * 1000 if face_detect_times else np.array([0])
    cls_ms = np.array(classify_times) * 1000 if classify_times else np.array([0])

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "std": float(np.std(arr)),
        }

    e2e_s = stats(e2e_ms)
    fd_s = stats(fd_ms)
    cls_s = stats(cls_ms)

    summary = {
        "total_frames": total_frames,
        "stream_duration_sec": stream_elapsed,
        "effective_fps": total_frames / stream_elapsed,
        "face_detect_count": len(face_detect_times),
        "classify_count": len(classify_times),
        "e2e_mean_ms": e2e_s["mean"],
        "e2e_median_ms": e2e_s["median"],
        "e2e_p95_ms": e2e_s["p95"],
        "e2e_p99_ms": e2e_s["p99"],
        "face_detect_mean_ms": fd_s["mean"],
        "face_detect_p95_ms": fd_s["p95"],
        "classify_mean_ms": cls_s["mean"],
        "classify_p95_ms": cls_s["p95"],
    }

    # --- Console Report ---
    print("\n" + "=" * 60)
    print("END-TO-END LATENCY MEASUREMENT RESULTS")
    print("=" * 60)
    print(f"Total frames:       {total_frames}")
    print(f"Stream duration:    {stream_elapsed:.1f}s")
    print(f"Effective FPS:      {summary['effective_fps']:.1f}")
    print()
    print("Component Latency (ms):")
    print(f"  Face Detection:   mean={fd_s['mean']:.2f}  "
          f"median={fd_s['median']:.2f}  p95={fd_s['p95']:.2f}")
    print(f"  Classification:   mean={cls_s['mean']:.2f}  "
          f"median={cls_s['median']:.2f}  p95={cls_s['p95']:.2f}")
    print()
    print("End-to-End Frame Latency (ms):")
    print(f"  Mean:    {e2e_s['mean']:.2f}")
    print(f"  Median:  {e2e_s['median']:.2f}")
    print(f"  P95:     {e2e_s['p95']:.2f}")
    print(f"  P99:     {e2e_s['p99']:.2f}")
    print(f"  Std:     {e2e_s['std']:.2f}")
    print()
    print("NOTE: The 104.47 ms/image benchmark latency from evaluate_model.py")
    print("is batched inference only. This measurement includes capture, face")
    print("detection, preprocessing, inference, temporal check, and response.")
    print("=" * 60)

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "e2e_latency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "e2e_latency.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by measure_e2e_latency.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{End-to-End Latency Measurement}\n")
        f.write("\\label{tab:e2e_latency}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Component & Mean (ms) & P95 (ms) \\\\\n\\midrule\n")
        f.write(f"Face Detection & {fd_s['mean']:.2f} & {fd_s['p95']:.2f} \\\\\n")
        f.write(f"Classification & {cls_s['mean']:.2f} & {cls_s['p95']:.2f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{End-to-End Frame}} & \\textbf{{{e2e_s['mean']:.2f}}} "
                f"& \\textbf{{{e2e_s['p95']:.2f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
