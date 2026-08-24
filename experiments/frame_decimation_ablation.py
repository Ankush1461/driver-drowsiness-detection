"""
Frame-Decimation Ablation for DriveSafe AI
===========================================
Simulates the real-time pipeline behavior: inference runs every N-th frame,
carrying forward the last prediction for skipped frames (as in app.py).

Measures:
  - Classification accuracy (all frames use the carried-forward prediction)
  - Effective throughput (frames/sec)
  - Per-frame latency (amortized with face detection every 15 frames)

Decimation factors: 1, 2, 3, 5, 10

The key insight: in a real webcam stream, consecutive frames are nearly
identical. Running the classifier every N-th frame should have minimal
accuracy impact but significant latency reduction.
"""
import os, sys, time, csv
import numpy as np
import tensorflow as tf

IMG_SIZE = (96, 96)
BATCH_SIZE = 64
SEED = 42
FACE_DETECT_EVERY = 15  # Matches app.py
preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    try:
        import tflite_runtime.interpreter as litert
    except ImportError:
        from tensorflow.lite.python.lite import Interpreter as litert

TFLITE_PATH = "drowsiness_robust.tflite"
DATASET_PATH = "external_dataset/akahana"
DECIMATION_FACTORS = [1, 2, 3, 5, 10]

def main():
    os.makedirs("results", exist_ok=True)

    # Load TFLite model
    print(f"Loading TFLite model: {TFLITE_PATH}")
    interpreter = litert.Interpreter(model_path=TFLITE_PATH, num_threads=2)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    # Resize for batch inference
    interpreter.resize_tensor_input(inp[0]["index"], [BATCH_SIZE, *IMG_SIZE, 3])
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    # Load dataset
    print(f"Loading dataset: {DATASET_PATH}")
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False, seed=SEED,
    )
    ds = ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)

    # Collect all images and labels
    all_images = []
    all_labels = []
    for imgs, labels in ds:
        all_images.append(imgs.numpy().astype(np.float32))
        all_labels.append(labels.numpy().flatten())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    total_frames = len(all_images)
    print(f"Total frames: {total_frames}")

    # Get all predictions (ground truth per-frame from batch inference)
    print("\nRunning batch inference to get all per-frame predictions...")
    all_preds_raw = []
    for i in range(0, total_frames, BATCH_SIZE):
        batch = all_images[i:i+BATCH_SIZE]
        bs = batch.shape[0]
        if bs < BATCH_SIZE:
            # Pad last batch to match expected input shape
            pad = np.zeros((BATCH_SIZE - bs, *IMG_SIZE, 3), dtype=np.float32)
            batch = np.concatenate([batch, pad], axis=0)
        interpreter.set_tensor(inp[0]["index"], batch)
        interpreter.invoke()
        preds = interpreter.get_tensor(out[0]["index"])
        all_preds_raw.extend([float(preds[j][0]) for j in range(bs)])
    all_preds_raw = np.array(all_preds_raw)

    # Also measure face detection amortized cost
    # From pipeline_latency.py: face detection = 79.85 ms every 15 frames
    FACE_DET_LATENCY = 79.85  # ms
    FACE_DET_EVERY = 15

    # Run ablation for each decimation factor
    print(f"\n{'='*75}")
    print("FRAME-DECIMATION ABLATION RESULTS")
    print(f"{'='*75}")

    results = []
    for decimation in DECIMATION_FACTORS:
        # Simulate frame-decimation pipeline
        # In real-time: frames come in order, every N-th frame triggers inference
        # Skipped frames carry forward the last prediction
        carried_pred = 0.0  # Default: assume awake (conservative)
        num_inferences = 0
        correct = 0

        all_p = []
        all_l = []

        for frame_idx in range(total_frames):
            if frame_idx % decimation == 0:
                # This frame triggers inference
                carried_pred = all_preds_raw[frame_idx]
                num_inferences += 1

            # Use carried-forward prediction for this frame's decision
            predicted_class = 1 if carried_pred > 0.5 else 0
            true_class = int(all_labels[frame_idx])
            all_p.append(predicted_class)
            all_l.append(true_class)
            if predicted_class == true_class:
                correct += 1

        acc = correct / total_frames
        tp = sum(1 for p, l in zip(all_p, all_l) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_p, all_l) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_p, all_l) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(all_p, all_l) if p == 0 and l == 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        spec = tn / max(tn + fp, 1)

        # Calculate amortized latency per frame
        # Each inference frame costs: face_det (every 15) + classification
        # Face detection is the dominant cost
        classification_cost = 3.88  # ms per inference (from pipeline_latency.py)
        face_det_amortized = FACE_DET_LATENCY / FACE_DET_EVERY  # ~5.32 ms/frame
        classify_amortized = classification_cost * (decimation / FACE_DET_EVERY) if decimation <= FACE_DET_EVERY else classification_cost
        # More accurate: on frames where inference runs, cost = face_det + classify
        # On frames where only face det runs (every 15), cost = face_det
        # On skipped frames, cost = ~0
        # Simplified amortized cost per frame:
        inference_rate = 1.0 / decimation  # fraction of frames that run inference
        amortized_ms = face_det_amortized + classification_cost * inference_rate

        throughput_fps = 1000.0 / amortized_ms if amortized_ms > 0 else float("inf")

        results.append({
            "decimation": decimation,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "specificity": spec,
            "inferences": num_inferences,
            "inference_rate": num_inferences / total_frames,
            "amortized_ms": amortized_ms,
            "throughput_fps": throughput_fps,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

        print(f"\n  Factor {decimation} (infer every {decimation}{'st' if decimation==1 else 'nd' if decimation==2 else 'rd' if decimation==3 else 'th'} frame):")
        print(f"    Accuracy:    {acc*100:.2f}%")
        print(f"    Precision:   {prec:.4f}")
        print(f"    Recall:      {rec:.4f}")
        print(f"    F1:          {f1:.4f}")
        print(f"    Specificity: {spec:.4f}")
        print(f"    Inferences:  {num_inferences}/{total_frames} ({num_inferences/total_frames*100:.1f}%)")
        print(f"    Amortized:   {amortized_ms:.1f} ms/frame ({throughput_fps:.0f} FPS)")

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Factor':>8} {'Acc%':>8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'Inf%':>6} {'ms/fr':>7} {'FPS':>6}")
    print(f"{'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")
    for r in results:
        print(f"{r['decimation']:>8} {r['accuracy']*100:>7.2f}% {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['specificity']:>7.4f} {r['inference_rate']*100:>5.1f}% {r['amortized_ms']:>6.1f} {r['throughput_fps']:>5.0f}")
    print(f"{'='*75}")

    # Key insight: accuracy delta from factor 1 to factor 3
    base = results[0]["accuracy"]
    for r in results:
        delta = (r["accuracy"] - base) * 100
        print(f"  Factor {r['decimation']}: {delta:+.2f} pp accuracy vs. every-frame, {r['throughput_fps']:.0f} FPS")

    # Save CSV
    csv_path = os.path.join("results", "frame_decimation_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decimation", "accuracy", "precision", "recall", "f1",
                                           "specificity", "inferences", "inference_rate",
                                           "amortized_ms", "throughput_fps", "tp", "fp", "fn", "tn"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nResults saved to {csv_path}")

    # Save LaTeX table
    tex_path = os.path.join("results", "frame_decimation_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by frame_decimation_ablation.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Frame-Decimation Ablation: Accuracy vs.\n")
        f.write("Throughput on \\texttt{akahana/Driver-Drowsiness-Dataset}\\,($n = 10{,}000$)}\n")
        f.write("\\label{tab:frame_decimation}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Decimation & Acc. (\\%) & F1 & Inf. Rate & Amort. & Throughput \\\\\n")
        f.write("Factor & & & (\\%) & (ms/frame) & (FPS) \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['decimation']}\\textsuperscript{{{'st' if r['decimation']==1 else 'nd' if r['decimation']==2 else 'rd' if r['decimation']==3 else 'th'}}} & {r['accuracy']*100:.2f} & {r['f1']:.4f} & {r['inference_rate']*100:.0f} & {r['amortized_ms']:.1f} & {r['throughput_fps']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
