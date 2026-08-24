"""
Quantization Ablation Study
============================
Compares the floating-point Keras model against the dynamically quantized
TFLite artifact. Measures accuracy difference and latency on the deployment CPU.

Usage:
    python experiments/ablation_quantization.py

Outputs:
    - results/quantization_ablation.csv
    - results/quantization_ablation.tex
"""

import csv
import os
import sys
import time
import numpy as np

# --- Configuration ---
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
DATASET_PATH = "dataset/train_cropped"
SEED = 123
VAL_SPLIT = 0.2
NUM_THREADS = 2
NUM_WARMUP = 5
OUTPUT_DIR = "results"

KERAS_MODEL_PATH = "drowsiness.keras"
TFLITE_MODEL_PATH = "drowsiness.tflite"


def evaluate_keras(keras_path, val_ds):
    """Evaluate the full Keras model (float32)."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    start = time.perf_counter()
    for images, labels in val_ds:
        preds = model(images, training=False)
        for i in range(images.shape[0]):
            pred_class = 1 if preds[i][0] > 0.5 else 0
            true_class = int(labels.numpy()[i])
            all_preds.append(pred_class)
            all_labels.append(true_class)
            if pred_class == true_class:
                correct += 1
            total += 1
    elapsed = time.perf_counter() - start

    return _compute_metrics(correct, total, elapsed, all_preds, all_labels)


def evaluate_tflite(tflite_path, val_ds, num_threads=NUM_THREADS):
    """Evaluate a TFLite model."""
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            import tensorflow as tf
            litert = tf.lite

    interpreter = litert.Interpreter(model_path=tflite_path,
                                     num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    # Warmup
    warmup = np.zeros((BATCH_SIZE, 96, 96, 3), dtype=np.float32)
    for _ in range(NUM_WARMUP):
        interpreter.set_tensor(input_details[0]['index'], warmup)
        interpreter.invoke()

    start = time.perf_counter()
    for images, labels in val_ds:
        bs = images.shape[0]
        if bs != BATCH_SIZE:
            interpreter.resize_tensor_input(input_details[0]['index'],
                                            [bs, 96, 96, 3])
            interpreter.allocate_tensors()

        preprocessed = images.numpy().astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], preprocessed)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        for i in range(bs):
            pred_class = 1 if preds[i][0] > 0.5 else 0
            true_class = int(labels.numpy()[i])
            all_preds.append(pred_class)
            all_labels.append(true_class)
            if pred_class == true_class:
                correct += 1
            total += 1
    elapsed = time.perf_counter() - start

    return _compute_metrics(correct, total, elapsed, all_preds, all_labels)


def _compute_metrics(correct, total, elapsed, all_preds, all_labels):
    accuracy = correct / total
    latency_ms = (elapsed / total) * 1000
    throughput = total / elapsed

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    import tensorflow as tf

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)

    results = []

    # 1. Float32 Keras
    if os.path.exists(KERAS_MODEL_PATH):
        print(f"\nEvaluating Keras float32 model ({KERAS_MODEL_PATH})...")
        keras_size_mb = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
        m = evaluate_keras(KERAS_MODEL_PATH, val_ds)
        m["format"] = "Keras float32"
        m["model_size_mb"] = keras_size_mb
        results.append(m)
        print(f"  Accuracy: {m['accuracy']:.4f} | Size: {keras_size_mb:.1f} MB | Latency: {m['latency_ms']:.1f} ms")

    # 2. Dynamic-range quantized TFLite
    if os.path.exists(TFLITE_MODEL_PATH):
        print(f"\nEvaluating TFLite dynamic-range model ({TFLITE_MODEL_PATH})...")
        tflite_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
        m = evaluate_tflite(TFLITE_MODEL_PATH, val_ds)
        m["format"] = "TFLite dynamic-range"
        m["model_size_mb"] = tflite_size_mb
        results.append(m)
        print(f"  Accuracy: {m['accuracy']:.4f} | Size: {tflite_size_mb:.1f} MB | Latency: {m['latency_ms']:.1f} ms")

    if not results:
        print("No model files found. Run drowsiness.py first.")
        return

    # --- Delta ---
    if len(results) == 2:
        acc_delta = results[1]["accuracy"] - results[0]["accuracy"]
        lat_speedup = results[0]["latency_ms"] / max(results[1]["latency_ms"], 0.01)
        size_ratio = results[0]["model_size_mb"] / max(results[1]["model_size_mb"], 0.01)
        print(f"\n--- Delta ---")
        print(f"Accuracy change: {acc_delta:+.4f}")
        print(f"Size reduction:  {size_ratio:.1f}x")
        print(f"Speedup:         {lat_speedup:.2f}x")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "quantization_ablation.csv")
    fieldnames = ["format", "model_size_mb", "accuracy", "precision", "recall",
                  "f1", "latency_ms", "throughput", "total"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "quantization_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by ablation_quantization.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Quantization Ablation Results}\n")
        f.write("\\label{tab:quantization_ablation}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Format & Size (MB) & Acc. & Prec. & Rec. & F1 & Lat. (ms) \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['format']} & {r['model_size_mb']:.1f} & {r['accuracy']:.4f} "
                    f"& {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} "
                    f"& {r['latency_ms']:.1f} \\\\\n")
        if len(results) == 2:
            f.write("\\midrule\n")
            acc_d = results[1]["accuracy"] - results[0]["accuracy"]
            lat_s = results[0]["latency_ms"] / max(results[1]["latency_ms"], 0.01)
            f.write(f"\\multicolumn{{7}}{{l}}{{{{Accuracy $\\Delta$: {acc_d:+.4f}}}}}"
                    f" & Speedup: {lat_s:.2f}x \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
