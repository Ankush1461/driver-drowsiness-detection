"""
Quantization Ablation Study — FP32 vs TFLite Dynamic-Range
==========================================================
Compares the full-precision Keras model against the dynamically quantized
TFLite artifact on the akahana/Driver-Drowsiness-Dataset (external benchmark).

Usage:
    python experiments/run_quantization_ablation.py

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
DATASET_PATH = "external_dataset/akahana"
SEED = 123
VAL_SPLIT = 0.2
NUM_THREADS = 4
NUM_WARMUP = 10
OUTPUT_DIR = "results"

KERAS_MODEL_PATH = "drowsiness.keras"
TFLITE_MODEL_PATH = "drowsiness.tflite"

def evaluate_keras(keras_path, val_ds):
    """Evaluate the full Keras model (float32)."""
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    start = time.perf_counter()
    for images, labels in val_ds:
        preds = model(images, training=False)
        labels_np = labels.numpy().flatten()
        preds_np = preds.numpy().flatten()
        for i in range(images.shape[0]):
            pred_class = 1 if preds_np[i] > 0.5 else 0
            true_class = int(labels_np[i])
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
    warmup = np.zeros((1, 96, 96, 3), dtype=np.float32)
    for _ in range(NUM_WARMUP):
        interpreter.set_tensor(input_details[0]['index'], warmup)
        interpreter.invoke()

    start = time.perf_counter()
    for images, labels in val_ds:
        bs = images.shape[0]
        preprocessed = images.numpy().astype(np.float32)
        labels_np = labels.numpy().flatten()

        # Resize if model expects different batch size
        expected_bs = input_details[0]['shape'][0]
        if expected_bs != bs and expected_bs != -1:
            try:
                interpreter.resize_tensor_input(input_details[0]['index'],
                                                [bs, 96, 96, 3])
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            except Exception:
                # Fall back to single-image inference
                for i in range(bs):
                    single = preprocessed[i:i+1]
                    interpreter.set_tensor(input_details[0]['index'], single)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    pred_class = 1 if pred[0][0] > 0.5 else 0
                    true_class = int(labels_np[i])
                    all_preds.append(pred_class)
                    all_labels.append(true_class)
                    if pred_class == true_class:
                        correct += 1
                    total += 1
                continue

        interpreter.set_tensor(input_details[0]['index'], preprocessed)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        for i in range(bs):
            pred_class = 1 if preds[i][0] > 0.5 else 0
            true_class = int(labels_np[i])
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
    specificity = tn / max(tn + fp, 1)

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
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
    class_names = val_ds.class_names
    print(f"Classes: {class_names}")
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)

    # Count samples
    val_count = val_ds.cardinality().numpy()
    print(f"Validation batches: {val_count}")

    results = []

    # 1. Float32 Keras
    if os.path.exists(KERAS_MODEL_PATH):
        print(f"\n{'='*60}")
        print(f"Evaluating Keras float32 model ({KERAS_MODEL_PATH})...")
        keras_size_mb = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
        print(f"  Model size: {keras_size_mb:.1f} MB")
        m = evaluate_keras(KERAS_MODEL_PATH, val_ds)
        m["format"] = "Keras float32"
        m["model_size_mb"] = keras_size_mb
        results.append(m)
        print(f"  Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        print(f"  Latency:   {m['latency_ms']:.2f} ms/image")
        print(f"  Throughput: {m['throughput']:.1f} images/s")
        print(f"  Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    else:
        print(f"\nKeras model not found at {KERAS_MODEL_PATH}")

    # 2. Dynamic-range quantized TFLite
    if os.path.exists(TFLITE_MODEL_PATH):
        print(f"\n{'='*60}")
        print(f"Evaluating TFLite dynamic-range model ({TFLITE_MODEL_PATH})...")
        tflite_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
        print(f"  Model size: {tflite_size_mb:.1f} MB")
        m = evaluate_tflite(TFLITE_MODEL_PATH, val_ds)
        m["format"] = "TFLite dynamic-range"
        m["model_size_mb"] = tflite_size_mb
        results.append(m)
        print(f"  Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        print(f"  Latency:   {m['latency_ms']:.2f} ms/image")
        print(f"  Throughput: {m['throughput']:.1f} images/s")
        print(f"  Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    else:
        print(f"\nTFLite model not found at {TFLITE_MODEL_PATH}")

    if not results:
        print("\nNo model files found.")
        return

    # --- Delta ---
    if len(results) == 2:
        acc_delta = results[1]["accuracy"] - results[0]["accuracy"]
        lat_speedup = results[0]["latency_ms"] / max(results[1]["latency_ms"], 0.01)
        size_ratio = results[0]["model_size_mb"] / max(results[1]["model_size_mb"], 0.01)

        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Model size:    {results[0]['model_size_mb']:.1f} MB -> {results[1]['model_size_mb']:.1f} MB ({size_ratio:.1f}x reduction)")
        print(f"  Accuracy:      {results[0]['accuracy']*100:.2f}% -> {results[1]['accuracy']*100:.2f}% ({acc_delta*100:+.2f} pp)")
        print(f"  F1:            {results[0]['f1']:.4f} -> {results[1]['f1']:.4f}")
        print(f"  Latency:       {results[0]['latency_ms']:.2f} ms -> {results[1]['latency_ms']:.2f} ms ({lat_speedup:.2f}x speedup)")
        print(f"  Throughput:    {results[0]['throughput']:.1f} -> {results[1]['throughput']:.1f} images/s")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "quantization_ablation.csv")
    fieldnames = ["format", "model_size_mb", "accuracy", "precision", "recall",
                  "f1", "specificity", "latency_ms", "throughput", "total",
                  "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "quantization_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by run_quantization_ablation.py\n")
        f.write("% Dataset: akahana/Driver-Drowsiness-Dataset\n")
        f.write("% External benchmark — model was NOT trained on this data\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Quantization Ablation: FP32 Keras vs.\n")
        f.write("Dynamic-Range TFLite on an External DDD Benchmark\n")
        f.write("(\\texttt{akahana/Driver-Drowsiness-Dataset})}\n")
        f.write("\\label{tab:quantization_ablation}\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("Format & Size & Acc. & Prec. & Rec. & F1 & Spec. & Lat. \\\\\n")
        f.write(" & (MB) & (\\%) & & & & & (ms) \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['format']} & {r['model_size_mb']:.1f} & {r['accuracy']*100:.2f} "
                    f"& {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} "
                    f"& {r['specificity']:.3f} & {r['latency_ms']:.1f} \\\\\n")
        if len(results) == 2:
            f.write("\\midrule\n")
            acc_d = (results[1]["accuracy"] - results[0]["accuracy"]) * 100
            lat_s = results[0]["latency_ms"] / max(results[1]["latency_ms"], 0.01)
            size_r = results[0]["model_size_mb"] / max(results[1]["model_size_mb"], 0.01)
            summary = (f"Size reduction: {size_r:.1f}x | "
                       f"Acc. Delta: {acc_d:+.2f} pp | "
                       f"Latency speedup: {lat_s:.2f}x")
            line = '\multicolumn{8}{l}{' + summary + '} \\\n'
            f.write(line)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
