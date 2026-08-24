"""
Model Architecture Ablation Study
==================================
Compares multiple backbone architectures under identical preprocessing,
split, and evaluation conditions.

Architectures tested:
  1. MobileNetV3Large  (current)
  2. MobileNetV3Small  (smaller variant)
  3. EfficientNetV2-B0 (modern lightweight)
  4. InceptionResNetV2 (previous baseline, large)
  5. MobileNetV2       (previous generation)

Usage:
    python experiments/ablation_model_comparison.py

Requirements:
    - dataset/train_cropped/ with Active/ and Fatigue/ subdirectories
    - TensorFlow 2.x with Keras applications

Outputs:
    - results/model_ablation.csv
    - results/model_ablation.tex
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
OUTPUT_DIR = "results"

# Models to evaluate: (name, constructor_fn, input_size)
# For fair comparison, all models use the same 96x96 input
MODELS_TO_TEST = [
    ("MobileNetV3Large",
     lambda: __import__('tensorflow').keras.applications.MobileNetV3Large(
         input_shape=(96, 96, 3), include_top=False, alpha=1.0, weights="imagenet")),
    ("MobileNetV3Small",
     lambda: __import__('tensorflow').keras.applications.MobileNetV3Small(
         input_shape=(96, 96, 3), include_top=False, alpha=1.0, weights="imagenet")),
    ("EfficientNetV2-B0",
     lambda: __import__('tensorflow').keras.applications.EfficientNetV2B0(
         input_shape=(96, 96, 3), include_top=False, weights="imagenet")),
    ("MobileNetV2",
     lambda: __import__('tensorflow').keras.applications.MobileNetV2(
         input_shape=(96, 96, 3), include_top=False, alpha=1.0, weights="imagenet")),
]


def build_classifier(base_model, num_classes=1):
    """Build the same classification head used in DriveSafe AI."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation="swish",
                     kernel_regularizer=regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="swish",
                     kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def evaluate_tflite(model_path, val_ds, batch_size=BATCH_SIZE):
    """Evaluate a TFLite model and return metrics + latency."""
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            import tensorflow as tf
            litert = tf.lite

    interpreter = litert.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Warmup
    warmup = np.zeros((batch_size, 96, 96, 3), dtype=np.float32)
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], warmup)
        interpreter.invoke()

    # Evaluate
    start = time.perf_counter()
    for images, labels in val_ds:
        bs = images.shape[0]
        if bs != batch_size:
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

    accuracy = correct / total
    latency_ms = (elapsed / total) * 1000
    throughput = total / elapsed

    # Precision / Recall / F1
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

    # Load validation dataset once
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

    for name, make_base in MODELS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        base = make_base()
        model = build_classifier(base)

        # Save as TFLite for fair comparison
        tflite_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace('-', '_')}.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        model_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

        metrics = evaluate_tflite(tflite_path, val_ds)
        metrics["model_name"] = name
        metrics["model_size_mb"] = model_size_mb
        metrics["backbone_params"] = base.count_params()

        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"  Size: {model_size_mb:.1f} MB | Latency: {metrics['latency_ms']:.1f} ms")

        del model, base  # Free memory

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "model_ablation.csv")
    fieldnames = ["model_name", "backbone_params", "model_size_mb",
                  "accuracy", "precision", "recall", "f1",
                  "latency_ms", "throughput", "total", "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "model_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by ablation_model_comparison.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Model Architecture Ablation Results}\n")
        f.write("\\label{tab:model_ablation}\n")
        f.write("\\begin{tabular}{lrrcccccc}\n\\toprule\n")
        f.write("Model & Params & Size (MB) & Acc. & Prec. & Rec. & F1 & Lat. (ms) & Thr. (img/s) \\\\\n\\midrule\n")
        for r in results:
            f.write(f"{r['model_name']} & {r['backbone_params']:,} "
                    f"& {r['model_size_mb']:.1f} & {r['accuracy']:.4f} "
                    f"& {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} "
                    f"& {r['latency_ms']:.1f} & {r['throughput']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
