"""
Quick eval: Convert drowsiness_robust.keras -> TFLite, evaluate on akahana + combined validation.
If results are poor (<90% on akahana), triggers retraining.
"""
import os, sys, csv, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (96, 96)
BATCH_SIZE = 64
KERAS_PATH = "drowsiness_robust.keras"
TFLITE_PATH = "drowsiness_robust.tflite"
TFLITE_SMALL = "drowsiness_robust_small.tflite"

def eval_tflite(tflite_path, dataset, name=""):
    """Evaluate a TFLite model on a dataset."""
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            from tensorflow.lite.python.lite import Interpreter as litert

    interpreter = litert.Interpreter(model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    all_p, all_l = [], []
    start = time.perf_counter()

    for imgs, labels in dataset:
        bs = imgs.shape[0]
        data = preprocess(imgs).numpy().astype(np.float32)
        labels_np = labels.numpy().flatten()
        try:
            interpreter.resize_tensor_input(inp[0]["index"], [bs, *IMG_SIZE, 3])
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
        except:
            pass
        interpreter.set_tensor(inp[0]["index"], data)
        interpreter.invoke()
        preds = interpreter.get_tensor(out[0]["index"])
        for i in range(bs):
            all_p.append(1 if preds[i][0] > 0.5 else 0)
            all_l.append(int(labels_np[i]))

    elapsed = time.perf_counter() - start
    total = len(all_p)
    tp = sum(1 for p,l in zip(all_p,all_l) if p==1 and l==1)
    fp = sum(1 for p,l in zip(all_p,all_l) if p==1 and l==0)
    fn = sum(1 for p,l in zip(all_p,all_l) if p==0 and l==1)
    tn = sum(1 for p,l in zip(all_p,all_l) if p==0 and l==0)
    acc = (tp+tn)/max(total,1)
    prec = tp/max(tp+fp,1)
    rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8)
    spec = tn/max(tn+fp,1)

    print(f"\n{'='*50}")
    print(f" {name} ({tflite_path})")
    print(f"{'='*50}")
    print(f"  Size:        {os.path.getsize(tflite_path)/(1024*1024):.2f} MB")
    print(f"  Samples:     {total}")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Latency:     {(elapsed/total)*1000:.1f} ms/image")
    print(f"  Confusion:   TP={tp} FP={fp} FN={fn} TN={tn}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "specificity": spec, "latency_ms": (elapsed/total)*1000,
            "size_mb": os.path.getsize(tflite_path)/(1024*1024),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total}


def main():
    os.makedirs("results", exist_ok=True)
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    if not os.path.exists(KERAS_PATH):
        print(f"ERROR: {KERAS_PATH} not found. Run train_combined.py first.")
        sys.exit(1)

    # Load model info
    model = keras.models.load_model(KERAS_PATH, compile=False)
    total_params = sum(tf.keras.backend.count_params(w) for w in model.weights)
    print(f"Robust model: {total_params:,} parameters")

    # === 1. Convert to TFLite (standard dynamic range) ===
    print("\nConverting to TFLite (dynamic range)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_std = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_std)
    print(f"  -> {TFLITE_PATH}: {len(tflite_std)/(1024*1024):.2f} MB")

    # === 2. Convert to smaller TFLite (more aggressive quantization) ===
    print("Converting to TFLite (aggressive quantization)...")
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    # Representative dataset for full integer quantization
    def representative_gen():
        for imgs, _ in tf.keras.utils.image_dataset_from_directory(
            "external_dataset/combined", validation_split=0.1, subset="validation",
            seed=42, image_size=IMG_SIZE, batch_size=1, label_mode="binary",
        ).take(200):
            yield [preprocess(imgs).numpy().astype(np.float32)]
    converter2.representative_dataset = representative_gen
    try:
        tflite_small = converter2.convert()
        with open(TFLITE_SMALL, "wb") as f:
            f.write(tflite_small)
        print(f"  -> {TFLITE_SMALL}: {len(tflite_small)/(1024*1024):.2f} MB")
    except Exception as e:
        print(f"  Aggressive quant failed: {e}")
        TFLITE_SMALL = None

    # === 3. Evaluate on akahana (external holdout) ===
    akahana_ds = tf.keras.utils.image_dataset_from_directory(
        "external_dataset/akahana", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    ).prefetch(tf.data.AUTOTUNE)

    r_std = eval_tflite(TFLITE_PATH, akahana_ds, "Akahana External (standard TFLite)")
    if TFLITE_SMALL and os.path.exists(TFLITE_SMALL):
        r_small = eval_tflite(TFLITE_SMALL, akahana_ds, "Akahana External (small TFLite)")

    # === 4. Evaluate on combined validation (in-distribution) ===
    combined_val = tf.keras.utils.image_dataset_from_directory(
        "external_dataset/combined", validation_split=0.15, subset="validation",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    ).prefetch(tf.data.AUTOTUNE)

    r_combined = eval_tflite(TFLITE_PATH, combined_val, "Combined Validation (in-distribution)")

    # === 5. Compare with baseline ===
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'External Acc':>12} {'In-Dist Acc':>12} {'Size':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    print(f"  {'Baseline (original)':30} {'63.60%':>12} {'~97%':>12} {'4.3 MB':>8}")
    print(f"  {'Domain-adapted (per-ds)':30} {'99.10%':>12} {'N/A':>12} {'4.4 MB':>8}")
    print(f"  {'Multi-dataset (standard)':30} {r_std['accuracy']*100:>11.2f}% {r_combined['accuracy']*100:>11.2f}% {r_std['size_mb']:>7.1f}MB")
    if TFLITE_SMALL and os.path.exists(TFLITE_SMALL):
        print(f"  {'Multi-dataset (small)':30} {r_small['accuracy']*100:>11.2f}% {r_combined['accuracy']*100:>11.2f}% {r_small['size_mb']:>7.1f}MB")

    # === 6. Save results ===
    results = {
        "external_acc": r_std["accuracy"], "external_f1": r_std["f1"],
        "external_recall": r_std["recall"], "external_precision": r_std["precision"],
        "external_latency_ms": r_std["latency_ms"],
        "internal_acc": r_combined["accuracy"], "internal_f1": r_combined["f1"],
        "size_standard_mb": r_std["size_mb"],
        "size_small_mb": r_small["size_mb"] if TFLITE_SMALL and os.path.exists(TFLITE_SMALL) else 0,
    }
    with open("results/multi_dataset_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results.keys())
        w.writeheader()
        w.writerow(results)
    print(f"\nResults saved to results/multi_dataset_results.csv")

    # === 7. Decision: is accuracy good enough? ===
    if r_std["accuracy"] < 0.90:
        print(f"\n*** External accuracy {r_std['accuracy']*100:.1f}% < 90%. RECOMMEND RETRAINING. ***")
        sys.exit(2)  # Signal to retrain
    else:
        print(f"\n*** External accuracy {r_std['accuracy']*100:.1f}% >= 90%. Model is good! ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
