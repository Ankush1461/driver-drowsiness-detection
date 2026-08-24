"""
Subject-Independent Cross-Validation Ablation
===============================================
5-fold stratified CV on the external akahana/Driver-Drowsiness-Dataset.
Reports mean ± std for accuracy, precision, recall, F1, and specificity.

Usage:
    python experiments/subject_independent_cv.py

Outputs:
    - results/cross_validation.csv
    - results/cross_validation.tex
"""

import csv
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
DATASET_PATH = "external_dataset/akahana"
NUM_THREADS = 4
NUM_WARMUP = 10
N_FOLDS = 5
OUTPUT_DIR = "results"

KERAS_MODEL_PATH = "drowsiness.keras"
TFLITE_MODEL_PATH = "drowsiness.tflite"


def load_dataset(path, img_size):
    """Load all images and labels from directory into numpy arrays."""
    images, labels = [], []
    class_names = sorted(os.listdir(path))
    for label_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(path, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        print(f"  {cls}: {len(files)} images")
        for f in files:
            try:
                img = Image.open(os.path.join(cls_dir, f)).convert("RGB").resize(img_size, Image.LANCZOS)
                images.append(np.array(img, dtype=np.float32))
                labels.append(label_idx)
            except Exception:
                pass
    return np.array(images), np.array(labels), class_names


def metrics(preds, labels):
    """Compute classification metrics."""
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    n = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def eval_keras(model, imgs, labels):
    """Batch-evaluate Keras model."""
    preds = model(imgs, training=False).numpy().flatten()
    return metrics((preds > 0.5).astype(int), labels)


def eval_tflite(tflite_path, imgs, labels):
    """Evaluate TFLite model in batches."""
    try:
        import ai_edge_litert.interpreter as interp_mod
    except ImportError:
        import tflite_runtime.interpreter as interp_mod

    interpreter = interp_mod.Interpreter(model_path=tflite_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    # Warmup
    for _ in range(NUM_WARMUP):
        interpreter.set_tensor(inp_idx, np.zeros((1, 96, 96, 3), dtype=np.float32))
        interpreter.invoke()

    all_preds = []
    n = len(imgs)

    # Process in BATCH_SIZE chunks
    for start in range(0, n, BATCH_SIZE):
        batch = imgs[start:start + BATCH_SIZE]
        bs = len(batch)

        # For TFLite, we must resize to exact batch size
        if bs != interpreter.get_input_details()[0]['shape'][0]:
            try:
                interpreter.resize_tensor_input(inp_idx, [bs, 96, 96, 3])
                interpreter.allocate_tensors()
                inp_idx = interpreter.get_input_details()[0]['index']
                out_idx = interpreter.get_output_details()[0]['index']
            except RuntimeError:
                # XNNPack fallback: process one by one
                for i in range(bs):
                    single = batch[i:i+1]
                    interpreter.set_tensor(inp_idx, single)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(out_idx)
                    all_preds.append(1 if pred[0][0] > 0.5 else 0)
                continue

        interpreter.set_tensor(inp_idx, batch.astype(np.float32))
        interpreter.invoke()
        preds = interpreter.get_tensor(out_idx).flatten()
        all_preds.extend((preds > 0.5).astype(int).tolist())

    return metrics(np.array(all_preds), labels)


def main():
    import tensorflow as tf
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    images, labels, class_names = load_dataset(DATASET_PATH, IMG_SIZE)
    print(f"Total: {len(images)} | active={np.sum(labels==0)} fatigue={np.sum(labels==1)}\n")

    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1"]

    results = {"keras": [], "tflite": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"{'='*50} FOLD {fold+1}/{N_FOLDS} {'='*50}")
        val_imgs = preprocess(images[val_idx].copy()).astype(np.float32)
        val_lbls = labels[val_idx]
        print(f"  Val: {len(val_idx)} images (active={np.sum(val_lbls==0)} fatigue={np.sum(val_lbls==1)})")

        # Keras
        if os.path.exists(KERAS_MODEL_PATH):
            t0 = time.perf_counter()
            model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            m = eval_keras(model, val_imgs, val_lbls)
            dt = time.perf_counter() - t0
            m["fold"] = fold + 1
            m["time_s"] = dt
            results["keras"].append(m)
            print(f"  Keras:  Acc={m['accuracy']*100:.2f}% F1={m['f1']:.4f} ({dt:.1f}s)")
            del model

        # TFLite
        if os.path.exists(TFLITE_MODEL_PATH):
            t0 = time.perf_counter()
            m = eval_tflite(TFLITE_MODEL_PATH, val_imgs, val_lbls)
            dt = time.perf_counter() - t0
            m["fold"] = fold + 1
            m["time_s"] = dt
            results["tflite"].append(m)
            print(f"  TFLite: Acc={m['accuracy']*100:.2f}% F1={m['f1']:.4f} ({dt:.1f}s)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS (mean ± std)")
    print(f"{'='*60}")

    for mtype in ["keras", "tflite"]:
        if not results[mtype]:
            continue
        print(f"\n  {mtype.upper()}:")
        for metric in metric_names:
            vals = [r[metric] for r in results[mtype]]
            print(f"    {metric:12s}: {np.mean(vals)*100:.2f}% ± {np.std(vals, ddof=1)*100:.2f}%")
        fold_accs = ' '.join(f"{r['accuracy']*100:.2f}%" for r in results[mtype])
        print(f"    Per-fold acc: {fold_accs}")

    # --- CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "cross_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "accuracy", "precision", "recall", "specificity", "f1",
                      "tp", "fp", "fn", "tn"])
        for mtype in ["keras", "tflite"]:
            for r in results[mtype]:
                w.writerow([mtype, r["fold"]] + [f"{r[m]:.6f}" for m in metric_names] +
                           [r["tp"], r["fp"], r["fn"], r["tn"]])
    print(f"\nCSV: {csv_path}")

    # --- LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "cross_validation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by subject_independent_cv.py\n")
        f.write("% 5-fold stratified CV on akahana/Driver-Drowsiness-Dataset\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{5-Fold Stratified Cross-Validation on External Benchmark\n")
        f.write("(\\texttt{akahana/Driver-Drowsiness-Dataset}).\n")
        f.write("Models were \\emph{not} trained on this dataset.}\n")
        f.write("\\label{tab:cross_validation}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Model & Accuracy & Precision & Recall & Specificity & F1 \\\\\n")
        f.write(" & (\\% $\\pm$ std) & (\\% $\\pm$ std) & (\\% $\\pm$ std) & (\\% $\\pm$ std) & (\\% $\\pm$ std) \\\\\n\\midrule\n")
        for mtype, label in [("keras", "Keras float32"), ("tflite", "TFLite dynamic-range")]:
            if not results[mtype]:
                continue
            cells = []
            for metric in metric_names:
                vals = [r[metric] for r in results[mtype]]
                cells.append(f"{np.mean(vals)*100:.2f} $\\pm$ {np.std(vals, ddof=1)*100:.2f}")
            f.write(f"{label} & {' & '.join(cells)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
