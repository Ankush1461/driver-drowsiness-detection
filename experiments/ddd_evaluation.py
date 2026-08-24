import numpy as np
import os, sys, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Config
DDD_PATH = "external_dataset/Driver Drowsiness Dataset (DDD)"
MODELS = {
    "Baseline": "drowsiness.tflite",
    "Pipeline v2": "drowsiness_pipeline_v2.tflite",
    "Robust": "drowsiness_robust.tflite",
}
IMG_SIZE = 96
BATCH_SIZE = 32
N_FOLDS = 5
WARMUP = 5

def load_dataset():
    images, labels = [], []
    for label, folder in [(0, "Non Drowsy"), (1, "Drowsy")]:
        folder_path = os.path.join(DDD_PATH, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(folder_path, fname))
                labels.append(label)
    return np.array(images), np.array(labels)

def load_and_preprocess(paths):
    imgs = []
    for p in paths:
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img.numpy().astype(np.float32) / 127.5 - 1.0  # MobileNet preprocessing
        imgs.append(img)
    return np.array(imgs)

def evaluate_model(model_path, paths, labels, model_name):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warmup
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for _ in range(WARMUP):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Check batch support
    batch_supported = input_details[0]["shape"][0] > 1 or input_details[0]["shape"][0] == -1
    actual_batch = BATCH_SIZE if batch_supported else 1

    # Batched evaluation
    all_preds, all_probs = [], []
    start = time.time()
    for i in range(0, len(paths), actual_batch):
        batch_paths = paths[i:i+actual_batch]
        batch_imgs = load_and_preprocess(batch_paths)
        interpreter.set_tensor(input_details[0]["index"], batch_imgs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        probs = output[:, 0] if output.ndim > 1 else output
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs if probs.ndim > 0 else [probs])
        all_preds.extend(preds if preds.ndim > 0 else [preds])
    elapsed = time.time() - start

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    acc = accuracy_score(labels, all_preds) * 100
    prec = precision_score(labels, all_preds, zero_division=0)
    rec = recall_score(labels, all_preds, zero_division=0)
    f1 = f1_score(labels, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, all_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    fps = len(paths) / elapsed if elapsed > 0 else 0

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "spec": spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fps": fps, "latency_ms": 1000/fps if fps > 0 else 0
    }

def main():
    print("Loading DDD dataset...")
    paths, labels = load_dataset()
    print(f"  Total: {len(paths)} images ({np.sum(labels==1)} drowsy, {np.sum(labels==0)} non-drowsy)")

    results = {}
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name}: {model_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name} ({model_path})")
        print(f"{'='*70}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels), 1):
            test_paths = paths[test_idx]
            test_labels = labels[test_idx]
            r = evaluate_model(model_path, test_paths, test_labels, model_name)
            fold_results.append(r)
            print(f"  Fold {fold}: Acc={r['acc']:.2f}% Prec={r['prec']:.3f} Rec={r['rec']:.3f} F1={r['f1']:.3f} Spec={r['spec']:.3f}")

        # Aggregate
        accs = [r["acc"] for r in fold_results]
        precs = [r["prec"] for r in fold_results]
        recs = [r["rec"] for r in fold_results]
        f1s = [r["f1"] for r in fold_results]
        specs = [r["spec"] for r in fold_results]

        print(f"\n  --- {model_name} Summary ---")
        print(f"  Accuracy:  {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")
        print(f"  Precision: {np.mean(precs):.3f} +/- {np.std(precs):.3f}")
        print(f"  Recall:    {np.mean(recs):.3f} +/- {np.std(recs):.3f}")
        print(f"  F1:        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        print(f"  Specificity: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")

        results[model_name] = {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "prec_mean": np.mean(precs), "prec_std": np.std(precs),
            "rec_mean": np.mean(recs), "rec_std": np.std(recs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "spec_mean": np.mean(specs), "spec_std": np.std(specs),
            "fold_accs": accs, "fold_precs": precs, "fold_recs": recs,
            "fold_f1s": f1s, "fold_specs": specs,
        }

    # Statistical tests
    if "Baseline" in results and "Robust" in results:
        print(f"\n{'='*70}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print(f"{'='*70}")

        from scipy import stats

        bl = results["Baseline"]
        rb = results["Robust"]
        pv = results.get("Pipeline v2", None)

        # Paired t-test: Baseline vs Robust
        t_stat, p_val = stats.ttest_rel(bl["fold_accs"], rb["fold_accs"])
        d = (np.mean(rb["fold_accs"]) - np.mean(bl["fold_accs"])) / np.std([np.mean(bl["fold_accs"]), np.mean(rb["fold_accs"])])
        print(f"\n  Baseline vs Robust (Accuracy):")
        print(f"    t = {t_stat:.2f}, p = {p_val:.6f}, Cohen's d = {d:.2f}")

        # Paired t-test: Pipeline v2 vs Robust
        if pv:
            t_stat2, p_val2 = stats.ttest_rel(pv["fold_accs"], rb["fold_accs"])
            d2 = (np.mean(rb["fold_accs"]) - np.mean(pv["fold_accs"])) / np.std([np.mean(pv["fold_accs"]), np.mean(rb["fold_accs"])])
            print(f"\n  Pipeline v2 vs Robust (Accuracy):")
            print(f"    t = {t_stat2:.2f}, p = {p_val2:.6f}, Cohen's d = {d2:.2f}")

        # Wilcoxon signed-rank
        try:
            w_stat, w_p = stats.wilcoxon(bl["fold_accs"], rb["fold_accs"])
            print(f"\n  Wilcoxon (Baseline vs Robust): W = {w_stat:.2f}, p = {w_p:.6f}")
        except ValueError:
            print(f"\n  Wilcoxon: insufficient variation for test")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ddd_5fold_results.txt", "w") as f:
        for model_name, r in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Accuracy:  {r['acc_mean']:.2f} +/- {r['acc_std']:.2f}%\n")
            f.write(f"  Precision: {r['prec_mean']:.3f} +/- {r['prec_std']:.3f}\n")
            f.write(f"  Recall:    {r['rec_mean']:.3f} +/- {r['rec_std']:.3f}\n")
            f.write(f"  F1:        {r['f1_mean']:.3f} +/- {r['f1_std']:.3f}\n")
            f.write(f"  Specificity: {r['spec_mean']:.3f} +/- {r['spec_std']:.3f}\n")
            f.write(f"  Fold accs: {[f'{a:.2f}' for a in r['fold_accs']]}\n")
    print("\nResults saved to results/ddd_5fold_results.txt")

if __name__ == "__main__":
    main()
