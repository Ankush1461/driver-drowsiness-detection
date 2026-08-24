"""Full benchmark CV with statistical tests on akahana."""
import os, sys, csv, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from PIL import Image

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
N_FOLDS = 5
DATASET_PATH = "external_dataset/akahana"

def load_dataset(path, img_size):
    images, labels = [], []
    class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    for li, cls in enumerate(class_names):
        cls_dir = os.path.join(path, cls)
        files = sorted(os.listdir(cls_dir))
        print(f"  {cls}: {len(files)} images")
        for f in files:
            try:
                img = Image.open(os.path.join(cls_dir, f)).convert("RGB").resize(img_size, Image.LANCZOS)
                images.append(np.array(img, dtype=np.float32))
                labels.append(li)
            except:
                pass
    return np.array(images), np.array(labels), class_names

def compute_metrics(preds, labels):
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
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def eval_tflite(model_path, imgs, labels):
    try:
        import ai_edge_litert.interpreter as interp_mod
    except:
        import tflite_runtime.interpreter as interp_mod
    
    interp = interp_mod.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    
    for _ in range(5):
        interp.set_tensor(inp_idx, np.zeros((1, 96, 96, 3), dtype=np.float32))
        interp.invoke()
    
    all_preds = []
    for start in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[start:start+BATCH_SIZE].astype(np.float32)
        bs = len(batch)
        try:
            interp.resize_tensor_input(inp_idx, [bs, 96, 96, 3])
            interp.allocate_tensors()
            inp_idx = interp.get_input_details()[0]['index']
            out_idx = interp.get_output_details()[0]['index']
        except:
            pass
        interp.set_tensor(inp_idx, batch)
        interp.invoke()
        preds = interp.get_tensor(out_idx).flatten()
        all_preds.extend((preds > 0.5).astype(int).tolist())
    
    return compute_metrics(np.array(all_preds), labels)

def main():
    import tensorflow as tf
    
    print("Loading akahana dataset...")
    images, labels, class_names = load_dataset(DATASET_PATH, IMG_SIZE)
    print(f"Total: {len(images)} | active={np.sum(labels==0)} fatigue={np.sum(labels==1)}\n")
    
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    models = {
        "baseline": "drowsiness.tflite",
        "robust": "drowsiness_robust.tflite",
        "pipeline_v2": "drowsiness_pipeline_v2.tflite",
    }
    
    all_results = {name: [] for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"=== FOLD {fold+1}/{N_FOLDS} ===")
        val_imgs = preprocess(images[val_idx].copy()).astype(np.float32)
        val_lbls = labels[val_idx]
        print(f"  Val: {len(val_idx)} images")
        
        for name, model_path in models.items():
            if not os.path.exists(model_path):
                print(f"  {name}: MODEL NOT FOUND")
                continue
            t0 = time.perf_counter()
            m = eval_tflite(model_path, val_imgs, val_lbls)
            dt = time.perf_counter() - t0
            m["fold"] = fold + 1
            m["latency_ms"] = dt * 1000 / len(val_idx)
            all_results[name].append(m)
            print(f"  {name:15s}: Acc={m['accuracy']*100:.2f}% F1={m['f1']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} ({dt:.1f}s)")
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1"]
    
    for name in models:
        if not all_results[name]:
            continue
        print(f"\n{name}:")
        for m in metric_names:
            vals = [r[m] for r in all_results[name]]
            print(f"  {m:12s}: {np.mean(vals)*100:.2f}% +/- {np.std(vals, ddof=1)*100:.2f}%")
    
    # Statistical tests: baseline vs robust
    if all_results["baseline"] and all_results["robust"]:
        print(f"\n{'='*70}")
        print("STATISTICAL SIGNIFICANCE TESTS (baseline vs robust)")
        print(f"{'='*70}")
        
        for m in ["accuracy", "f1", "recall"]:
            base_vals = [r[m]*100 for r in all_results["baseline"]]
            rob_vals = [r[m]*100 for r in all_results["robust"]]
            
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(rob_vals, base_vals)
            # Wilcoxon signed-rank
            try:
                w_stat, w_pval = stats.wilcoxon(rob_vals, base_vals)
            except:
                w_stat, w_pval = float('nan'), float('nan')
            # Cohen's d
            diff = np.array(rob_vals) - np.array(base_vals)
            cohens_d = np.mean(diff) / max(np.std(diff, ddof=1), 1e-8)
            
            sig_t = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "ns"
            sig_w = "***" if w_pval < 0.001 else "**" if w_pval < 0.01 else "*" if w_pval < 0.05 else "ns"
            
            print(f"\n  {m}:")
            print(f"    baseline: {np.mean(base_vals):.2f}% +/- {np.std(base_vals, ddof=1):.2f}%")
            print(f"    robust:   {np.mean(rob_vals):.2f}% +/- {np.std(rob_vals, ddof=1):.2f}%")
            print(f"    Paired t-test: t={t_stat:.3f}, p={t_pval:.4f} {sig_t}")
            print(f"    Wilcoxon:      W={w_stat:.1f}, p={w_pval:.4f} {sig_w}")
            print(f"    Cohen's d:     {cohens_d:.3f}")
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/full_benchmark_cv.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "accuracy", "precision", "recall", "specificity", "f1", "tp", "fp", "fn", "tn", "latency_ms"])
        for name in models:
            for r in all_results[name]:
                w.writerow([name, r["fold"]] + [f"{r[m]:.6f}" for m in metric_names] + [r["tp"], r["fp"], r["fn"], r["tn"], f"{r['latency_ms']:.2f}"])
    print(f"\nCSV saved: {csv_path}")

if __name__ == "__main__":
    main()
