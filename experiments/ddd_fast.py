import numpy as np, os, glob, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

DDD = "external_dataset/Driver Drowsiness Dataset (DDD)"
IMG_SIZE = 96

# Load file paths and labels (fast)
paths, labels = [], []
for lbl, folder in [(0, "Non Drowsy"), (1, "Drowsy")]:
    fp = os.path.join(DDD, folder)
    for f in sorted(glob.glob(os.path.join(fp, "*.png"))):
        paths.append(f)
        labels.append(lbl)
paths = np.array(paths)
labels = np.array(labels)
print(f"Total: {len(paths)} images ({np.sum(labels==1)} drowsy, {np.sum(labels==0)} non-drowsy)")

def eval_model(model_path, test_paths, test_labels, name):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_input_details()
    out_det = interp.get_output_details()
    batched = inp[0]["shape"][0] == -1 or inp[0]["shape"][0] > 1
    bs = 128 if batched else 1
    
    # Warmup
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for _ in range(3):
        interp.set_tensor(inp[0]["index"], dummy)
        interp.invoke()
    
    start = time.time()
    preds = []
    for i in range(0, len(test_paths), bs):
        batch_files = test_paths[i:i+bs]
        batch = []
        for f in batch_files:
            raw = tf.io.read_file(f)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            batch.append(img.numpy().astype(np.float32) / 127.5 - 1.0)
        batch = np.array(batch)
        interp.set_tensor(inp[0]["index"], batch)
        interp.invoke()
        ov = interp.get_tensor(out_det[0]["index"])
        p = ov[:, 0] if ov.ndim > 1 else ov
        preds.extend((p > 0.5).astype(int))
    elapsed = time.time() - start
    
    preds = np.array(preds)
    acc = accuracy_score(test_labels, preds) * 100
    prec = precision_score(test_labels, preds, zero_division=0)
    rec = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
    spec = tn / (tn + fp)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "spec": spec, "time": elapsed}

# Evaluate Robust model (most important)
models = {
    "Robust": "drowsiness_robust.tflite",
    "Pipeline v2": "drowsiness_pipeline_v2.tflite",
    "Baseline": "drowsiness.tflite",
}

all_results = {}
for mname, mpath in models.items():
    if not os.path.exists(mpath):
        print(f"\nSkipping {mpath} (not found)")
        continue
    print(f"\n{'='*60}\n{mname}\n{'='*60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold, (tr, te) in enumerate(skf.split(paths, labels), 1):
        t0 = time.time()
        r = eval_model(mpath, paths[te], labels[te], f"Fold {fold}")
        print(f"  Fold {fold}: Acc={r['acc']:.2f}% Prec={r['prec']:.3f} Rec={r['rec']:.3f} F1={r['f1']:.3f} Spec={r['spec']:.3f} ({r['time']:.1f}s)")
        folds.append(r)
    accs = [f["acc"] for f in folds]
    precs = [f["prec"] for f in folds]
    recs = [f["rec"] for f in folds]
    f1s = [f["f1"] for f in folds]
    specs = [f["spec"] for f in folds]
    print(f"  MEAN: Acc={np.mean(accs):.2f}+/-{np.std(accs):.2f} Prec={np.mean(precs):.3f}+/-{np.std(precs):.3f} Rec={np.mean(recs):.3f}+/-{np.std(recs):.3f} F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f} Spec={np.mean(specs):.3f}+/-{np.std(specs):.3f}")
    all_results[mname] = {"accs": accs, "precs": precs, "recs": recs, "f1s": f1s, "specs": specs}

# Statistical tests
if len(all_results) >= 2:
    from scipy import stats
    print(f"\n{'='*60}\nSTATISTICAL TESTS\n{'='*60}")
    names = list(all_results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            t, p = stats.ttest_rel(all_results[a]["accs"], all_results[b]["accs"])
            diff = np.mean(all_results[b]["accs"]) - np.mean(all_results[a]["accs"])
            print(f"  {a} vs {b}: t={t:.2f} p={p:.6f} (diff={diff:+.2f}pp)")

os.makedirs("results", exist_ok=True)
with open("results/ddd_results.txt", "w") as f:
    for mn, mr in all_results.items():
        f.write(f"{mn}: Acc={np.mean(mr['accs']):.2f}+/-{np.std(mr['accs']):.2f} Prec={np.mean(mr['precs']):.3f}+/-{np.std(mr['precs']):.3f} Rec={np.mean(mr['recs']):.3f}+/-{np.std(mr['recs']):.3f} F1={np.mean(mr['f1s']):.3f}+/-{np.std(mr['f1s']):.3f} Spec={np.mean(mr['specs']):.3f}+/-{np.std(mr['specs']):.3f}\n")
        f.write(f"  Fold accs: {[f'{a:.2f}' for a in mr['accs']]}\n")
print("\nSaved to results/ddd_results.txt")
