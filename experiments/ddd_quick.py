import numpy as np, os, time, glob
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

DDD = "external_dataset/Driver Drowsiness Dataset (DDD)"
IMG_SIZE = 96
WARMUP = 3

def load_all():
    imgs, labels = [], []
    for lbl, folder in [(0, "Non Drowsy"), (1, "Drowsy")]:
        fp = os.path.join(DDD, folder)
        files = sorted(glob.glob(os.path.join(fp, "*.png")))
        print(f"  Loading {folder}: {len(files)} images")
        for f in files:
            im = Image.open(f).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
            imgs.append(arr)
            labels.append(lbl)
    return np.stack(imgs), np.array(labels)

def eval_one(model_path, X, y, name):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    batched = inp[0]["shape"][0] == -1 or inp[0]["shape"][0] > 1
    bs = 256 if batched else 1
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for _ in range(WARMUP):
        interp.set_tensor(inp[0]["index"], dummy)
        interp.invoke()
    start = time.time()
    preds = []
    for i in range(0, len(X), bs):
        batch = X[i:i+bs]
        interp.set_tensor(inp[0]["index"], batch)
        interp.invoke()
        out_val = interp.get_tensor(out[0]["index"])
        p = out_val[:, 0] if out_val.ndim > 1 else out_val
        preds.extend((p > 0.5).astype(int))
    elapsed = time.time() - start
    preds = np.array(preds)
    acc = accuracy_score(y, preds) * 100
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    spec = tn / (tn + fp)
    fps = len(X) / elapsed
    print(f"  {name}: Acc={acc:.2f}% Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} Spec={spec:.3f} ({fps:.1f} img/s)")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "spec": spec, "fps": fps, "time": elapsed}

print("Loading DDD dataset...")
X, y = load_all()
print(f"Total: {len(X)} images ({np.sum(y==1)} drowsy, {np.sum(y==0)} non-drowsy)")

models = {
    "Baseline": "drowsiness.tflite",
    "Pipeline v2": "drowsiness_pipeline_v2.tflite",
    "Robust": "drowsiness_robust.tflite",
}

all_results = {}
for mname, mpath in models.items():
    if not os.path.exists(mpath):
        print(f"\nSkipping {mpath} (not found)")
        continue
    print(f"\n{'='*60}\n{mname}\n{'='*60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        r = eval_one(mpath, X[te], y[te], f"Fold {fold}")
        folds.append(r)
    accs = [f["acc"] for f in folds]
    precs = [f["prec"] for f in folds]
    recs = [f["rec"] for f in folds]
    f1s = [f["f1"] for f in folds]
    specs = [f["spec"] for f in folds]
    print(f"\n  MEAN: Acc={np.mean(accs):.2f}+/-{np.std(accs):.2f} Prec={np.mean(precs):.3f}+/-{np.std(precs):.3f} Rec={np.mean(recs):.3f}+/-{np.std(recs):.3f} F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f} Spec={np.mean(specs):.3f}+/-{np.std(specs):.3f}")
    all_results[mname] = {"accs": accs, "precs": precs, "recs": recs, "f1s": f1s, "specs": specs}

if len(all_results) >= 2:
    from scipy import stats
    print(f"\n{'='*60}\nSTATISTICAL TESTS\n{'='*60}")
    names = list(all_results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            t, p = stats.ttest_rel(all_results[a]["accs"], all_results[b]["accs"])
            diff = np.mean(all_results[b]["accs"]) - np.mean(all_results[a]["accs"])
            pooled = np.std([np.mean(all_results[a]["accs"]), np.mean(all_results[b]["accs"])])
            d = diff / pooled if pooled > 0 else 0
            print(f"  {a} vs {b}: t={t:.2f} p={p:.6f} d={d:.2f} (diff={diff:+.2f}pp)")

os.makedirs("results", exist_ok=True)
with open("results/ddd_results.txt", "w") as f:
    for mn, mr in all_results.items():
        f.write(f"{mn}: Acc={np.mean(mr['accs']):.2f}+/-{np.std(mr['accs']):.2f} Prec={np.mean(mr['precs']):.3f}+/-{np.std(mr['precs']):.3f} Rec={np.mean(mr['recs']):.3f}+/-{np.std(mr['recs']):.3f} F1={np.mean(mr['f1s']):.3f}+/-{np.std(mr['f1s']):.3f} Spec={np.mean(mr['specs']):.3f}+/-{np.std(mr['specs']):.3f}\n")
        f.write(f"  Fold accs: {[f'{a:.2f}' for a in mr['accs']]}\n")
print("\nDone! Results saved to results/ddd_results.txt")
