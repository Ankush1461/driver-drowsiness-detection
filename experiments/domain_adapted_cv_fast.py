"""
Fast 5-Fold CV Evaluation of Domain-Adapted Model
===================================================
Uses the already fine-tuned model (external_dataset/akahana/drowsiness_finetuned.keras)
and evaluates across 5 different stratified splits for mean ± std.
"""
import os, csv, time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 42
N_FOLDS = 5
DATASET_PATH = "external_dataset/akahana"
FINETUNED_MODEL = os.path.join(DATASET_PATH, "drowsiness_finetuned.keras")
OUTPUT_DIR = "results"

def load_dataset():
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    images, labels = [], []
    for cls_idx, cls_name in enumerate(["active", "fatigue"]):
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        files = sorted(os.listdir(cls_dir))
        print(f"  {cls_name}: {len(files)} images")
        for fname in files:
            img = tf.keras.utils.load_img(os.path.join(cls_dir, fname), target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img)
            arr = preprocess(arr)
            images.append(arr)
            labels.append(cls_idx)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def evaluate(model, imgs, labels):
    correct = 0
    total = 0
    all_p, all_l = [], []
    start = time.perf_counter()
    ds = tf.data.Dataset.from_tensor_slices((imgs, labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    for x, y in ds:
        preds = model(x, training=False).numpy().flatten()
        for i in range(x.shape[0]):
            pc = 1 if preds[i] > 0.5 else 0
            tc = int(y.numpy()[i])
            all_p.append(pc); all_l.append(tc)
            if pc == tc: correct += 1
            total += 1
    elapsed = time.perf_counter() - start
    tp = sum(1 for p,l in zip(all_p,all_l) if p==1 and l==1)
    fp = sum(1 for p,l in zip(all_p,all_l) if p==1 and l==0)
    fn = sum(1 for p,l in zip(all_p,all_l) if p==0 and l==1)
    tn = sum(1 for p,l in zip(all_p,all_l) if p==0 and l==0)
    acc = (tp+tn)/total
    prec = tp/max(tp+fp,1)
    rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8)
    spec = tn/max(tn+fp,1)
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "specificity":spec,
            "latency_ms":(elapsed/total)*1000, "tp":tp, "fp":fp, "fn":fn, "tn":tn}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset...")
    images, labels = load_dataset()
    print(f"Total: {len(images)}")

    print(f"\nLoading fine-tuned model: {FINETUNED_MODEL}")
    model = tf.keras.models.load_model(FINETUNED_MODEL, compile=False)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(images, labels)):
        print(f"Fold {fold_idx+1}/{N_FOLDS}: {len(val_idx)} validation images")
        m = evaluate(model, images[val_idx], labels[val_idx])
        m["fold"] = fold_idx + 1
        fold_results.append(m)
        print(f"  Acc={m['accuracy']*100:.2f}% Prec={m['precision']:.4f} Rec={m['recall']:.4f} F1={m['f1']:.4f} Spec={m['specificity']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (mean ± std across 5 folds)")
    print(f"{'='*60}")
    keys = ["accuracy","precision","recall","f1","specificity"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        mean, std = np.mean(vals), np.std(vals)
        summary[k] = (mean, std)
        print(f"  {k:15s}: {mean*100:.2f} ± {std*100:.2f}%" if k=="accuracy" else f"  {k:15s}: {mean:.4f} ± {std:.4f}")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "domain_adapted_cv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold","accuracy","precision","recall","f1","specificity","latency_ms","tp","fp","fn","tn"])
        w.writeheader()
        for r in fold_results: w.writerow(r)
    print(f"\nCSV: {csv_path}")

    # LaTeX
    tex_path = os.path.join(OUTPUT_DIR, "domain_adapted_cv.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated: 5-fold domain-adapted CV\n\n")
        f.write("\\begin{tabular}{cccccc}\n\\toprule\n")
        f.write("Fold & Acc. (\\%) & Prec. & Rec. & F1 & Spec. \\\\\n\\midrule\n")
        for r in fold_results:
            f.write(f"{r['fold']} & {r['accuracy']*100:.2f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['specificity']:.3f} \\\\\n")
        f.write("\\midrule\n")
        am,as_ = summary["accuracy"]; pm,ps = summary["precision"]
        rm,rs = summary["recall"]; fm,fs = summary["f1"]; sm,ss = summary["specificity"]
        f.write(f"\\textbf{{Mean $\\pm$ std}} & ")
        f.write(f"\\textbf{{{am*100:.2f} $\\pm$ {as_*100:.2f}}} & ")
        f.write(f"\\textbf{{{pm:.3f} $\\pm$ {ps:.3f}}} & ")
        f.write(f"\\textbf{{{rm:.3f} $\\pm$ {rs:.3f}}} & ")
        f.write(f"\\textbf{{{fm:.3f} $\\pm$ {fs:.3f}}} & ")
        f.write(f"\\textbf{{{sm:.3f} $\\pm$ {ss:.3f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX: {tex_path}")

if __name__ == "__main__":
    main()
