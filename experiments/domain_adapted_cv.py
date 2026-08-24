"""
Domain-Adapted 5-Fold Cross-Validation
========================================
For each fold:
  1. Load the original pre-trained model
  2. Fine-tune on 4/5 of the akahana dataset (domain adaptation)
  3. Evaluate on the remaining 1/5
  4. Collect metrics

This measures the stability of the domain adaptation process itself.

Usage:
    python experiments/domain_adapted_cv.py

Outputs:
    - results/domain_adapted_cv.csv
    - results/domain_adapted_cv.tex
"""

import csv
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold

# --- Configuration ---
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 42
N_FOLDS = 5
DATASET_PATH = "external_dataset/akahana"
KERAS_MODEL_PATH = "drowsiness.keras"
OUTPUT_DIR = "results"

# Fine-tuning config (same as domain_adaptation.py)
FINETUNE_EPOCHS = 8
FINETUNE_LR = 3e-4
FINETUNE_PATIENCE = 3
AUGMENTATION_STRENGTH = 0.3


def load_dataset_as_arrays():
    """Load all images as numpy arrays for fold splitting."""
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    images_list = []
    labels_list = []

    for cls_idx, cls_name in enumerate(["active", "fatigue"]):
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        files = sorted(os.listdir(cls_dir))
        print(f"  Loading {cls_name}: {len(files)} images")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            img = tf.keras.utils.load_img(fpath, target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img)
            arr = preprocess(arr)
            images_list.append(arr)
            labels_list.append(cls_idx)

    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    print(f"  Total: {len(images)} images, class distribution: {np.bincount(labels)}")
    return images, labels


def create_fold_datasets(train_imgs, val_imgs, train_labels, val_labels):
    """Create tf.data.Dataset from numpy arrays."""
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    train_ds = train_ds.shuffle(len(train_imgs), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_augmented_dataset(train_ds):
    """Add augmentation to training dataset."""
    augmenter = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUGMENTATION_STRENGTH * 0.3),
        layers.RandomTranslation(AUGMENTATION_STRENGTH, AUGMENTATION_STRENGTH),
        layers.RandomZoom(AUGMENTATION_STRENGTH * 0.5),
        layers.RandomBrightness(AUGMENTATION_STRENGTH * 0.5),
        layers.RandomContrast(AUGMENTATION_STRENGTH * 0.5),
    ], name="augmentation")

    return train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)


def fine_tune_for_fold(train_ds, val_ds, fold_idx):
    """Fine-tune the pre-trained model for one fold."""
    # Load fresh copy of pre-trained model
    model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Find and prepare backbone
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 50:
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = False
        unfreeze_from = len(base_model.layers) - 30
        for i, sublayer in enumerate(base_model.layers):
            if i >= unfreeze_from:
                sublayer.trainable = True
                if 'batch_normalization' in sublayer.name.lower():
                    sublayer.trainable = False

    # Always make head trainable
    for layer in model.layers:
        if layer.name in ['dense', 'dense_1', 'dense_2']:
            layer.trainable = True

    # Compile
    loss_fn = keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Fine-tune
    history = model.fit(
        train_ds,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=FINETUNE_PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ],
        verbose=0,  # Silent per fold
    )

    best_acc = max(history.history["val_accuracy"])
    print(f"    Fold {fold_idx+1}: best val_acc = {best_acc*100:.2f}% after {len(history.history['loss'])} epochs")
    return model


def evaluate_model(model, val_ds):
    """Evaluate model and return metrics."""
    all_preds, all_labels = [], []
    start = time.perf_counter()

    for images, labels in val_ds:
        preds = model(images, training=False)
        preds_np = preds.numpy().flatten()
        labels_np = labels.numpy().flatten()
        for i in range(images.shape[0]):
            pred_class = 1 if preds_np[i] > 0.5 else 0
            true_class = int(labels_np[i])
            all_preds.append(pred_class)
            all_labels.append(true_class)

    elapsed = time.perf_counter() - start
    total = len(all_preds)
    latency_ms = (elapsed / total) * 1000

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "latency_ms": latency_ms,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Domain-Adapted 5-Fold Cross-Validation")
    print("=" * 60)

    # Load all data
    print("\nLoading dataset...")
    images, labels = load_dataset_as_arrays()

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} ---")
        print(f"  Train: {len(train_idx)} images, Val: {len(val_idx)} images")

        train_imgs = images[train_idx]
        train_labels = labels[train_idx]
        val_imgs = images[val_idx]
        val_labels = labels[val_idx]

    # Create datasets
    train_ds, val_ds = create_fold_datasets(train_imgs, val_imgs, train_labels, val_labels)

    # Fine-tune and evaluate (skip augmentation for speed)
    model = fine_tune_for_fold(train_ds, val_ds, fold_idx)
    metrics = evaluate_model(model, val_ds)
    metrics["fold"] = fold_idx + 1
    fold_results.append(metrics)

    print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")

    # Free memory
    del model
    tf.keras.backend.clear_session()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std across 5 folds)")
    print("=" * 60)

    metrics_keys = ["accuracy", "precision", "recall", "f1", "specificity"]
    summary = {}
    for key in metrics_keys:
        values = [r[key] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[key] = {"mean": mean_val, "std": std_val}
        if key == "accuracy":
            print(f"  {key:15s}: {mean_val*100:.2f} ± {std_val*100:.2f}%")
        else:
            print(f"  {key:15s}: {mean_val:.4f} ± {std_val:.4f}")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "domain_adapted_cv.csv")
    fieldnames = ["fold", "accuracy", "precision", "recall", "f1", "specificity",
                  "latency_ms", "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in fold_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "domain_adapted_cv.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by domain_adapted_cv.py\n")
        f.write("% 5-fold domain-adapted CV on akahana/Driver-Drowsiness-Dataset\n\n")

        # Per-fold table
        f.write("% Per-fold results\n")
        f.write("\\begin{tabular}{cccccc}\n\\toprule\n")
        f.write("Fold & Acc. (\\%) & Prec. & Rec. & F1 & Spec. \\\\\n\\midrule\n")
        for r in fold_results:
            f.write(f"{r['fold']} & {r['accuracy']*100:.2f} & {r['precision']:.3f} "
                    f"& {r['recall']:.3f} & {r['f1']:.3f} & {r['specificity']:.3f} \\\\\n")
        f.write("\\midrule\n")

        # Mean ± std
        acc_m, acc_s = summary["accuracy"]["mean"], summary["accuracy"]["std"]
        pre_m, pre_s = summary["precision"]["mean"], summary["precision"]["std"]
        rec_m, rec_s = summary["recall"]["mean"], summary["recall"]["std"]
        f1_m, f1_s = summary["f1"]["mean"], summary["f1"]["std"]
        spc_m, spc_s = summary["specificity"]["mean"], summary["specificity"]["std"]

        f.write(f"\\textbf{{Mean $\\pm$ std}} & ")
        f.write(f"\\textbf{{{acc_m*100:.2f} $\\pm$ {acc_s*100:.2f}}} & ")
        f.write(f"\\textbf{{{pre_m:.3f} $\\pm$ {pre_s:.3f}}} & ")
        f.write(f"\\textbf{{{rec_m:.3f} $\\pm$ {rec_s:.3f}}} & ")
        f.write(f"\\textbf{{{f1_m:.3f} $\\pm$ {f1_s:.3f}}} & ")
        f.write(f"\\textbf{{{spc_m:.3f} $\\pm$ {spc_s:.3f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
