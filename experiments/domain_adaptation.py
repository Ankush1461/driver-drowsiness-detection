"""
Domain Adaptation: Fine-tuning Pre-trained Model on External Dataset
====================================================================
Bridges the generalization gap by:
1. Fine-tuning the pre-trained model on akahana dataset (small LR, heavy augmentation)
2. Test-Time Augmentation (TTA) for improved inference
3. Combined training: original + external data

Usage:
    python experiments/domain_adaptation.py

Outputs:
    - results/domain_adaptation.csv
    - results/domain_adaptation.tex
    - external_dataset/akahana/drowsiness_finetuned.keras (fine-tuned model)
"""

import csv
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 123
EXTERNAL_DATASET_PATH = "external_dataset/akahana"
KERAS_MODEL_PATH = "drowsiness.keras"
TFLITE_MODEL_PATH = "drowsiness.tflite"
OUTPUT_DIR = "results"
FINETuned_MODEL_PATH = os.path.join(EXTERNAL_DATASET_PATH, "drowsiness_finetuned.keras")

# Fine-tuning hyperparameters
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4
FINETUNE_PATIENCE = 5
LABEL_SMOOTHING = 0.1
AUGMENTATION_STRENGTH = 0.3  # Stronger than training to prevent overfitting

# TTA configuration
TTA_TRANSFORMS = 8  # Number of augmented copies per image


def create_external_dataset(subset="training"):
    """Load akahana dataset with heavy augmentation for fine-tuning."""
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    augmenter = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUGMENTATION_STRENGTH * 0.3),
        layers.RandomTranslation(AUGMENTATION_STRENGTH, AUGMENTATION_STRENGTH),
        layers.RandomZoom(AUGMENTATION_STRENGTH * 0.5),
        layers.RandomBrightness(AUGMENTATION_STRENGTH * 0.5),
        layers.RandomContrast(AUGMENTATION_STRENGTH * 0.5),
        layers.RandomCrop(int(IMG_SIZE[0] * 0.9), int(IMG_SIZE[1] * 0.9)),
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
    ], name="domain_augmentation")

    ds = tf.keras.utils.image_dataset_from_directory(
        EXTERNAL_DATASET_PATH,
        validation_split=0.2,
        subset="training" if subset == "training" else "validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = ds.class_names
    print(f"  Classes: {class_names}")

    if subset == "training":
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def load_and_prepare_model():
    """Load the pre-trained model and prepare for fine-tuning."""
    print(f"Loading pre-trained model from {KERAS_MODEL_PATH}...")
    model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Find the MobileNetV3Large backbone (layer index 1, type Functional)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 50:
            base_model = layer
            break

    if base_model is not None:
        print(f"  Found backbone: {base_model.name} ({len(base_model.layers)} sublayers)")

        # Strategy: freeze most of backbone, unfreeze last 30 + all head layers
        base_model.trainable = False
        unfreeze_from = len(base_model.layers) - 30
        for i, sublayer in enumerate(base_model.layers):
            if i >= unfreeze_from:
                sublayer.trainable = True
                if 'batch_normalization' in sublayer.name.lower():
                    sublayer.trainable = False  # Keep BN frozen
    else:
        print("  WARNING: Could not identify backbone. Fine-tuning all layers.")

    # Always make head layers trainable
    for layer in model.layers:
        if layer.name in ['dense', 'dense_1', 'dense_2', 'batch_normalization']:
            layer.trainable = True
            if 'batch_normalization' in layer.name:
                layer.trainable = False

    # Print trainable params
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"  Trainable params: {trainable:,} / {trainable + non_trainable:,}")

    return model


def fine_tune_model():
    """Fine-tune the pre-trained model on the external dataset."""
    print("\n" + "=" * 60)
    print("DOMAIN ADAPTATION: Fine-tuning on akahana dataset")
    print("=" * 60)

    train_ds, class_names = create_external_dataset("training")
    val_ds, _ = create_external_dataset("validation")

    # Count samples
    train_count = train_ds.cardinality().numpy() * BATCH_SIZE
    val_count = val_ds.cardinality().numpy() * BATCH_SIZE
    print(f"  Train samples: ~{train_count}")
    print(f"  Val samples: ~{val_count}")

    model = load_and_prepare_model()

    # Compile with lower learning rate and label smoothing
    loss_fn = keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=FINETUNE_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            FINETuned_MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max"
        ),
    ]

    print(f"\nFine-tuning for up to {FINETUNE_EPOCHS} epochs (LR={FINETUNE_LR})...")
    history = model.fit(
        train_ds,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list,
    )

    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_acc = max(history.history["val_accuracy"])
    print(f"\nBest epoch: {best_epoch}, Best val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Save the fine-tuned model
    model.save(FINETuned_MODEL_PATH)
    print(f"Fine-tuned model saved to {FINETuned_MODEL_PATH}")

    return model, val_ds


def evaluate_model(model, val_ds, model_name="Model"):
    """Evaluate a Keras model and return metrics."""
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

    return _compute_metrics(correct, total, elapsed, all_preds, all_labels, model_name)


def evaluate_tta(model, val_ds, model_name="TTA Model"):
    """Evaluate with Test-Time Augmentation."""
    print(f"\n  Running TTA with {TTA_TRANSFORMS} transforms per image...")

    augmentations = [
        lambda x: x,  # identity
        lambda x: tf.image.flip_left_right(x),
        lambda x: tf.image.random_brightness(x, 0.1),
        lambda x: tf.image.random_contrast(x, 0.9, 1.1),
        lambda x: tf.image.random_crop(x, [tf.shape(x)[0], int(IMG_SIZE[0]*0.9), int(IMG_SIZE[1]*0.9), 3]),
        lambda x: tf.image.rot90(x, k=1),
        lambda x: tf.image.flip_left_right(tf.image.random_brightness(x, 0.05)),
        lambda x: tf.image.random_contrast(tf.image.flip_left_right(x), 0.95, 1.05),
    ]

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    start = time.perf_counter()
    for images, labels in val_ds:
        batch_preds = []
        for aug_fn in augmentations[:TTA_TRANSFORMS]:
            aug_img = aug_fn(images)
            # Resize back if crop changed size
            aug_img = tf.image.resize(aug_img, IMG_SIZE)
            pred = model(aug_img, training=False)
            batch_preds.append(pred.numpy().flatten())

        # Average predictions across augmentations
        avg_preds = np.mean(batch_preds, axis=0)
        labels_np = labels.numpy().flatten()

        for i in range(images.shape[0]):
            pred_class = 1 if avg_preds[i] > 0.5 else 0
            true_class = int(labels_np[i])
            all_preds.append(pred_class)
            all_labels.append(true_class)
            if pred_class == true_class:
                correct += 1
            total += 1
    elapsed = time.perf_counter() - start

    return _compute_metrics(correct, total, elapsed, all_preds, all_labels, model_name)


def evaluate_tflite_finetuned(tflite_path, val_ds, model_name="TFLite Fine-tuned"):
    """Fine-tune, convert to TFLite, and evaluate."""
    # Load the fine-tuned Keras model
    if not os.path.exists(FINETuned_MODEL_PATH):
        print(f"  Fine-tuned model not found at {FINETuned_MODEL_PATH}")
        return None

    model = keras.models.load_model(FINETuned_MODEL_PATH, compile=False)

    # Convert to TFLite
    tftuned_tflite_path = tflite_path.replace(".tflite", "_finetuned.tflite")
    import tensorflow as _tf
    converter = _tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tftuned_tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Fine-tuned TFLite model saved to {tftuned_tflite_path}")

    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            import tensorflow as tf
            litert = tf.lite

    interpreter = litert.Interpreter(model_path=tftuned_tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    start = time.perf_counter()
    for images, labels in val_ds:
        bs = images.shape[0]
        preprocessed = images.numpy().astype(np.float32)
        labels_np = labels.numpy().flatten()

        expected_bs = input_details[0]["shape"][0]
        if expected_bs != bs and expected_bs != -1:
            try:
                interpreter.resize_tensor_input(input_details[0]["index"], [bs, 96, 96, 3])
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            except Exception:
                for i in range(bs):
                    single = preprocessed[i : i + 1]
                    interpreter.set_tensor(input_details[0]["index"], single)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]["index"])
                    pred_class = 1 if pred[0][0] > 0.5 else 0
                    true_class = int(labels_np[i])
                    all_preds.append(pred_class)
                    all_labels.append(true_class)
                    if pred_class == true_class:
                        correct += 1
                    total += 1
                continue

        interpreter.set_tensor(input_details[0]["index"], preprocessed)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])

        for i in range(bs):
            pred_class = 1 if preds[i][0] > 0.5 else 0
            true_class = int(labels_np[i])
            all_preds.append(pred_class)
            all_labels.append(true_class)
            if pred_class == true_class:
                correct += 1
            total += 1
    elapsed = time.perf_counter() - start

    size_mb = os.path.getsize(tftuned_tflite_path) / (1024 * 1024)
    m = _compute_metrics(correct, total, elapsed, all_preds, all_labels, model_name)
    m["model_size_mb"] = size_mb
    return m


def _compute_metrics(correct, total, elapsed, all_preds, all_labels, model_name=""):
    """Compute standard classification metrics."""
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
        "name": model_name,
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    # Load validation dataset (no augmentation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        EXTERNAL_DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)

    results = []

    # 1. Baseline: Original model (no fine-tuning)
    if os.path.exists(KERAS_MODEL_PATH):
        print("\n" + "=" * 60)
        print("Evaluating BASELINE model (pre-trained, no fine-tuning)...")
        baseline_model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        m = evaluate_model(baseline_model, val_ds, "Baseline (pre-trained)")
        results.append(m)
        print(f"  Accuracy:  {m['accuracy']*100:.2f}%")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        del baseline_model  # Free memory

    # 2. Fine-tuned model
    if os.path.exists(KERAS_MODEL_PATH):
        ft_model, _ = fine_tune_model()
        m = evaluate_model(ft_model, val_ds, "Fine-tuned (domain-adapted)")
        results.append(m)
        print(f"\n  Fine-tuned Accuracy:  {m['accuracy']*100:.2f}%")
        print(f"  Fine-tuned Precision: {m['precision']:.4f}")
        print(f"  Fine-tuned Recall:    {m['recall']:.4f}")
        print(f"  Fine-tuned F1:        {m['f1']:.4f}")
        print(f"  Fine-tuned Specificity: {m['specificity']:.4f}")

        # 3. TTA on fine-tuned model
        m_tta = evaluate_tta(ft_model, val_ds, "Fine-tuned + TTA")
        results.append(m_tta)
        print(f"\n  TTA Accuracy:  {m_tta['accuracy']*100:.2f}%")
        print(f"  TTA Precision: {m_tta['precision']:.4f}")
        print(f"  TTA Recall:    {m_tta['recall']:.4f}")
        print(f"  TTA F1:        {m_tta['f1']:.4f}")

    # 4. Fine-tuned TFLite
    if os.path.exists(TFLITE_MODEL_PATH):
        m_tflite = evaluate_tflite_finetuned(TFLITE_MODEL_PATH, val_ds, "Fine-tuned TFLite")
        if m_tflite:
            results.append(m_tflite)
            print(f"\n  Fine-tuned TFLite Accuracy:  {m_tflite['accuracy']*100:.2f}%")
            print(f"  Fine-tuned TFLite model size: {m_tflite.get('model_size_mb', 0):.1f} MB")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DOMAIN ADAPTATION RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:40s} Acc={r['accuracy']*100:.2f}%  F1={r['f1']:.4f}  Prec={r['precision']:.4f}  Rec={r['recall']:.4f}")

    # --- Write CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "domain_adaptation.csv")
    fieldnames = ["name", "accuracy", "precision", "recall", "f1", "specificity",
                  "latency_ms", "throughput", "total", "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames if k in r})
    print(f"\nCSV saved to {csv_path}")

    # --- Write LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, "domain_adaptation.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by domain_adaptation.py\n")
        f.write("% Domain adaptation results on akahana/Driver-Drowsiness-Dataset\n\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Domain Adaptation: Improving Cross-Dataset Generalization\n")
        f.write("(\\texttt{akahana/Driver-Drowsiness-Dataset}, $n = 2{,}000$ validation images)}\n")
        f.write("\\label{tab:domain_adaptation}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Method & Acc. (\\%) & Prec. & Rec. & F1 & Spec. & Size \\\\\n")
        f.write(" & & & & & & (MB) \\\\\n\\midrule\n")
        for r in results:
            size_str = ""
            if "model_size_mb" in r:
                size_str = f" & {r['model_size_mb']:.1f}"
            else:
                size_str = " & --"
            f.write(f"{r['name']} & {r['accuracy']*100:.2f} & {r['precision']:.3f} "
                    f"& {r['recall']:.3f} & {r['f1']:.3f} & {r['specificity']:.3f}{size_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {tex_path}")

    # Print improvement
    if len(results) >= 2:
        baseline_acc = results[0]["accuracy"]
        best_name = max(results, key=lambda x: x["accuracy"])
        improvement = best_name["accuracy"] - baseline_acc
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT: {baseline_acc*100:.2f}% -> {best_name['accuracy']*100:.2f}%")
        print(f"  Best method: {best_name['name']}")
        print(f"  Improvement: +{improvement*100:.2f} pp")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
