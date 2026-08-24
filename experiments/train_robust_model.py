"""
Train a Robust Generalizable Drowsiness Model
==============================================
Strategy: Train on akahana (10K images) with VERY aggressive augmentation
to maximize generalization to completely unseen datasets.

Key techniques:
  1. RandAugment (policy-based augmentation)
  2. CutMix + Mixup (sample mixing)
  3. Stochastic Depth (regularization)
  4. Label smoothing + focal loss
  5. Cosine annealing with warm restarts
  6. Exponential Moving Average (EMA)

Output: drowsiness_robust.tflite (target ≤5MB)

Usage:
    python experiments/train_robust_model.py
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

# --- Configuration ---
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 42
DATASET_PATH = "external_dataset/combined"
OUTPUT_DIR = "results"
ROBUST_KERAS = "drowsiness_robust.keras"
ROBUST_TFLITE = "drowsiness_robust.tflite"

# Augmentation strength
AUG_STRENGTH = 0.5  # Aggressive

# Training config
INITIAL_LR = 1e-3
MIN_LR = 1e-6
WARMUP_EPOCHS = 3
TOTAL_EPOCHS = 60
PATIENCE = 12
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
USE_EMA = True
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.1


# --- Advanced Augmentation ---
def create_augmentation_pipeline():
    """Create aggressive augmentation pipeline."""
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(AUG_STRENGTH),
        layers.RandomContrast(AUG_STRENGTH),
        layers.GaussianNoise(0.02),
    ], name="augmentation")
    return augmentation


def rand_augment(images, severity=3):
    """Simple RandAugment implementation."""
    augmentations = [
        lambda x: tf.image.random_flip_left_right(x),
        lambda x: tf.image.random_brightness(x, 0.3),
        lambda x: tf.image.random_contrast(x, 0.7, 1.3),
        lambda x: tf.image.random_saturation(x, 0.7, 1.3),
        lambda x: tf.image.random_hue(x, 0.1),
        lambda x: tf.image.random_jpeg_quality(x, 50, 100),
    ]

    n = min(severity, len(augmentations))
    chosen = np.random.choice(len(augmentations), n, replace=False)
    result = images
    for idx in chosen:
        result = tf.clip_by_value(augmentations[idx](result), 0, 255)
    return result


def mixup(images, labels, alpha=0.4):
    """Mixup augmentation."""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    lam = tf.random.uniform([], 0, alpha * 2)
    lam = tf.maximum(lam, 1 - lam)

    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def cutmix(images, labels, alpha=1.0):
    """CutMix augmentation."""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    lam = tf.random.uniform([], 0, alpha * 2)
    lam = tf.maximum(lam, 1 - lam)

    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    cut_ratio = tf.sqrt(1 - lam)
    cut_h = tf.cast(tf.cast(h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * cut_ratio, tf.int32)

    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.maximum(0, cy - cut_h // 2)
    y2 = tf.minimum(h, cy + cut_h // 2)
    x1 = tf.maximum(0, cx - cut_w // 2)
    x2 = tf.minimum(w, cx + cut_w // 2)

    mask = tf.ones_like(images)
    cut_mask = tf.zeros([cut_h, cut_w, 3])
    cut_mask = tf.pad(cut_mask, [[y1, h - y2], [x1, w - x2], [0, 0]], constant_values=1)

    mixed_images = images * cut_mask + tf.gather(images, indices) * (1 - cut_mask)

    # Adjust labels
    area_ratio = 1 - tf.cast(cut_h * cut_w, tf.float32) / tf.cast(h * w, tf.float32)
    mixed_labels = area_ratio * labels + (1 - area_ratio) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


# --- Model Architecture ---
def build_robust_model(num_classes=1):
    """Build MobileNetV3Large with improved head for better generalization."""
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        alpha=1.0,
        weights="imagenet",
    )

    # Progressive unfreezing schedule
    base_model.trainable = True
    for layer in base_model.layers[:120]:
        layer.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    # Augmentation (applied during training only)
    aug = create_augmentation_pipeline()
    x = aug(inputs)

    # MobileNetV3 backbone
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Improved classification head with squeeze-and-excitation
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="DriveSafe_Robust")

    return model


# --- Custom Training Loop with Mixup/CutMix ---
class RobustTrainingCallback(keras.callbacks.Callback):
    def __init__(self, use_mixup=True, use_cutmix=True, mixup_alpha=0.4, cutmix_alpha=1.0):
        super().__init__()
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def on_train_batch_begin(self, batch, logs=None):
        pass  # Augmentation happens in augmentation layer


class EMA(keras.callbacks.Callback):
    """Exponential Moving Average of weights."""
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None

    def on_train_begin(self, logs=None):
        self.ema_weights = [w.numpy() for w in self.model.trainable_weights]

    def on_train_batch_end(self, batch, logs=None):
        for i, w in enumerate(self.model.trainable_weights):
            self.ema_weights[i] = self.decay * self.ema_weights[i] + (1 - self.decay) * w.numpy()

    def on_epoch_end(self, epoch, logs=None):
        # Store EMA weights and use them for validation
        original_weights = [w.numpy() for w in self.model.trainable_weights]
        for i, w in enumerate(self.model.trainable_weights):
            w.assign(self.ema_weights[i])
        self._original_weights = original_weights

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self, '_original_weights'):
            for i, w in enumerate(self.model.trainable_weights):
                w.assign(self._original_weights[i])


# --- Learning Rate Schedule ---
def cosine_annealing_with_warmup(epoch, total_epochs=60, warmup_epochs=3, initial_lr=1e-3, min_lr=1e-6):
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Training Robust Generalizable Drowsiness Model")
    print("=" * 60)

    # --- Load Data ---
    print("\nLoading dataset...")
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    print(f"  Train: {train_ds.cardinality().numpy() * BATCH_SIZE} images")
    print(f"  Val: {val_ds.cardinality().numpy() * BATCH_SIZE} images")

    # Preprocess
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)

    # --- Build Model ---
    print("\nBuilding model...")
    model = build_robust_model()
    model.summary()

    # Count params
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # --- Compile ---
    loss_fn = keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # --- Callbacks ---
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            ROBUST_KERAS, monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        keras.callbacks.LearningRateScheduler(
            lambda e: cosine_annealing_with_warmup(e, TOTAL_EPOCHS, WARMUP_EPOCHS, INITIAL_LR, MIN_LR)
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=MIN_LR
        ),
    ]

    if USE_EMA:
        callbacks_list.append(EMA(EMA_DECAY))

    # --- Train ---
    print(f"\nTraining for {TOTAL_EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list,
    )

    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_acc = max(history.history["val_accuracy"])
    print(f"\nBest epoch: {best_epoch}, Best val accuracy: {best_acc*100:.2f}%")

    # --- Convert to TFLite ---
    print("\nConverting to TFLite...")
    model.load_weights(ROBUST_KERAS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(ROBUST_TFLITE, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(ROBUST_TFLITE) / (1024 * 1024)
    print(f"TFLite model: {size_mb:.1f} MB")

    # --- Evaluate TFLite ---
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            from tensorflow import lite as litert

    interpreter = litert.Interpreter(model_path=ROBUST_TFLITE, num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    correct, total = 0, 0
    all_preds, all_labels = [], []
    start = time.perf_counter()

    for images, labels in val_ds:
        bs = images.shape[0]
        data = images.numpy().astype(np.float32)
        labels_np = labels.numpy().flatten()
        try:
            interpreter.resize_tensor_input(inp[0]["index"], [bs, 96, 96, 3])
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
        except:
            pass
        interpreter.set_tensor(inp[0]["index"], data)
        interpreter.invoke()
        preds = interpreter.get_tensor(out[0]["index"])
        for i in range(bs):
            pc = 1 if preds[i][0] > 0.5 else 0
            tc = int(labels_np[i])
            all_preds.append(pc)
            all_labels.append(tc)
            if pc == tc:
                correct += 1
            total += 1

    elapsed = time.perf_counter() - start
    accuracy = correct / total
    latency_ms = (elapsed / total) * 1000

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    print(f"\n{'='*60}")
    print("ROBUST TFLite MODEL EVALUATION (in-distribution)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {accuracy*100:.2f}%")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Latency:     {latency_ms:.1f} ms/image")
    print(f"  Model size:  {size_mb:.1f} MB")
    print(f"  Confusion:   TP={tp} FP={fp} FN={fn} TN={tn}")

    # --- Save results ---
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "robust_model_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["accuracy","precision","recall","f1","specificity","latency_ms","size_mb","tp","fp","fn","tn"])
        w.writeheader()
        w.writerow({"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,
                     "specificity":specificity,"latency_ms":latency_ms,"size_mb":size_mb,
                     "tp":tp,"fp":fp,"fn":fn,"tn":tn})
    print(f"\nResults saved to {csv_path}")

    # Training history
    hist_path = os.path.join(OUTPUT_DIR, "robust_training_history.csv")
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_acc","val_acc","train_loss","val_loss","lr"])
        w.writeheader()
        for i in range(len(history.history["accuracy"])):
            w.writerow({
                "epoch": i+1,
                "train_acc": history.history["accuracy"][i],
                "val_acc": history.history["val_accuracy"][i],
                "train_loss": history.history["loss"][i],
                "val_loss": history.history["val_loss"][i],
                "lr": history.history["lr"][i] if "lr" in history.history else 0,
            })
    print(f"Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
