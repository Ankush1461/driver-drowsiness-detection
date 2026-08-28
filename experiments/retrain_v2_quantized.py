"""
Train MobileNetV2 + post-training quantization.
MobileNetV2 uses ReLU (not swish) so quantization works properly.
Target: <5 MB TFLite with 98%+ accuracy.
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

print("=" * 60)
print("RETRAIN WITH MobileNetV2 + QUANTIZATION")
print("=" * 60)

# Build MobileNetV2 model (matches original architecture)
print("\nBuilding MobileNetV2 model...")
base = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=(96, 96, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

print(f"  Parameters: {model.count_params():,}")
print(f"  Keras model size: {sum(np.prod(w.shape) for w in model.weights) * 4 / 1024**2:.1f} MB")

# Copy compatible weights from V3 model
print("\nLoading V3 weights for transfer...")
v3_model = keras.models.load_model("drowsiness.keras", compile=False)

# Transfer what we can (skip incompatible layers)
v3_layers = [l for l in v3_model.layers if isinstance(l, (keras.layers.Dense, keras.layers.Dropout))]
v2_layers = [l for l in model.layers if isinstance(l, (keras.layers.Dense, keras.layers.Dropout))]

for v3l, v2l in zip(v3_layers, v2_layers):
    if v3l.get_weights() and v2l.get_weights():
        try:
            v2l.set_weights(v3l.get_weights())
            print(f"  Transferred: {v2l.name}")
        except:
            print(f"  Skipped (shape mismatch): {v2l.name}")

# Load data
from tensorflow.keras.utils import image_dataset_from_directory
data_dir = Path("D:/Programming/driver_drowsiness_2/dataset/train_cropped")
print(f"\nLoading data from {data_dir}...")

train_ds = image_dataset_from_directory(
    data_dir, seed=123, validation_split=0.2, subset="training",
    image_size=(96, 96), batch_size=32, label_mode="binary"
)
val_ds = image_dataset_from_directory(
    data_dir, seed=123, validation_split=0.2, subset="validation",
    image_size=(96, 96), batch_size=32, label_mode="binary"
)
print(f"  Classes: {train_ds.class_names}")
print(f"  Train: {train_ds.cardinality().numpy() * 32}, Val: {val_ds.cardinality().numpy() * 32}")

# Preprocessing
normalization = keras.layers.Rescaling(1.0 / 127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(2000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Phase 1: Train head
print("\nPHASE 1: Training head (frozen backbone, 15 epochs)")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
]

h1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, verbose=1)
print(f"  Phase 1 best: {max(h1.history['val_accuracy']):.4f}")

# Phase 2: Fine-tune
print("\nPHASE 2: Fine-tuning backbone (30 epochs)")
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"  Trainable layers: {trainable}")

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=["accuracy"]
)

h2 = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks, verbose=1)
best_val = max(h2.history["val_accuracy"])
print(f"  Phase 2 best: {best_val:.4f}")

# Save .keras
model.save("drowsiness_v2.keras")
print(f"\nSaved drowsiness_v2.keras")

# Validate .keras
print("\nValidating .keras model...")
active_preds, fatigue_preds = [], []
count = 0
for images, labels in val_ds:
    preds = model(images, training=False).numpy().flatten()
    for p, l in zip(preds, labels.numpy()):
        if l == 0: active_preds.append(p)
        else: fatigue_preds.append(p)
        count += 1
    if count >= 1000: break

a_mean, f_mean = np.mean(active_preds), np.mean(fatigue_preds)
thr = (a_mean + f_mean) / 2
all_p = np.array(active_preds + fatigue_preds)
all_l = np.array([0]*len(active_preds) + [1]*len(fatigue_preds))
acc = np.mean((all_p > thr).astype(int) == all_l)
print(f"  Active: {a_mean:.4f}, Fatigue: {f_mean:.4f}")
print(f"  Threshold: {thr:.4f}, Accuracy: {acc*100:.2f}%")

# Post-training quantization
print("\n--- POST-TRAINING QUANTIZATION ---")

# Method 1: Dynamic range (weights only)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()
with open("drowsiness_v2_dynamic.tflite", "wb") as f:
    f.write(tflite_dynamic)
sz = len(tflite_dynamic) / 1024**2
print(f"  Dynamic range: {sz:.1f} MB")

# Method 2: Float16
converter.target_spec.supported_types = [tf.float16]
tflite_f16 = converter.convert()
with open("drowsiness_v2_f16.tflite", "wb") as f:
    f.write(tflite_f16)
sz16 = len(tflite_f16) / 1024**2
print(f"  Float16: {sz16:.1f} MB")

# Method 3: INT8 with calibration
def representative_dataset():
    cnt = 0
    for imgs, _ in train_ds.unbatch().batch(1):
        yield [imgs]
        cnt += 1
        if cnt >= 200: break

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.float32
converter_int8.inference_output_type = tf.float32
tflite_int8 = converter_int8.convert()
with open("drowsiness_v2_int8.tflite", "wb") as f:
    f.write(tflite_int8)
sz8 = len(tflite_int8) / 1024**2
print(f"  INT8: {sz8:.1f} MB")

# Validate all TFLite models
print("\n--- TFLITE VALIDATION ---")
for name in ["drowsiness_v2_dynamic.tflite", "drowsiness_v2_f16.tflite", "drowsiness_v2_int8.tflite"]:
    interp = tf.lite.Interpreter(model_path=name)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    
    a_preds, f_preds = [], []
    cnt = 0
    for images, labels in val_ds:
        for img, lab in zip(images.numpy(), labels.numpy()):
            t = np.expand_dims(img, 0).astype(np.float32)
            interp.set_tensor(inp_det["index"], t)
            interp.invoke()
            p = interp.get_tensor(out_det["index"])[0][0]
            if lab == 0: a_preds.append(p)
            else: f_preds.append(p)
            cnt += 1
            if cnt >= 500: break
        if cnt >= 500: break
    
    am, fm = np.mean(a_preds), np.mean(f_preds)
    th = (am + fm) / 2
    all_p = np.array(a_preds + f_preds)
    all_l = np.array([0]*len(a_preds) + [1]*len(f_preds))
    a = np.mean((all_p > th).astype(int) == all_l)
    sz = os.path.getsize(name) / 1024**2
    status = "OK" if am < 0.3 and fm > 0.7 else "BROKEN"
    print(f"  {name}: {sz:.1f}MB | Act={am:.4f} Fat={fm:.4f} | Acc={a*100:.2f}% | {status}")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
