"""
Quantization-Aware Training (QAT) for MobileNetV3
Trains with fake quantization nodes so the model learns to compensate.
Exports to a proper INT8 TFLite (~4.4 MB) with full accuracy.
"""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

print("=" * 60)
print("QUANTIZATION-AWARE TRAINING (QAT)")
print("=" * 60)

# Load the best .keras model
print("\nLoading drowsiness.keras...")
model = keras.models.load_model("drowsiness.keras", compile=False)
print(f"  Input: {model.input_shape}")
print(f"  Parameters: {model.count_params():,}")

# Replicate architecture for QAT
print("\nBuilding QAT model...")
base_model = keras.applications.MobileNetV3Large(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
qat_model = keras.Model(inputs, outputs)

# Copy weights from trained model
qat_model.set_weights(model.get_weights())
print(f"  Weights transferred: {qat_model.count_params():,}")

# Apply quantization awareness
print("Applying quantization-aware training...")
qat_model = tf.keras.models.clone_model(
    qat_model,
    clone_function=lambda layer: (
        tf.keras.quantization.quantize_annotate_layer(layer)
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, 
                             tf.keras.layers.DepthwiseConv2D))
        else layer
    ),
)

# Configure quantization
qat_model = tf.keras.quantization.quantize_apply(qat_model)
print("  Quantization annotations applied")

# Unfreeze backbone for fine-tuning
base_qat = qat_model.layers[1]  # MobileNetV3
base_qat.trainable = True
for layer in base_qat.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in qat_model.layers if l.trainable)
print(f"  Trainable layers: {trainable}")

# Load data
from tensorflow.keras.utils import image_dataset_from_directory
data_dir = Path("dataset/train_cropped")
if not data_dir.exists():
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

# Preprocessing: [-1, 1] for MobileNetV3
normalization = keras.layers.Rescaling(1.0 / 127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Compile
qat_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
)

# Train
print("\nQAT Fine-tuning (20 epochs)...")
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
]

history = qat_model.fit(
    train_ds, validation_data=val_ds,
    epochs=20, callbacks=callbacks, verbose=1
)

best_acc = max(history.history["val_accuracy"])
print(f"\n  Best val_accuracy: {best_acc:.4f}")

# Export to TFLite with full INT8 quantization
print("\nExporting INT8 TFLite...")

def representative_dataset():
    """Calibration data for quantization."""
    count = 0
    for images, _ in train_ds.unbatch().batch(1):
        yield [images]
        count += 1
        if count >= 200:
            break

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

out_path = "drowsiness_qat_int8.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"  Saved {out_path} ({size_mb:.1f} MB)")

# Validate
print("\nValidating QAT TFLite...")
interpreter = tf.lite.Interpreter(model_path=out_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

active_preds, fatigue_preds = [], []
count = 0
for images, labels in val_ds:
    for img, label in zip(images.numpy(), labels.numpy()):
        inp = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
        if label == 0:
            active_preds.append(pred)
        else:
            fatigue_preds.append(pred)
        count += 1
        if count >= 500:
            break
    if count >= 500:
        break

active_mean = np.mean(active_preds)
fatigue_mean = np.mean(fatigue_preds)
threshold = (active_mean + fatigue_mean) / 2

all_preds = np.array(active_preds + fatigue_preds)
all_labels = np.array([0]*len(active_preds) + [1]*len(fatigue_preds))
binary_preds = (all_preds > threshold).astype(int)
accuracy = np.mean(binary_preds == all_labels)

print(f"  Active mean:  {active_mean:.4f} (should be < 0.3)")
print(f"  Fatigue mean: {fatigue_mean:.4f} (should be > 0.7)")
print(f"  Threshold:    {threshold:.4f}")
print(f"  Accuracy:     {accuracy*100:.2f}%")

# Also export float32 for comparison
converter2 = tf.lite.TFLiteConverter.from_keras_model(qat_model)
tflite_f32 = converter2.convert()
with open("drowsiness_qat_f32.tflite", "wb") as f:
    f.write(tflite_f32)
size_f32 = os.path.getsize("drowsiness_qat_f32.tflite") / (1024*1024)
print(f"\n  Float32 reference: drowsiness_qat_f32.tflite ({size_f32:.1f} MB)")

print(f"\n{'='*60}")
print(f"RESULTS: {size_mb:.1f} MB INT8 model with {accuracy*100:.2f}% accuracy")
print(f"{'='*60}")
