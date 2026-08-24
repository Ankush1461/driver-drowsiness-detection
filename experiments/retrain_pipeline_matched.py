"""
Standalone script to retrain the model on SSD face-cropped data.
This closes the 12 pp gap between full-image (99.8%) and pipeline (87.9%) accuracy.

Usage:
    python experiments/retrain_pipeline_matched.py

Requirements:
    - TensorFlow 2.x
    - OpenCV (for SSD face detection)
    - external_dataset/combined_cropped/ (pre-cropped face data)

Output:
    - drowsiness_pipeline_v2.keras
    - drowsiness_pipeline_v2.tflite (target: ~4.4 MB)
    - results/pipeline_v2_evaluation.csv
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import glob
import random
import time

# === CONFIG ===
IMG = 96
BATCH = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
SEED = 42
DATA_DIR = 'external_dataset/combined_cropped'

# === LOAD DATA ===
print("=" * 60)
print("Step 1: Loading face-cropped data")
print("=" * 60)

# Load akahana crops (PNG) - these are the target domain
akahana_active = sorted([f for f in glob.glob(f"{DATA_DIR}/active/*.png") if 'img_' in f])
akahana_fatigue = sorted([f for f in glob.glob(f"{DATA_DIR}/fatigue/*.png") if 'img_' in f])

# Also load manith crops (JPG) for diversity
manith_active = sorted(glob.glob(f"{DATA_DIR}/active/*.jpg"))
manith_fatigue = sorted(glob.glob(f"{DATA_DIR}/fatigue/*.jpg"))

print(f"  Akahana: {len(akahana_active)} active, {len(akahana_fatigue)} fatigue")
print(f"  Manith:  {len(manith_active)} active, {len(manith_fatigue)} fatigue")

# Use all akahana + subsample manith for balance
all_active = akahana_active + random.sample(manith_active, min(len(manith_active), len(akahana_active)))
all_fatigue = akahana_fatigue + random.sample(manith_fatigue, min(len(manith_fatigue), len(akahana_fatigue)))

files = [(f, 0) for f in all_active] + [(f, 1) for f in all_fatigue]
random.seed(SEED)
random.shuffle(files)

n_val = int(len(files) * VAL_SPLIT)
val_files = files[:n_val]
train_files = files[n_val:]

print(f"  Total: {len(files)} images")
print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

# === LOAD IMAGES ===
print("\nLoading images into memory...")
t0 = time.time()

def load_images(file_list):
    images, labels = [], []
    for fp, lbl in file_list:
        img = cv2.imread(fp)
        if img is None:
            continue
        # Convert BGR to RGB (matching production app.py)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(rgb.astype(np.float32))
        labels.append(lbl)
    return np.array(images), np.array(labels)

X_train, y_train = load_images(train_files)
X_val, y_val = load_images(val_files)
print(f"  Loaded in {time.time()-t0:.0f}s")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
print(f"  Train class balance: {np.mean(y_train):.3f} fatigue rate")

# === BUILD MODEL ===
print("\nBuilding MobileNetV3Large model...")
base = keras.applications.MobileNetV3Large(
    input_shape=(IMG, IMG, 3),
    include_top=False,
    weights='imagenet'
)

# Fine-tune last 30 layers
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='swish'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=['accuracy', keras.metrics.Precision(name='prec'), keras.metrics.Recall(name='rec')]
)

model.summary()

# === TRAIN ===
print("\nTraining...")
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        monitor='val_loss'
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks
)

# === SAVE KERAS ===
model.save('drowsiness_pipeline_v2.keras')
print("\nSaved drowsiness_pipeline_v2.keras")

# === CONVERT TO TFLITE ===
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('drowsiness_pipeline_v2.tflite', 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"Saved drowsiness_pipeline_v2.tflite ({size_mb:.2f} MB)")

# === EVALUATE ON FULL PIPELINE ===
print("\n" + "=" * 60)
print("Step 2: Evaluating on full pipeline (SSD -> Crop -> RGB -> Classify)")
print("=" * 60)

# Load TFLite interpreter
from tensorflow.lite.python.interpreter import Interpreter
interp = Interpreter(model_path='drowsiness_pipeline_v2.tflite')
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']

# Load face detector
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load akahana test set
active_test = sorted(glob.glob('external_dataset/akahana/active/*.png'))
fatigue_test = sorted(glob.glob('external_dataset/akahana/fatigue/*.png'))
test_files = [(f, 0) for f in active_test] + [(f, 1) for f in fatigue_test]
random.seed(42)
random.shuffle(test_files)

print(f"Evaluating on {len(test_files)} akahana images...")

tp = fp = tn = fn = face_miss = 0
for i, (fp_, label) in enumerate(test_files):
    img = cv2.imread(fp_)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # SSD face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    det = net.forward()
    
    best_conf, best_box = 0, None
    for j in range(det.shape[2]):
        c = det[0, 0, j, 2]
        if c > 0.5 and c > best_conf:
            best_conf = c
            best_box = det[0, 0, j, 3:7] * np.array([w, h, w, h])
    
    if best_box is None:
        face_miss += 1
        if label == 0: tn += 1
        else: fn += 1
        continue
    
    x1, y1, x2, y2 = best_box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    
    # Preprocess: resize + BGR->RGB (matching app.py)
    resized = cv2.resize(crop, (IMG, IMG))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb.astype(np.float32), 0)
    
    # Classify
    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    pred = float(interp.get_tensor(out_idx)[0][0])
    pred_label = 1 if pred > 0.5 else 0
    
    if pred_label == 1 and label == 1: tp += 1
    elif pred_label == 1 and label == 0: fp += 1
    elif pred_label == 0 and label == 0: tn += 1
    else: fn += 1
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(test_files)} done...")

total = tp + fp + tn + fn
acc = (tp + tn) / total * 100
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-8)
spec = tn / max(tn + fp, 1)

print(f"\n{'='*60}")
print(f"RESULTS: Full Pipeline (SSD -> Crop -> RGB -> Classify)")
print(f"{'='*60}")
print(f"Accuracy:   {acc:.2f}%")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"Specificity:{spec:.4f}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"Face misses: {face_miss}")
print(f"Model size: {size_mb:.2f} MB")

# Save results
os.makedirs('results', exist_ok=True)
with open('results/pipeline_v2_evaluation.csv', 'w') as f:
    f.write('metric,value\n')
    f.write(f'accuracy,{acc/100:.4f}\n')
    f.write(f'precision,{prec:.4f}\n')
    f.write(f'recall,{rec:.4f}\n')
    f.write(f'f1,{f1:.4f}\n')
    f.write(f'specificity,{spec:.4f}\n')
    f.write(f'tp,{tp}\n')
    f.write(f'fp,{fp}\n')
    f.write(f'tn,{tn}\n')
    f.write(f'fn,{fn}\n')
    f.write(f'face_misses,{face_miss}\n')
    f.write(f'n,{total}\n')
    f.write(f'model_size_mb,{size_mb:.2f}\n')

print(f"\nSaved results/pipeline_v2_evaluation.csv")
print(f"\nDone! Run this script periodically to track improvement.")
