import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np, glob, random

BATCH = 32
EPOCHS = 15
IMG = 96
DATA = 'external_dataset/combined_cropped'
VAL_SPLIT = 0.15

def load_data():
    active = sorted(glob.glob(f'{DATA}/active/*.jpg') + glob.glob(f'{DATA}/active/*.png'))
    fatigue = sorted(glob.glob(f'{DATA}/fatigue/*.jpg') + glob.glob(f'{DATA}/fatigue/*.png'))
    files = [(f, 0) for f in active] + [(f, 1) for f in fatigue]
    random.seed(42)
    random.shuffle(files)
    n_val = int(len(files) * VAL_SPLIT)
    return files[n_val:], files[:n_val]

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img

def make_dataset(files, training=False):
    def gen():
        for fp, lbl in files:
            raw = tf.io.read_file(fp)
            img = tf.io.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, [IMG, IMG])
            # Keep [0,255] like production app.py
            if training:
                img = augment(img)
            yield img, lbl
    ds = tf.data.Dataset.from_generator(gen,
        output_signature=(tf.TensorSpec((IMG, IMG, 3), tf.float32),
                          tf.TensorSpec((), tf.int32)))
    if training:
        ds = ds.shuffle(5000)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def build_model():
    base = keras.applications.MobileNetV3Large(
        input_shape=(IMG, IMG, 3), include_top=False, weights='imagenet')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['accuracy', keras.metrics.Precision(name='prec'),
                 keras.metrics.Recall(name='rec')])
    return model

print("Loading data...")
train_files, val_files = load_data()
print(f"Train: {len(train_files)}, Val: {len(val_files)}")

train_ds = make_dataset(train_files, training=True)
val_ds = make_dataset(val_files, training=False)

print("Building model...")
model = build_model()
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                   monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7)
]

print("Training...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Evaluate
print("\nEvaluating on validation set...")
results = model.evaluate(val_ds, verbose=0)
print(f"Val Accuracy: {results[1]:.4f}, Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")

# Save Keras
model.save('drowsiness_pipeline_matched.keras')
print("Saved drowsiness_pipeline_matched.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open('drowsiness_pipeline_matched.tflite', 'wb') as f:
    f.write(tflite)
size_mb = len(tflite) / (1024*1024)
print(f"Saved drowsiness_pipeline_matched.tflite ({size_mb:.2f} MB)")

# Quick TFLite eval
from tensorflow.lite.python.interpreter import Interpreter
interp = Interpreter(model_path='drowsiness_pipeline_matched.tflite')
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']

correct = tp = fp = tn = fn = 0
for fp_, lbl in val_files:
    import cv2
    img = cv2.imread(fp_)
    if img is None: continue
    resized = cv2.resize(img, (IMG, IMG))
    arr = np.expand_dims(resized.astype(np.float32), 0)  # [0,255]
    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    pred = 1 if float(interp.get_tensor(out_idx)[0][0]) > 0.5 else 0
    if pred == lbl: correct += 1
    if pred == 1 and lbl == 1: tp += 1
    elif pred == 1 and lbl == 0: fp += 1
    elif pred == 0 and lbl == 0: tn += 1
    else: fn += 1

total = tp + fp + tn + fn
print(f"\nTFLite eval: {correct}/{total} = {correct/total*100:.2f}%")
print(f"Prec={tp/max(tp+fp,1):.4f} Rec={tp/max(tp+fn,1):.4f} F1={2*tp/max(2*tp+fp+fn,1):.4f} Spec={tn/max(tn+fp,1):.4f}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
