import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np, glob, random, cv2, time

BATCH = 64
EPOCHS = 8
IMG = 96
DATA = "external_dataset/combined_cropped"

active = sorted(glob.glob(f"{DATA}/active/*.jpg") + glob.glob(f"{DATA}/active/*.png"))
fatigue = sorted(glob.glob(f"{DATA}/fatigue/*.jpg") + glob.glob(f"{DATA}/fatigue/*.png"))
random.seed(42)
act_s = random.sample(active, 5000)
fat_s = random.sample(fatigue, 5000)
files = [(f, 0) for f in act_s] + [(f, 1) for f in fat_s]
random.shuffle(files)
n_val = 1500
val_files = files[:n_val]
train_files = files[n_val:]
print(f"Train: {len(train_files)}, Val: {len(val_files)}")

def load_batch(file_list):
    imgs, lbls = [], []
    for fp, lbl in file_list:
        img = cv2.imread(fp)
        if img is None: continue
        img = cv2.resize(img, (IMG, IMG))
        imgs.append(img.astype(np.float32))  # [0,255]
        lbls.append(lbl)
    return np.array(imgs), np.array(lbls, dtype=np.int32)

print("Loading train data...")
t0 = time.time()
X_train, y_train = load_batch(train_files)
print(f"Loaded train in {time.time()-t0:.1f}s: {X_train.shape}")

print("Loading val data...")
X_val, y_val = load_batch(val_files)
print(f"Loaded val in {time.time()-t0:.1f}s: {X_val.shape}")

base = keras.applications.MobileNetV3Large(input_shape=(IMG,IMG,3), include_top=False, weights="imagenet")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model = keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=["accuracy"])
model.summary()

print("Training...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")])

model.save("drowsiness_pipeline_matched.keras")
print("Saved Keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open("drowsiness_pipeline_matched.tflite", "wb") as f:
    f.write(tflite)
print(f"Saved TFLite: {len(tflite)/1024/1024:.2f} MB")

# Quick eval
from tensorflow.lite.python.interpreter import Interpreter
interp = Interpreter(model_path="drowsiness_pipeline_matched.tflite")
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]["index"]
out_idx = interp.get_output_details()[0]["index"]
correct = tp = fp = tn = fn = 0
for img_arr, lbl in zip(X_val, y_val):
    arr = np.expand_dims(img_arr, 0)
    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    pred = 1 if float(interp.get_tensor(out_idx)[0][0]) > 0.5 else 0
    if pred == lbl: correct += 1
    if pred==1 and lbl==1: tp+=1
    elif pred==1 and lbl==0: fp+=1
    elif pred==0 and lbl==0: tn+=1
    else: fn+=1
total = tp+fp+tn+fn
print(f"TFLite val: {correct}/{total} = {correct/total*100:.2f}%")
print(f"P={tp/max(tp+fp,1):.4f} R={tp/max(tp+fn,1):.4f} F1={2*tp/max(2*tp+fp+fn,1):.4f} Sp={tn/max(tn+fp,1):.4f}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
