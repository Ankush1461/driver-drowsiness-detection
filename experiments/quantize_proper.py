import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np, cv2, glob

DATASET = "D:/Programming/driver_drowsiness_2/dataset/train_cropped"
model = tf.keras.models.load_model("drowsiness.keras")
print(f"Model loaded: {model.count_params()} params")

# Get calibration data (200 images from validation set)
active = glob.glob(os.path.join(DATASET, "Active Subjects/*.jpg"))
fatigue = glob.glob(os.path.join(DATASET, "Fatigue Subjects/*.jpg"))
rng = np.random.RandomState(42)
all_files = [(f, 0) for f in active] + [(f, 1) for f in fatigue]
rng.shuffle(all_files)
cal_files = all_files[:200]
val_files = all_files[200:520]

def calibrate():
    for fp, _ in cal_files:
        raw = cv2.imread(fp)
        if raw is None: continue
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (96, 96)).astype(np.float32)
        preprocessed = (resized / 127.5) - 1.0
        yield [np.expand_dims(preprocessed, 0)]

# Method: Dynamic range quantization with float16 fallback
print("\nConverting with float16 quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite = converter.convert()
with open("drowsiness_f16.tflite", "wb") as f: f.write(tflite)
print(f"  Size: {len(tflite)/1024/1024:.1f} MB")

# Evaluate
from tensorflow.lite.python.interpreter import Interpreter
interp = Interpreter(model_path="drowsiness_f16.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print(f"  Input dtype: {inp[0]['dtype']}, Output dtype: {out[0]['dtype']}")

preds = []; labels = []
for fp, label in val_files:
    raw = cv2.imread(fp)
    if raw is None: continue
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (96,96)).astype(np.float32)
    preprocessed = (resized / 127.5) - 1.0
    interp.set_tensor(inp[0]['index'], np.expand_dims(preprocessed, 0))
    interp.invoke()
    sig = float(interp.get_tensor(out[0]['index'])[0][0])
    preds.append(sig); labels.append(label)

preds = np.array(preds); labels = np.array(labels)
a = preds[labels==0]; f = preds[labels==1]
best_acc = 0; best_thr = 0.5
for thr in np.arange(0.1, 0.9, 0.01):
    tp = np.sum((preds>=thr)&(labels==1)); tn = np.sum((preds<thr)&(labels==0))
    acc = (tp+tn)/len(preds)*100
    if acc > best_acc: best_acc = acc; best_thr = thr
tp = np.sum((preds>=best_thr)&(labels==1)); tn = np.sum((preds<best_thr)&(labels==0))
fp = np.sum((preds>=best_thr)&(labels==0)); fn = np.sum((preds<best_thr)&(labels==1))
prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0; rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"  Active mu={a.mean():.4f} Fatigue mu={f.mean():.4f}")
print(f"  Best: thr={best_thr:.2f} Acc={best_acc:.2f}% Prec={prec:.2f}% Rec={rec:.2f}% F1={f1:.2f}%")
