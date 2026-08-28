import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np, cv2, glob, zipfile, tempfile, time

UTA_DIR = "external_dataset/uta_rldd"
IMG_SIZE = 96
BATCH_SIZE = 16
FINE_TUNE_LR = 5e-5
EPOCHS = 20
STEPS_PER_VIDEO = 30  # More frames per video
SSD_PROTO = "deploy.prototxt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONF_THRESH = 0.5  # Lower threshold to get more faces
FACE_MIN_SIZE = 30

print("=" * 60)
print("UTA-RLDD FINE-TUNING (FIXED: V2 + SSD + CLAHE)")
print("=" * 60)

# CLAHE preprocessing (same as app.py and LOSO)
def apply_clahe(face_bgr):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def detect_faces(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    faces = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > FACE_CONF_THRESH:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                fh, fw = face.shape[:2]
                if fh < FACE_MIN_SIZE or fw < FACE_MIN_SIZE:
                    continue
                side = max(fh, fw)
                ph = (side - fh) // 2
                pw = (side - fw) // 2
                face = cv2.copyMakeBorder(face, ph, side-fh-ph, pw, side-fw-pw, cv2.BORDER_REFLECT)
                face = apply_clahe(face)  # CLAHE preprocessing!
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                faces.append(face)
    return faces

print("\nStep 1: Extracting SSD-cropped + CLAHE faces from UTA-RLDD...")
ssd_net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
ssd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ssd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

faces_data = []
labels_data = []
tmpdir = tempfile.mkdtemp(prefix="uta_frames_")
zips = sorted(glob.glob(os.path.join(UTA_DIR, "*.zip")))
print(f"  Found {len(zips)} zip files")

video_count = 0
for zp in zips:
    zf = zipfile.ZipFile(zp)
    video_names = [n for n in zf.namelist() if n.endswith((".mov", ".MOV", ".mp4", ".MP4"))]
    for vn in video_names:
        basename = os.path.basename(vn)
        level_str = basename.split(".")[0]
        try: level = int(level_str)
        except ValueError: continue
        label = 1 if level >= 8 else 0  # 0=alert, 1=drowsy
        vfile = os.path.join(tmpdir, f"v{video_count}.mp4")
        try:
            with zf.open(vn) as src, open(vfile, "wb") as dst: dst.write(src.read())
        except: continue
        cap = cv2.VideoCapture(vfile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: cap.release(); continue
        interval = max(1, int(fps))
        extracted = 0; frame_idx = 0
        while cap.isOpened() and extracted < STEPS_PER_VIDEO:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                detected = detect_faces(ssd_net, frame)
                for face in detected:
                    faces_data.append(face)
                    labels_data.append(label)
                    extracted += 1
                    if extracted >= STEPS_PER_VIDEO:
                        break
            frame_idx += 1
        cap.release(); video_count += 1
        try: os.remove(vfile)
        except: pass
        if video_count % 10 == 0:
            print(f"  Processed {video_count} videos, {len(faces_data)} faces")

print(f"  Total: {len(faces_data)} faces from {video_count} videos")
active_count = sum(1 for l in labels_data if l == 0)
fatigue_count = sum(1 for l in labels_data if l == 1)
print(f"  Active: {active_count} | Fatigue: {fatigue_count}")

print("\nStep 2: Fine-tuning MobileNetV2 on UTA-RLDD...")
model = tf.keras.models.load_model("drowsiness_v2.keras")
# Freeze all but last 5 layers
for layer in model.layers[:-5]:
    layer.trainable = False
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
trainable = sum(1 for l in model.layers if l.trainable)
print(f"  Trainable: {trainable}/{len(model.layers)} layers")

# Convert to numpy arrays
faces_np = np.array(faces_data, dtype=np.float32) / 127.5 - 1.0
labels_np = np.array(labels_data, dtype=np.float32)

# Shuffle and split
idx = np.random.RandomState(42).permutation(len(faces_np))
faces_np = faces_np[idx]; labels_np = labels_np[idx]
n_val = int(len(faces_np) * 0.2)
x_val, y_val = faces_np[:n_val], labels_np[:n_val]
x_train, y_train = faces_np[n_val:], labels_np[n_val:]
print(f"  Train: {len(x_train)} | Val: {n_val}")

# Class weights for imbalance
n_active = sum(1 for l in y_train if l == 0)
n_fatigue = sum(1 for l in y_train if l == 1)
class_weight = {0: len(y_train)/(2*n_active+1e-7), 1: len(y_train)/(2*n_fatigue+1e-7)}

loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss=loss_fn, metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
t0 = time.time()
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(x_val, y_val), callbacks=[early_stop], class_weight=class_weight)
print(f"  Fine-tuning done in {time.time()-t0:.0f}s")

# Evaluate
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"  Val accuracy: {val_acc:.4f}")

print("\nStep 3: Exporting to TFLite...")
model.save("drowsiness_v2_finetuned.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("drowsiness_v2_finetuned.tflite", "wb") as f: f.write(tflite_model)
print(f"  Saved drowsiness_v2_finetuned.tflite ({len(tflite_model)/1024/1024:.1f} MB)")

print("\nStep 4: Quick validation...")
from tensorflow.lite.python.interpreter import Interpreter
interp = Interpreter(model_path="drowsiness_v2_finetuned.tflite")
interp.allocate_tensors()
inp = interp.get_input_details(); out = interp.get_output_details()
preds = []
for i in range(len(x_val)):
    interp.set_tensor(inp[0]["index"], np.expand_dims(x_val[i], 0))
    interp.invoke()
    preds.append(float(interp.get_tensor(out[0]["index"])[0][0]))
preds = np.array(preds)
a = preds[y_val==0]; f_ = preds[y_val==1]
print(f"  Active sigma: {a.mean():.4f} +/- {a.std():.4f}")
print(f"  Fatigue sigma: {f_.mean():.4f} +/- {f_.std():.4f}")
best_acc=0; best_thr=0.5
for thr in np.arange(0.1, 0.9, 0.01):
    pred_labels = (preds > thr).astype(int)
    acc = np.mean(pred_labels == y_val)
    if acc > best_acc: best_acc = acc; best_thr = thr
print(f"  Best threshold: {best_thr:.2f}, Accuracy: {best_acc*100:.2f}%")
print("\nDONE - Use drowsiness_v2_finetuned.tflite with LOSO script")
