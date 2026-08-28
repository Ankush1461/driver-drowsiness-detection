# TRUE LOSO v6 - Combined CNN+LSTM+Geometric+Adaptive
# CNN(1280-dim) + geometric(6-dim) -> LSTM(128) -> Dense -> classifier
# Plus subject-adaptive fine-tuning + few-shot calibration
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv, time, zipfile, tempfile, shutil, numpy as np, cv2
import tensorflow as tf

UTA_RLDD_DIR = "external_dataset/uta_rldd"
BASE_MODEL = "drowsiness_v2.keras"
IMG_SIZE = 96; BATCH_SIZE = 16; STEPS = 60
SSD_P = "deploy.prototxt"; SSD_M = "res10_300x300_ssd_iter_140000.caffemodel"
CONF = 0.5; MINSZ = 30; OUT = "results"
K_SHOTS = 10; ADAPT_LR = 1e-4; ADAPT_EPOCHS = 5
SEQ_LEN = 5; LSTM_UNITS = 128; CNN_FEAT_DIM = 1280; GEOM_DIM = 6


def augment_batch(images, rng):
    aug = images.copy(); n = len(aug)
    brightness = 1.0 + rng.uniform(-0.3, 0.3, size=(n,1,1,1))
    aug = np.clip(aug * brightness, -1.0, 1.0)
    mean = aug.mean(axis=(1,2,3), keepdims=True)
    contrast = 0.7 + rng.uniform(0, 0.6, size=(n,1,1,1))
    aug = np.clip((aug - mean) * contrast + mean, -1.0, 1.0)
    flips = rng.random(n) > 0.5
    aug[flips] = aug[flips, :, ::-1, :]
    for i in range(n):
        angle = rng.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
        aug[i] = cv2.warpAffine(aug[i], M, (IMG_SIZE, IMG_SIZE),
            borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    noise = rng.normal(0, 0.02, size=aug.shape).astype(np.float32)
    aug = np.clip(aug + noise, -1.0, 1.0)
    erase_mask = rng.random(n) > 0.9
    for i in np.where(erase_mask)[0]:
        eh = rng.randint(5, 15); ew = rng.randint(5, 15)
        ey = rng.randint(0, IMG_SIZE - eh); ex = rng.randint(0, IMG_SIZE - ew)
        aug[i, ey:ey+eh, ex:ex+ew] = rng.uniform(-1.0, 1.0)
    return aug.astype(np.float32)

def clahe(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(2.0,(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def detect_faces(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))
    net.setInput(blob); d = net.forward()
    out = []
    for i in range(d.shape[2]):
        c = d[0,0,i,2]
        if c > CONF:
            bx = d[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = [int(v) for v in bx]
            x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
            fc = frame[y1:y2,x1:x2]
            if fc.size > 0 and min(fc.shape[:2]) >= MINSZ:
                sh, sw = fc.shape[:2]; side = max(sh, sw)
                t = (side - sh) // 2; b = side - sh - t
                l = (side - sw) // 2; r = side - sw - l
                fc = cv2.copyMakeBorder(fc, t, b, l, r, cv2.BORDER_REFLECT)
                fc = clahe(cv2.resize(fc,(IMG_SIZE,IMG_SIZE)))
                out.append(fc)
    return out

def discover():
    subs = {}
    for zn in sorted(os.listdir(UTA_RLDD_DIR)):
        if not zn.endswith(".zip"): continue
        with zipfile.ZipFile(os.path.join(UTA_RLDD_DIR,zn),"r") as zf:
            for n in zf.namelist():
                if not n.endswith((".mp4",".mov",".MOV")): continue
                p = n.split("/")
                if len(p) < 3: continue
                sid, vn = p[1], p[-1]; lv = vn.split(".")[0]
                subs.setdefault(sid,[]).append({"zip":zn,"path":n,
                    "level":int(lv) if lv.isdigit() else -1,
                    "label":1 if lv=="10" else 0})
    return subs

def extract_faces_with_order(ssd, subs, sid, tmp):
    fl, lb, vid_ids = [], [], []
    for vi, vi_info in enumerate(subs[sid]):
        vf = os.path.join(tmp,f"s{sid}_v{vi}.mp4")
        try:
            with zipfile.ZipFile(os.path.join(UTA_RLDD_DIR,vi_info["zip"]),"r") as z:
                with z.open(vi_info["path"]) as src, open(vf,"wb") as dst:
                    dst.write(src.read())
        except Exception: continue
        cap = cv2.VideoCapture(vf); fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: cap.release(); continue
        iv = max(1,int(fps//2))
        ex = 0; fi = 0
        while cap.isOpened() and ex < STEPS:
            ret, fr = cap.read()
            if not ret: break
            if fi % iv == 0:
                for fc in detect_faces(ssd, fr):
                    fl.append(fc); lb.append(vi_info["label"])
                    vid_ids.append(vi); ex += 1
                    if ex >= STEPS: break
            fi += 1
        cap.release()
        try: os.remove(vf)
        except: pass
    if not fl: return None, None, None
    return (np.array(fl, dtype=np.float32)/127.5-1.0,
            np.array(lb, dtype=np.float32),
            np.array(vid_ids, dtype=np.int32))

_landmarker = None
def get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import vision as mp_vision, BaseOptions
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [61,13,14,17,82,87]
NOSE_TIP = 1; CHIN = 152; L_EAR = 234; R_EAR = 454; FOREHEAD = 10

def compute_ear(landmarks, eye_idx):
    pts = [landmarks[i] for i in eye_idx]
    A = np.sqrt(sum((pts[1][j]-pts[5][j])**2 for j in range(3)))
    B = np.sqrt(sum((pts[2][j]-pts[4][j])**2 for j in range(3)))
    C = np.sqrt(sum((pts[0][j]-pts[3][j])**2 for j in range(3)))
    return (A + B) / (2.0 * C + 1e-6)

def compute_mar(landmarks):
    pts = [landmarks[i] for i in MOUTH]
    A = np.sqrt(sum((pts[1][j]-pts[5][j])**2 for j in range(3)))
    B = np.sqrt(sum((pts[2][j]-pts[4][j])**2 for j in range(3)))
    C = np.sqrt(sum((pts[0][j]-pts[3][j])**2 for j in range(3)))
    return (A + B) / (2.0 * C + 1e-6)

def compute_head_pose(landmarks):
    chin = np.array(landmarks[CHIN][:3])
    l_ear = np.array(landmarks[L_EAR][:3])
    r_ear = np.array(landmarks[R_EAR][:3])
    forehead = np.array(landmarks[FOREHEAD][:3])
    pitch = np.arctan2(chin[1]-forehead[1], chin[2]-forehead[2])
    yaw = np.arctan2(r_ear[0]-l_ear[0], r_ear[2]-l_ear[2])
    return pitch, yaw

import mediapipe as mp

def extract_geom_features(face_rgb_uint8):
    lm = get_landmarker()
    face_rgb = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    det = lm.detect(mp_img)
    if not det.face_landmarks:
        return np.zeros(GEOM_DIM, dtype=np.float32)
    fl = det.face_landmarks[0]
    landmarks = [(l.x*IMG_SIZE, l.y*IMG_SIZE, l.z) for l in fl]
    ear_l = compute_ear(landmarks, LEFT_EYE)
    ear_r = compute_ear(landmarks, RIGHT_EYE)
    mar = compute_mar(landmarks)
    pitch, yaw = compute_head_pose(landmarks)
    avg_ear = (ear_l + ear_r) / 2.0
    return np.array([avg_ear, ear_l, ear_r, pitch, yaw, mar], dtype=np.float32)

def extract_geom_batch(face_images_norm):
    feats = []
    for img in face_images_norm:
        img_uint8 = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        feats.append(extract_geom_features(img_uint8))
    return np.array(feats, dtype=np.float32)

def build_cnn_extractor(model):
    feat_model = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
    feat_model.trainable = False
    return feat_model

def extract_cnn_features(feat_model, face_images):
    feats = feat_model.predict(face_images, batch_size=32, verbose=0)
    return feats.astype(np.float32)

def make_sequences(cnn_feats, labels, geom_feats, seq_len=SEQ_LEN, stride=1):
    n = len(cnn_feats)
    if n < seq_len:
        pad = seq_len - n
        cnn_feats = np.concatenate([cnn_feats, np.zeros((pad, cnn_feats.shape[1]), dtype=cnn_feats.dtype)])
        labels = np.concatenate([labels, np.zeros(pad, dtype=labels.dtype)])
        geom_feats = np.concatenate([geom_feats, np.zeros((pad, geom_feats.shape[1]), dtype=geom_feats.dtype)])
        n = len(cnn_feats)
    seq_cnn, seq_geom, seq_labels = [], [], []
    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len
        seq_cnn.append(cnn_feats[start:end])
        seq_geom.append(geom_feats[start:end])
        seq_labels.append(1.0 if labels[start:end].mean() > 0.5 else 0.0)
    return (np.array(seq_cnn, dtype=np.float32),
            np.array(seq_geom, dtype=np.float32),
            np.array(seq_labels, dtype=np.float32))

def build_simple_lstm_model():
    cnn_input = tf.keras.Input(shape=(SEQ_LEN, CNN_FEAT_DIM), name="cnn_input")
    geom_input = tf.keras.Input(shape=(SEQ_LEN, GEOM_DIM), name="geom_input")
    combined = tf.keras.layers.Concatenate(axis=-1)([cnn_input, geom_input])
    x = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(combined)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(LSTM_UNITS // 2, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=[cnn_input, geom_input], outputs=output)

def train_lstm_model(model, tc, tg, tl, vc, vg, vl):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1),
        metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                          restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                  patience=3, min_lr=1e-6)
    model.fit([tc, tg], tl, validation_data=([vc, vg], vl),
        batch_size=BATCH_SIZE, epochs=30, callbacks=[es, rlrop], verbose=0)
    return model

def adapt_to_subject(model, cnn_feats_k, geom_k, labels_k):
    """Adapt model to new subject using k-shot frames grouped into SEQ_LEN sequences."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        model.save(f.name)
        adapted = tf.keras.models.load_model(f.name)
    os.unlink(f.name)
    for l in adapted.layers: l.trainable = False
    for l in adapted.layers:
        if any(tn in l.name.lower() for tn in ["lstm", "dense", "dropout"]):
            l.trainable = True
    adapted.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ADAPT_LR),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.2),
        metrics=["accuracy"])
    # Group k-shot frames into sequences of SEQ_LEN for the LSTM
    k = cnn_feats_k.shape[1]  # number of k-shot frames
    if k < SEQ_LEN:
        # Pad to at least SEQ_LEN
        pad = SEQ_LEN - k
        cnn_feats_k = np.concatenate([cnn_feats_k, np.zeros((1, pad, cnn_feats_k.shape[2]), dtype=cnn_feats_k.dtype)], axis=1)
        geom_k = np.concatenate([geom_k, np.zeros((1, pad, geom_k.shape[2]), dtype=geom_k.dtype)], axis=1)
        labels_k = np.concatenate([labels_k, np.zeros((1, pad), dtype=labels_k.dtype)], axis=1)
        k = SEQ_LEN
    # Create overlapping sequences of length SEQ_LEN from the k-shot frames
    flat_cnn = cnn_feats_k[0]  # (k, 1280)
    flat_geom = geom_k[0]      # (k, 6)
    flat_labels = labels_k[0]  # (k,)
    seq_cnn_list, seq_geom_list, seq_label_list = [], [], []
    for start in range(0, k - SEQ_LEN + 1):
        end = start + SEQ_LEN
        seq_cnn_list.append(flat_cnn[start:end])
        seq_geom_list.append(flat_geom[start:end])
        seq_label_list.append(1.0 if flat_labels[start:end].mean() > 0.5 else 0.0)
    if not seq_cnn_list:
        # Fallback: use all frames padded to SEQ_LEN
        seq_cnn_list.append(flat_cnn[:SEQ_LEN])
        seq_geom_list.append(flat_geom[:SEQ_LEN])
        seq_label_list.append(1.0 if flat_labels[:SEQ_LEN].mean() > 0.5 else 0.0)
    aug_cnn = np.array(seq_cnn_list, dtype=np.float32)  # (n_seq, SEQ_LEN, 1280)
    aug_geom = np.array(seq_geom_list, dtype=np.float32) # (n_seq, SEQ_LEN, 6)
    aug_labels = np.array(seq_label_list, dtype=np.float32) # (n_seq,)
    # Augment by repeating with noise
    rng = np.random.RandomState(42)
    repeats = max(20, 100 // max(len(aug_labels), 1))
    aug_cnn = np.tile(aug_cnn, (repeats, 1, 1))
    aug_geom = np.tile(aug_geom, (repeats, 1, 1))
    aug_labels = np.tile(aug_labels, repeats)
    noise = rng.normal(0, 0.05, size=aug_cnn.shape).astype(np.float32)
    aug_cnn = np.clip(aug_cnn + noise, -1.0, 1.0)
    n_total = len(aug_labels)
    nv = max(int(n_total * 0.1), 1)
    nv = min(nv, n_total - 4)  # ensure val split has at least 4
    adapted.fit([aug_cnn[:nv], aug_geom[:nv]], aug_labels[:nv],
        validation_data=([aug_cnn[nv:nv+4], aug_geom[nv:nv+4]], aug_labels[nv:nv+4]),
        batch_size=min(BATCH_SIZE, n_total), epochs=ADAPT_EPOCHS, verbose=0)
    return adapted

def find_best_threshold(probs, labels):
    best_acc = 0; best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (probs > t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc: best_acc = acc; best_t = t
    return best_t
def main():
    os.makedirs(OUT, exist_ok=True)
    sep = "=" * 70
    print(sep); print("TRUE LOSO v6 - Combined CNN+LSTM+Geometric+Adaptive"); print(sep)
    subs = discover(); sids = sorted(subs.keys())
    print("Found {} subjects".format(len(subs)))
    if len(subs) < 2: print("ERROR"); return
    print("Loading SSD...")
    ssd = cv2.dnn.readNetFromCaffe(SSD_P, SSD_M)
    ssd.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print("Loading CNN model...")
    base_model = tf.keras.models.load_model(BASE_MODEL)
    cnn_extractor = build_cnn_extractor(base_model)
    print("CNN feature dim: {}".format(CNN_FEAT_DIM))
    print("Pre-extracting faces + geometric features (one-time)...")
    tmp = tempfile.mkdtemp(); sf = {}
    for sid in sids:
        frames, labels, vid_ids = extract_faces_with_order(ssd, subs, sid, tmp)
        if frames is not None:
            cnn_feats = extract_cnn_features(cnn_extractor, frames)
            geom_feats = extract_geom_batch(frames)
            sf[sid] = {"frames": frames, "labels": labels, "vid_ids": vid_ids,
                        "cnn_feats": cnn_feats, "geom_feats": geom_feats}
            print("  {}: {} faces".format(sid, len(frames)))
    vs = [s for s in sids if s in sf]
    print("Valid: {}/{}".format(len(vs), len(sids)))
    shutil.rmtree(tmp, ignore_errors=True)
    print("Config: seq_len={}, lstm={}, k_shots={}".format(SEQ_LEN, LSTM_UNITS, K_SHOTS))
    print("TRUE LOSO v6 ({} folds)".format(len(vs)))
    results = []; preds = []
    for fi, ts in enumerate(vs):
        t0 = time.time()
        train_cnn, train_geom, train_labels = [], [], []
        for s in vs:
            if s == ts: continue
            train_cnn.append(sf[s]["cnn_feats"])
            train_geom.append(sf[s]["geom_feats"])
            train_labels.append(sf[s]["labels"])
        all_cnn = np.concatenate(train_cnn)
        all_geom = np.concatenate(train_geom)
        all_labels = np.concatenate(train_labels)
        perm = np.random.RandomState(42).permutation(len(all_cnn))
        all_cnn = all_cnn[perm]; all_geom = all_geom[perm]; all_labels = all_labels[perm]
        seq_cnn, seq_geom, seq_labels = make_sequences(all_cnn, all_labels, all_geom)
        nv = max(int(len(seq_labels) * 0.1), 1)
        vc, vg, vl = seq_cnn[:nv], seq_geom[:nv], seq_labels[:nv]
        tc, tg, tl = seq_cnn[nv:], seq_geom[nv:], seq_labels[nv:]
        model = build_simple_lstm_model()
        model = train_lstm_model(model, tc, tg, tl, vc, vg, vl)
        test_cnn = sf[ts]["cnn_feats"]; test_geom = sf[ts]["geom_feats"]
        test_labels = sf[ts]["labels"]
        n_test = len(test_cnn)
        k = min(K_SHOTS, n_test // 3); k = max(k, 3)
        rng = np.random.RandomState(fi)
        adapt_idx = rng.choice(n_test, size=k, replace=False)
        adapt_model = adapt_to_subject(model,
            test_cnn[adapt_idx].reshape(1, k, CNN_FEAT_DIM),
            test_geom[adapt_idx].reshape(1, k, GEOM_DIM),
            test_labels[adapt_idx].reshape(1, k))
        all_probs = np.zeros(n_test); all_counts = np.zeros(n_test)
        for start in range(0, n_test - SEQ_LEN + 1):
            end = start + SEQ_LEN
            c_batch = test_cnn[start:end].reshape(1, SEQ_LEN, -1)
            g_batch = test_geom[start:end].reshape(1, SEQ_LEN, -1)
            prob = adapt_model.predict([c_batch, g_batch], verbose=0).flatten()[0]
            all_probs[start:end] += prob; all_counts[start:end] += 1
        all_counts[all_counts == 0] = 1; all_probs /= all_counts
        adapt_probs = all_probs[adapt_idx]
        best_thresh = find_best_threshold(adapt_probs, test_labels[adapt_idx])
        eval_idx = np.array([i for i in range(n_test) if i not in adapt_idx])
        eval_probs = all_probs[eval_idx]; eval_labels_arr = test_labels[eval_idx]
        pl = (eval_probs > best_thresh).astype(int)
        acc = float(np.mean(pl == eval_labels_arr))
        tp = int(np.sum((pl==1)&(eval_labels_arr==1)))
        fp_ = int(np.sum((pl==1)&(eval_labels_arr==0)))
        fn = int(np.sum((pl==0)&(eval_labels_arr==1)))
        tn = int(np.sum((pl==0)&(eval_labels_arr==0)))
        pr = tp/max(tp+fp_,1); rc = tp/max(tp+fn,1)
        f1 = 2*pr*rc/max(pr+rc,1e-8)
        nv2 = len(subs[ts]); vps, vls = [], []
        for vi in range(nv2):
            s2 = vi * STEPS; e2 = min((vi + 1) * STEPS, n_test)
            if s2 < n_test:
                vps.append(1 if np.mean(all_probs[s2:e2]) > best_thresh else 0)
                vls.append(subs[ts][vi]["label"])
        va2 = float(np.mean(np.array(vps) == np.array(vls))) if vps else 0
        dt = time.time() - t0
        results.append({"fold": fi+1, "sub": ts, "train": len(tc), "k": k,
            "thresh": best_thresh, "acc": acc, "pr": pr, "rc": rc, "f1": f1,
            "vac2": va2, "tp": tp, "fp": fp_, "fn": fn, "tn": tn,
            "nf": len(eval_idx), "nv": len(vps), "dt": dt})
        for i2 in range(len(eval_idx)):
            preds.append({"sub": ts, "fi": int(eval_idx[i2]),
                "prob": float(all_probs[eval_idx[i2]]),
                "pl": int(pl[i2]), "tl": int(eval_labels_arr[i2])})
        tag = "OK" if va2==1 else "PART"
        print("  F{:2d}/{} | {} | k={:2d} th={:.2f} | Fr={:.1f}% Vi={:.1f}% F1={:.3f} ({}f {:.0f}s) [{}]".format(
            fi+1, len(vs), ts, k, best_thresh, acc*100, va2*100, f1, len(eval_idx), dt, tag))
        del model, adapt_model
    print(""); print(sep); print("TRUE LOSO v6 RESULTS (mean +/- std)"); print(sep)
    for mk, kn in [("acc","Accuracy"),("pr","Precision"),("rc","Recall"),("f1","F1")]:
        v = [r[mk] for r in results]
        print("  {:12s}: {:.2f}% +/- {:.2f}%".format(kn, np.mean(v)*100, np.std(v,ddof=1)*100))
    va = [r["vac2"] for r in results]
    print("  Video Acc:  {:.2f}% +/- {:.2f}%".format(np.mean(va)*100, np.std(va,ddof=1)*100))
    ttp=sum(r["tp"] for r in results); tfp=sum(r["fp"] for r in results)
    tfn=sum(r["fn"] for r in results); ttn=sum(r["tn"] for r in results)
    tt=ttp+tfp+tfn+ttn
    print("  Pooled: Acc={:.2f}% Prec={:.2f}% Rec={:.2f}% FAR={:.2f}%".format(
        (ttp+ttn)/max(tt,1)*100, ttp/max(ttp+tfp,1)*100,
        ttp/max(ttp+tfn,1)*100, tfp/max(tfp+ttn,1)*100))
    print("  Confusion: TP={} FP={} FN={} TN={}".format(ttp, tfp, tfn, ttn))
    tt2=sum(r["dt"] for r in results)
    print("  Time: {:.1f} min ({:.0f}s/fold)".format(tt2/60, tt2/len(results)))
    print(""); print("  === Comparison ===")
    print("  v2: 53.2% +/- 19.3% (baseline)")
    v6acc = [r["acc"] for r in results]
    print("  v6: {:.1f}% +/- {:.1f}%".format(np.mean(v6acc)*100, np.std(v6acc,ddof=1)*100))
    cp = os.path.join(OUT, "uta_rldd_true_loso_v6.csv")
    with open(cp, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fold","subject","train_seqs","k_shots","threshold",
            "accuracy","precision","recall","f1","video_acc","tp","fp","fn","tn",
            "n_frames","n_videos","time_s"])
        for r in results:
            cw.writerow([r["fold"],r["sub"],r["train"],r["k"],r["thresh"],
                r["acc"],r["pr"],r["rc"],r["f1"],r["vac2"],r["tp"],r["fp"],
                r["fn"],r["tn"],r["nf"],r["nv"],r["dt"]])
    print("  CSV: {}".format(cp))
    fp2 = os.path.join(OUT, "uta_rldd_true_loso_v6_frames.csv")
    with open(fp2, "w", newline="") as f:
        cw2 = csv.writer(f)
        cw2.writerow(["subject","frame_idx","pred_prob","pred_label","true_label"])
        for p in preds:
            cw2.writerow([p["sub"],p["fi"],p["prob"],p["pl"],p["tl"]])
    print("  Frames CSV: {}".format(fp2))
    print(""); print(sep)
    print("LOSO v6: CNN + Geometric + LSTM + Subject-Adaptive"); print(sep)

if __name__ == "__main__":
    main()
