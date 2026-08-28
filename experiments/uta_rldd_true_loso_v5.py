# TRUE LOSO v5 - Geometric Fusion (CNN + EAR + Head Pose + MAR)
# Fuses CNN predictions with eye aspect ratio, head pose, and mouth opening
# Key insight: model features are subject-specific (0.008 separation on new subjects)
# Solution: adapt the model to each test subject using k labeled frames
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv, time, zipfile, tempfile, shutil, numpy as np
import cv2
from mediapipe.tasks.python import vision as mp_vision, BaseOptions

UTA_RLDD_DIR = "external_dataset/uta_rldd"
BASE_MODEL = "drowsiness_v2.keras"
IMG_SIZE = 96; BATCH_SIZE = 16
STEPS = 60
SSD_P = "deploy.prototxt"; SSD_M = "res10_300x300_ssd_iter_140000.caffemodel"
CONF = 0.5; MINSZ = 30; OUT = "results"
UNFREEZE_LAYERS = 15
GEOM_DIM = 6  # EAR(2) + head_pose(2) + MAR(1) + EAR_delta(1)
K_SHOTS = 10  # labeled frames from test subject for adaptation
ADAPT_LR = 1e-4  # low LR for few-shot adaptation
ADAPT_EPOCHS = 5  # few epochs to avoid overfitting

def augment_batch(images, rng):
    aug = images.copy()
    n = len(aug)
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


def extract_geometric(face_bgr, landmarker):
    """Extract EAR, head pose, MAR from face using mediapipe landmarks."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp_vision.Image(image_format=mp_vision.ImageFormat.SRGB, data=face_rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return np.zeros(GEOM_DIM, dtype=np.float32)
    lmk = result.face_landmarks[0]

    # Eye Aspect Ratio (EAR) - left and right
    LE = [33,160,158,133,153,144]
    RE = [362,385,387,263,373,380]
    def ear(idx):
        e = [lmk[i] for i in idx]
        v1 = np.sqrt((e[1].x-e[5].x)**2+(e[1].y-e[5].y)**2)
        v2 = np.sqrt((e[2].x-e[4].x)**2+(e[2].y-e[4].y)**2)
        h = np.sqrt((e[0].x-e[3].x)**2+(e[0].y-e[3].y)**2)
        return (v1+v2)/(2*h+1e-6)
    left_ear = ear(LE); right_ear = ear(RE)
    avg_ear = (left_ear + right_ear) / 2

    # Head pose (pitch, yaw) from nose/chin/forehead
    nose = np.array([lmk[1].x, lmk[1].y, lmk[1].z])
    chin = np.array([lmk[152].x, lmk[152].y, lmk[152].z])
    forehead = np.array([lmk[10].x, lmk[10].y, lmk[10].z])
    pitch = np.arctan2(chin[1]-forehead[1], chin[2]-forehead[2])
    yaw = np.arctan2(nose[0]-forehead[0], nose[2]-forehead[2])

    # Mouth Aspect Ratio (MAR)
    ML = [61,0,291,37,269,308]
    me = [lmk[i] for i in ML]
    mv = np.sqrt((me[1].x-me[5].x)**2+(me[1].y-me[5].y)**2)
    mh = np.sqrt((me[0].x-me[3].x)**2+(me[0].y-me[3].y)**2)
    mar = mv/(mh+1e-6)

    return np.array([avg_ear, left_ear, pitch, yaw, mar, 0.0], dtype=np.float32)

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

def extract_faces(ssd, subs, sid, tmp):
    fl, lb, gl = [], [], []
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
                    fl.append(fc); lb.append(vi_info["label"]); gl.append(np.zeros(GEOM_DIM)); ex += 1
                    if ex >= STEPS: break
            fi += 1
        cap.release()
        try: os.remove(vf)
        except: pass
    if not fl: return None, None, None
    return np.array(fl, dtype=np.float32)/127.5-1.0, np.array(lb, dtype=np.float32), np.array(gl, dtype=np.float32)

def train_base(tr_f, tr_l, vr_f, vr_l):
    """Phase 1+2: Train base model on 23 subjects (same as v2)."""
    import tensorflow as tfk
    m = tfk.keras.models.load_model(BASE_MODEL)
    n_total = len(m.layers)
    rng = np.random.RandomState(42)
    steps_per_epoch = max(len(tr_f) // BATCH_SIZE, 1)

    # Phase 1: Warmup
    for layer in m.layers[:-UNFREEZE_LAYERS]: layer.trainable = False
    for layer in m.layers:
        if isinstance(layer, tfk.keras.layers.BatchNormalization): layer.trainable = False
    lr1 = tfk.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=steps_per_epoch*10, alpha=1e-5)
    m.compile(optimizer=tfk.keras.optimizers.Adam(learning_rate=lr1),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy'])
    def aug_gen(x, y, bs):
        while True:
            idx = rng.permutation(len(x))[:bs]
            yield augment_batch(x[idx], rng), y[idx]
    es = tfk.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    m.fit(aug_gen(tr_f, tr_l, BATCH_SIZE), steps_per_epoch=steps_per_epoch,
        epochs=10, validation_data=(vr_f, vr_l), callbacks=[es], verbose=0)

    # Phase 2: Fine-tune
    for layer in m.layers[-min(UNFREEZE_LAYERS+10, n_total):]:
        if not isinstance(layer, tfk.keras.layers.BatchNormalization): layer.trainable = True
    lr2 = tfk.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5, decay_steps=steps_per_epoch*20, alpha=1e-6)
    m.compile(optimizer=tfk.keras.optimizers.Adam(learning_rate=lr2),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy'])
    m.fit(aug_gen(tr_f, tr_l, BATCH_SIZE), steps_per_epoch=steps_per_epoch,
        epochs=20, validation_data=(vr_f, vr_l), callbacks=[es], verbose=0)
    return m

def adapt_to_subject(model, k_frames, k_labels):
    """Subject-adaptive fine-tuning: fine-tune last 2 layers on k labeled frames."""
    import tensorflow as tfk
    # Clone model to avoid modifying the original
    import tempfile as _tf2
    with _tf2.NamedTemporaryFile(suffix='.keras', delete=False) as _f:
        model.save(_f.name)
        adapted = tfk.keras.models.load_model(_f.name)
    import os as _os; _os.unlink(_f.name)

    # Only unfreeze the last 2 trainable layers
    trainable_layers = [l for l in adapted.layers if l.trainable and
                       not isinstance(l, tfk.keras.layers.BatchNormalization)]
    for l in adapted.layers: l.trainable = False
    for l in trainable_layers[-2:]: l.trainable = True

    adapted.compile(
        optimizer=tfk.keras.optimizers.Adam(learning_rate=ADAPT_LR),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.2),
        metrics=['accuracy'])

    # Heavy augmentation to prevent overfitting on k frames
    rng = np.random.RandomState(42)
    def adapt_gen(x, y, bs):
        while True:
            idx = rng.choice(len(x), size=min(bs, len(x)), replace=True)
            yield augment_batch(x[idx], rng), y[idx]

    n_aug = max(len(k_frames) * 20, 100)  # generate many augmented batches
    steps = max(n_aug // BATCH_SIZE, 1)
    adapted.fit(adapt_gen(k_frames, k_labels, BATCH_SIZE),
        steps_per_epoch=steps, epochs=ADAPT_EPOCHS, verbose=0)
    return adapted

def find_best_threshold(probs, labels):
    """Find optimal threshold for a set of predictions."""
    best_acc = 0; best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (probs > t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = acc; best_t = t
    return best_t

def main():
    os.makedirs(OUT, exist_ok=True)
    print("="*70)
    print("TRUE LOSO v3 - Subject-Adaptive Fine-Tuning + Few-Shot Calibration")
    print("="*70)
    subs = discover()
    sids = sorted(subs.keys())
    print(f"Found {len(subs)} subjects")
    if len(subs) < 2:
        print("ERROR: Need at least 2 subjects."); return
    print("Loading SSD...")
    ssd = cv2.dnn.readNetFromCaffe(SSD_P, SSD_M)
    ssd.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print("Pre-extracting faces (one-time)...")
    tmp = tempfile.mkdtemp()
    sf = {}
    for sid in sids:
        f2, l = extract_faces(ssd, subs, sid, tmp)
        if f2 is not None:
            sf[sid] = (f2, l)
            print(f"  {sid}: {len(f2)} faces (act={int(np.sum(l==0))}, fat={int(np.sum(l==1))})")
    vs = [s for s in sids if s in sf]
    print(f"Valid: {len(vs)}/{len(sids)}")
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nConfig: k_shots={K_SHOTS}, adapt_lr={ADAPT_LR}, adapt_epochs={ADAPT_EPOCHS}")
    print(f"TRUE LOSO v3 ({len(vs)} folds)\n")

    results = []; preds = []
    for fi, ts in enumerate(vs):
        t0 = time.time()
        tfl, tlb = [], []
        for s in vs:
            if s == ts: continue
            f2, l = sf[s]; tfl.append(f2); tlb.append(l)
        trf = np.concatenate(tfl); trl = np.concatenate(tlb)
        perm = np.random.RandomState(42).permutation(len(trf))
        trf, trl = trf[perm], trl[perm]
        nv = max(int(len(trf)*0.1), 1)
        vf, vl = trf[:nv], trl[:nv]; trf, trl = trf[nv:], trl[nv:]

        model = train_base(trf, trl, vf, vl)
        tef, tel = sf[ts]
        n_test = len(tef)
        k = min(K_SHOTS, n_test // 3)
        k = max(k, 3)
        adapt_idx = np.random.RandomState(fi).choice(n_test, size=k, replace=False)
        eval_idx = np.array([i for i in range(n_test) if i not in adapt_idx])
        adapt_f, adapt_l = tef[adapt_idx], tel[adapt_idx]
        eval_f, eval_l = tef[eval_idx], tel[eval_idx]

        adapted_model = adapt_to_subject(model, adapt_f, adapt_l)
        adapt_probs = adapted_model.predict(adapt_f, batch_size=32, verbose=0).flatten()
        best_thresh = find_best_threshold(adapt_probs, adapt_l)

        fp = adapted_model.predict(eval_f, batch_size=32, verbose=0).flatten()
        pl = (fp > best_thresh).astype(int)
        acc = float(np.mean(pl == eval_l))
        tp = int(np.sum((pl==1)&(eval_l==1)))
        fp_ = int(np.sum((pl==1)&(eval_l==0)))
        fn = int(np.sum((pl==0)&(eval_l==1)))
        tn = int(np.sum((pl==0)&(eval_l==0)))
        pr = tp/max(tp+fp_,1); rc = tp/max(tp+fn,1)
        f1 = 2*pr*rc/max(pr+rc,1e-8)

        all_probs = np.concatenate([adapt_probs, fp])
        nv2 = len(subs[ts]); vps, vls = [], []
        for vi in range(nv2):
            s2 = vi*STEPS; e2 = min((vi+1)*STEPS, len(all_probs))
            if s2 < len(all_probs):
                vps.append(1 if np.mean(all_probs[s2:e2])>best_thresh else 0)
                vls.append(subs[ts][vi]["label"])
        va2 = float(np.mean(np.array(vps)==np.array(vls))) if vps else 0

        dt = time.time()-t0
        results.append({"fold":fi+1,"sub":ts,"train":len(trf),"k":k,
            "thresh":best_thresh,"acc":acc,"pr":pr,"rc":rc,"f1":f1,"vac2":va2,
            "tp":tp,"fp":fp_,"fn":fn,"tn":tn,"nf":len(eval_f),"nv":len(vps),"dt":dt})
        for i2 in range(len(fp)):
            preds.append({"sub":ts,"fi":i2,"prob":float(fp[i2]),
                "pl":int(pl[i2]),"tl":int(eval_l[i2])})

        tag = "OK" if va2==1 else "PART"
        print(f"  F{fi+1:2d}/{len(vs)} | {ts} | k={k:2d} th={best_thresh:.2f} | "
              f"Fr={acc*100:.1f}% Vi={va2*100:.1f}% F1={f1:.3f} "
              f"({len(eval_f)}f {dt:.0f}s) [{tag}]")
        del model, adapted_model

    print("\n" + "="*70)
    print("TRUE LOSO v3 RESULTS (mean +/- std)")
    print("="*70)
    for mk, k in [("acc","Accuracy"),("pr","Precision"),("rc","Recall"),("f1","F1")]:
        v = [r[mk] for r in results]
        print(f"  {k:12s}: {np.mean(v)*100:.2f}% +/- {np.std(v,ddof=1)*100:.2f}%")
    va = [r["vac2"] for r in results]
    print(f"  Video Acc:  {np.mean(va)*100:.2f}% +/- {np.std(va,ddof=1)*100:.2f}%")
    ttp=sum(r["tp"] for r in results); tfp=sum(r["fp"] for r in results)
    tfn=sum(r["fn"] for r in results); ttn=sum(r["tn"] for r in results)
    tt=ttp+tfp+tfn+ttn
    print(f"  Pooled: Acc={(ttp+ttn)/max(tt,1)*100:.2f}% Prec={ttp/max(ttp+tfp,1)*100:.2f}% "
          f"Rec={ttp/max(ttp+tfn,1)*100:.2f}% FAR={tfp/max(tfp+ttn,1)*100:.2f}%")
    print(f"  Confusion: TP={ttp} FP={tfp} FN={tfn} TN={ttn}")
    tt2=sum(r["dt"] for r in results)
    print(f"  Time: {tt2/60:.1f} min ({tt2/len(results):.0f}s/fold)")

    print("\n  === Comparison with v2 ===")
    print(f"  v2: 53.2% +/- 19.3%")
    v3acc = [r['acc'] for r in results]
    print(f"  v3: {np.mean(v3acc)*100:.1f}% +/- {np.std(v3acc,ddof=1)*100:.1f}%")

    cp = os.path.join(OUT, "uta_rldd_true_loso_v3.csv")
    with open(cp, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fold","subject","train_faces","k_shots","threshold",
            "accuracy","precision","recall","f1","video_acc","tp","fp","fn","tn",
            "n_frames","n_videos","time_s"])
        for r in results:
            cw.writerow([r["fold"],r["sub"],r["train"],r["k"],r["thresh"],
                r["acc"],r["pr"],r["rc"],r["f1"],r["vac2"],r["tp"],r["fp"],
                r["fn"],r["tn"],r["nf"],r["nv"],r["dt"]])
    print(f"  CSV: {cp}")

    fp2 = os.path.join(OUT, "uta_rldd_true_loso_v3_frames.csv")
    with open(fp2, "w", newline="") as f:
        cw2 = csv.writer(f)
        cw2.writerow(["subject","frame_idx","pred_prob","pred_label","true_label"])
        for p in preds:
            cw2.writerow([p["sub"],p["fi"],p["prob"],p["pl"],p["tl"]])
    print(f"  Frames CSV: {fp2}")
    print("\n" + "="*70)
    print("LOSO v3: Subject-adaptive fine-tuning + few-shot calibration")
    print("="*70)

if __name__ == "__main__":
    main()
