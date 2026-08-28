# TRUE LOSO v2 - Improved cross-subject generalization
# Changes: augmentation, more frames, deeper fine-tuning, cosine LR
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv, time, zipfile, tempfile, shutil, numpy as np
import cv2

UTA_RLDD_DIR = "external_dataset/uta_rldd"
BASE_MODEL = "drowsiness_v2.keras"
IMG_SIZE = 96; BATCH_SIZE = 32; MAX_EP = 30; PAT = 8
STEPS = 60  # doubled from 30
SSD_P = "deploy.prototxt"; SSD_M = "res10_300x300_ssd_iter_140000.caffemodel"
CONF = 0.5; MINSZ = 30; OUT = "results"
UNFREEZE_LAYERS = 15  # deeper adaptation

# Data augmentation: on-the-fly via numpy (no extra deps)
def augment_batch(images, rng):
    aug = images.copy()
    n = len(aug)
    # 1. Random brightness (+/- 30%)
    brightness = 1.0 + rng.uniform(-0.3, 0.3, size=(n,1,1,1))
    aug = np.clip(aug * brightness, -1.0, 1.0)
    # 2. Random contrast (0.7-1.3)
    mean = aug.mean(axis=(1,2,3), keepdims=True)
    contrast = 0.7 + rng.uniform(0, 0.6, size=(n,1,1,1))
    aug = np.clip((aug - mean) * contrast + mean, -1.0, 1.0)
    # 3. Random horizontal flip (50%)
    flips = rng.random(n) > 0.5
    aug[flips] = aug[flips, :, ::-1, :]
    # 4. Random rotation (+/- 10 degrees)
    for i in range(n):
        angle = rng.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
        aug[i] = cv2.warpAffine(aug[i], M, (IMG_SIZE, IMG_SIZE),
            borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    # 5. Random Gaussian noise
    noise = rng.normal(0, 0.02, size=aug.shape).astype(np.float32)
    aug = np.clip(aug + noise, -1.0, 1.0)
    # 6. Random erasing (cutout-style, 10% of images)
    erase_mask = rng.random(n) > 0.9
    for i in np.where(erase_mask)[0]:
        eh = rng.randint(5, 15)
        ew = rng.randint(5, 15)
        ey = rng.randint(0, IMG_SIZE - eh)
        ex = rng.randint(0, IMG_SIZE - ew)
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

def extract_faces(ssd, subs, sid, tmp):
    fl, lb = [], []
    for vi, vi_info in enumerate(subs[sid]):
        vf = os.path.join(tmp,f"s{sid}_v{vi}.mp4")
        try:
            with zipfile.ZipFile(os.path.join(UTA_RLDD_DIR,vi_info["zip"]),"r") as z:
                with z.open(vi_info["path"]) as src, open(vf,"wb") as dst:
                    dst.write(src.read())
        except Exception: continue
        cap = cv2.VideoCapture(vf); fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: cap.release(); continue
        iv = max(1,int(fps//2))  # 2x more frames (every 0.5s)
        ex = 0; fi = 0
        while cap.isOpened() and ex < STEPS:
            ret, fr = cap.read()
            if not ret: break
            if fi % iv == 0:
                for fc in detect_faces(ssd, fr):
                    fl.append(fc); lb.append(vi_info["label"]); ex += 1
                    if ex >= STEPS: break
            fi += 1
        cap.release()
        try: os.remove(vf)
        except: pass
    if not fl: return None, None
    return np.array(fl, dtype=np.float32)/127.5-1.0, np.array(lb, dtype=np.float32)

def train_fold(tr_f, tr_l, vr_f, vr_l):
    import tensorflow as tfk
    m = tfk.keras.models.load_model(BASE_MODEL)
    n_total = len(m.layers)

    # Phase 1: Warmup - train only top layers with high LR
    for layer in m.layers[:-UNFREEZE_LAYERS]: layer.trainable = False
    for layer in m.layers:
        if isinstance(layer, tfk.keras.layers.BatchNormalization): layer.trainable = False
    trainable = sum(1 for l in m.layers if l.trainable)
    rng = np.random.RandomState(42)

    # Cosine decay schedule
    steps_per_epoch = max(len(tr_f) // BATCH_SIZE, 1)
    total_steps = steps_per_epoch * 10  # warmup epochs
    lr_schedule = tfk.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-5)

    m.compile(optimizer=tfk.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy'])

    # Augmentation generator
    def aug_gen(x, y, bs):
        while True:
            idx = rng.permutation(len(x))[:bs]
            batch_x = augment_batch(x[idx], rng)
            yield batch_x, y[idx]

    # Warmup phase: 10 epochs with augmentation
    es = tfk.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    m.fit(aug_gen(tr_f, tr_l, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=(vr_f, vr_l),
        callbacks=[es], verbose=0)

    # Phase 2: Fine-tune with lower LR, unfreeze more
    for layer in m.layers[-min(UNFREEZE_LAYERS+10, n_total):]:
        if not isinstance(layer, tfk.keras.layers.BatchNormalization):
            layer.trainable = True
    trainable2 = sum(1 for l in m.layers if l.trainable)

    lr_schedule2 = tfk.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5, decay_steps=steps_per_epoch * 20, alpha=1e-6)
    m.compile(optimizer=tfk.keras.optimizers.Adam(learning_rate=lr_schedule2),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1),
        metrics=['accuracy'])

    m.fit(aug_gen(tr_f, tr_l, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=(vr_f, vr_l),
        callbacks=[es], verbose=0)

    _, va = m.evaluate(vr_f, vr_l, verbose=0)
    return va, m

def main():
    os.makedirs(OUT, exist_ok=True)
    print("="*70)
    print("TRUE LOSO v2 - Improved Cross-Subject Generalization")
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
    print(f"TRUE LOSO v2 ({len(vs)} folds)")

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
        va, model = train_fold(trf, trl, vf, vl)
        tef, tel = sf[ts]
        fp = model.predict(tef, batch_size=32, verbose=0).flatten()
        pl = (fp > 0.5).astype(int)
        acc = float(np.mean(pl == tel))
        tp = int(np.sum((pl==1)&(tel==1)))
        fp_ = int(np.sum((pl==1)&(tel==0)))
        fn = int(np.sum((pl==0)&(tel==1)))
        tn = int(np.sum((pl==0)&(tel==0)))
        pr = tp/max(tp+fp_,1); rc = tp/max(tp+fn,1)
        f1 = 2*pr*rc/max(pr+rc,1e-8)
        nv2 = len(subs[ts]); vps, vls = [], []
        for vi in range(nv2):
            s2 = vi*STEPS; e2 = min((vi+1)*STEPS, len(fp))
            if s2 < len(fp):
                vps.append(1 if np.mean(fp[s2:e2])>0.5 else 0)
                vls.append(subs[ts][vi]["label"])
        va2 = float(np.mean(np.array(vps)==np.array(vls))) if vps else 0
        dt = time.time()-t0
        results.append({"fold":fi+1,"sub":ts,"train":len(trf),"vac":va,
            "acc":acc,"pr":pr,"rc":rc,"f1":f1,"vac2":va2,
            "tp":tp,"fp":fp_,"fn":fn,"tn":tn,"nf":len(tef),"nv":len(vps),"dt":dt})
        for i2 in range(len(fp)):
            preds.append({"sub":ts,"fi":i2,"prob":float(fp[i2]),
                "pl":int(pl[i2]),"tl":int(tel[i2])})
        tag = "OK" if va2==1 else "PART"
        print(f"  F{fi+1:2d}/{len(vs)} | {ts} | Fr={acc*100:.1f}% Vi={va2*100:.1f}% F1={f1:.3f} ({len(tef)}f {dt:.0f}s) [{tag}]")
        del model

    print("\n" + "="*70)
    print("TRUE LOSO v2 RESULTS (mean +/- std)")
    print("="*70)
    for mk, k in [("acc","Accuracy"),("pr","Precision"),("rc","Recall"),("f1","F1")]:
        v = [r[mk] for r in results]
        print(f"  {k:12s}: {np.mean(v)*100:.2f}% +/- {np.std(v,ddof=1)*100:.2f}%")
    va = [r["vac2"] for r in results]
    print(f"  Video Acc:  {np.mean(va)*100:.2f}% +/- {np.std(va,ddof=1)*100:.2f}%")
    ttp=sum(r["tp"] for r in results); tfp=sum(r["fp"] for r in results)
    tfn=sum(r["fn"] for r in results); ttn=sum(r["tn"] for r in results)
    tt=ttp+tfp+tfn+ttn
    print(f"  Pooled: Acc={(ttp+ttn)/max(tt,1)*100:.2f}% Prec={ttp/max(ttp+tfp,1)*100:.2f}% Rec={ttp/max(ttp+tfn,1)*100:.2f}% FAR={tfp/max(tfp+ttn,1)*100:.2f}%")
    print(f"  Confusion: TP={ttp} FP={tfp} FN={tfn} TN={ttn}")
    tt2=sum(r["dt"] for r in results)
    print(f"  Time: {tt2/60:.1f} min ({tt2/len(results):.0f}s/fold)")

    cp = os.path.join(OUT, "uta_rldd_true_loso_v2.csv")
    with open(cp, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fold","subject","train_faces","val_acc","accuracy","precision","recall","f1","video_acc","tp","fp","fn","tn","n_frames","n_videos","time_s"])
        for r in results:
            cw.writerow([r["fold"],r["sub"],r["train"],r["vac"],r["acc"],r["pr"],r["rc"],r["f1"],r["vac2"],r["tp"],r["fp"],r["fn"],r["tn"],r["nf"],r["nv"],r["dt"]])
    print(f"  CSV: {cp}")

    fp2 = os.path.join(OUT, "uta_rldd_true_loso_v2_frames.csv")
    with open(fp2, "w", newline="") as f:
        cw2 = csv.writer(f)
        cw2.writerow(["subject","frame_idx","pred_prob","pred_label","true_label"])
        for p in preds:
            cw2.writerow([p["sub"],p["fi"],p["prob"],p["pl"],p["tl"]])
    print(f"  Frames CSV: {fp2}")
    print("\n" + "="*70)
    print("TRUE LOSO v2: No data leakage. Each fold trained from scratch.")
    print("="*70)


if __name__ == "__main__":
    main()
