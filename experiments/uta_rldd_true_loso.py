import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv, time, zipfile, tempfile, shutil, numpy as np
import cv2
UTA_RLDD_DIR = "external_dataset/uta_rldd"
BASE_MODEL = "drowsiness_v2.keras"
IMG_SIZE = 96; BATCH_SIZE = 16; LR = 5e-5; MAX_EP = 20; PAT = 5
STEPS = 30; SSD_P = "deploy.prototxt"; SSD_M = "res10_300x300_ssd_iter_140000.caffemodel"
CONF = 0.5; MINSZ = 30; OUT = "results"
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
        iv = max(1,int(fps)); ex = 0; fi = 0
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
    for layer in m.layers[:-5]: layer.trainable = False
    for layer in m.layers:
        if isinstance(layer, tfk.keras.layers.BatchNormalization): layer.trainable = False
    n_a = max(sum(1 for x in tr_l if x==0), 1)
    n_f = max(sum(1 for x in tr_l if x==1), 1)
    cw = {0: len(tr_l)/(2*n_a+1e-7), 1: len(tr_l)/(2*n_f+1e-7)}
    m.compile(optimizer=tfk.keras.optimizers.Adam(LR),
        loss=tfk.keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.05), metrics=["accuracy"])
    es = tfk.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PAT, restore_best_weights=True)
    m.fit(tr_f, tr_l, epochs=MAX_EP, batch_size=BATCH_SIZE,
        validation_data=(vr_f, vr_l), callbacks=[es], class_weight=cw, verbose=0)
    _, va = m.evaluate(vr_f, vr_l, verbose=0)
    return va, m
def main():
    os.makedirs(OUT, exist_ok=True)
    print("="*70)
    print("TRUE Leave-One-Subject-Out (LOSO) Evaluation")
    print("Each fold trains from scratch, test subject COMPLETELY excluded")
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
        f, l = extract_faces(ssd, subs, sid, tmp)
        if f is not None:
            sf[sid] = (f, l)
            print(f"  {sid}: {len(f)} faces (act={int(np.sum(l==0))}, fat={int(np.sum(l==1))})")
    vs = [s for s in sids if s in sf]
    print(f"Valid: {len(vs)}/{len(sids)}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("TRUE LOSO (" + str(len(vs)) + " folds)")
    results = []; preds = []
    for fi, ts in enumerate(vs):
        t0 = time.time()
        tfl, tlb = [], []
        for s in vs:
            if s == ts: continue
            f, l = sf[s]; tfl.append(f); tlb.append(l)
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
    print("TRUE LOSO RESULTS (mean +/- std)")
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
    cp = os.path.join(OUT, "uta_rldd_true_loso.csv")
    with open(cp, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fold","subject","train_faces","val_acc","accuracy","precision","recall","f1","video_acc","tp","fp","fn","tn","n_frames","n_videos","time_s"])
        for r in results:
            cw.writerow([r["fold"],r["sub"],r["train"],r["vac"],r["acc"],r["pr"],r["rc"],r["f1"],r["vac2"],r["tp"],r["fp"],r["fn"],r["tn"],r["nf"],r["nv"],r["dt"]])
    print(f"  CSV: {cp}")
    fp2 = os.path.join(OUT, "uta_rldd_true_loso_frames.csv")
    with open(fp2, "w", newline="") as f:
        cw2 = csv.writer(f)
        cw2.writerow(["subject","frame_idx","pred_prob","pred_label","true_label"])
        for p in preds:
            cw2.writerow([p["sub"],p["fi"],p["prob"],p["pl"],p["tl"]])
    print(f"  Frames CSV: {fp2}")
    print("TRUE LOSO: No data leakage. Each fold trained from scratch.")
    print("="*70)
if __name__ == "__main__":
    main()
