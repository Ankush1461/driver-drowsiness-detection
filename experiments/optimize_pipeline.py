"""
DriveSafe AI Pipeline Optimization
====================================
Goal: Minimize false alarms while maintaining 100% drowsy recall.

Tests:
  1. All 6 TFLite models on UTA-RLDD (24 subjects, 72 videos)
  2. Multiple preprocessing pipelines (raw, CLAHE, normalization variants)
  3. Confidence threshold optimization (sweep 0.1 to 0.99)
  4. Face quality filtering (size, blur, brightness)
  5. Per-frame sigmoid distribution analysis
  6. Best model + config selection

Usage:
    python experiments/optimize_pipeline.py
"""

import csv
import os
import sys
import time
import zipfile
import tempfile
import shutil
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python required")
    sys.exit(1)

# === PATHS ===
UTA_RLDD_DIR = "external_dataset/uta_rldd"
SSD_PROTO = "deploy.prototxt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
IMG_SIZE = 96
FRAME_INTERVAL = 30  # 1 FPS sampling
FACE_CONF_THRESH = 0.7
FACE_MIN_SIZE = 30
MAX_FRAMES_PER_VIDEO = 30
OUTPUT_DIR = "results"

# All available TFLite models
MODELS = {
    "drowsiness": "drowsiness.tflite",
    "drowsiness_robust": "drowsiness_robust.tflite",
    "pipeline": "drowsiness_pipeline.tflite",
    "pipeline_v2": "drowsiness_pipeline_v2.tflite",
    "pipeline_matched": "drowsiness_pipeline_matched.tflite",
    "akahana_matched": "drowsiness_akahana_matched.tflite",
}

# === FACE DETECTION ===
def load_ssd():
    return cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)

def detect_faces(net, frame, conf_thresh=FACE_CONF_THRESH, min_size=FACE_MIN_SIZE):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()
    faces = []
    boxes = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > conf_thresh:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                fh, fw = face.shape[:2]
                if fh < min_size or fw < min_size:
                    continue
                # Pad to square
                side = max(fh, fw)
                ph = (side - fh) // 2
                pw = (side - fw) // 2
                face = cv2.copyMakeBorder(
                    face, ph, side-fh-ph, pw, side-fw-pw,
                    cv2.BORDER_REFLECT
                )
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                faces.append(face)
                boxes.append((x1, y1, x2, y2))
    return faces, boxes

# === FACE QUALITY ===
def face_quality_score(face):
    """Score 0-1, higher = better quality for classification."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    
    # 1. Brightness (not too dark, not too bright)
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2  # peak at 0.5
    
    # 2. Contrast (not too flat)
    contrast = np.std(gray) / 128.0
    contrast_score = min(contrast, 1.0)
    
    # 3. Sharpness (Laplacian variance)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = min(float(lap.var()) / 500.0, 1.0)
    
    # 4. Face fill ratio (face should fill most of the crop)
    # Check center region energy
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    fill_score = np.mean(center > 20) * 0.5 + np.mean(center < 235) * 0.5
    
    # Weighted combination
    score = 0.2 * brightness_score + 0.2 * contrast_score + 0.4 * sharpness + 0.2 * fill_score
    return score

# === PREPROCESSING VARIANTS ===
def preprocess_raw(face):
    """Standard preprocessing: /127.5 - 1.0"""
    return (face.astype(np.float32) / 127.5) - 1.0

def preprocess_clahe(face):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (enhanced.astype(np.float32) / 127.5) - 1.0

def preprocess_gamma(face):
    """Gamma correction to normalize brightness"""
    # Estimate gamma from mean brightness
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray) / 255.0
    gamma = max(0.3, min(3.0, np.log(0.5) / np.log(max(mean_brightness, 0.01))))
    gamma = min(gamma, 2.0)  # cap
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(face, table)
    return (corrected.astype(np.float32) / 127.5) - 1.0

def preprocess_gray(face):
    """Convert to grayscale (single channel, replicated to 3)"""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return (gray3.astype(np.float32) / 127.5) - 1.0

PREPROCESSING = {
    "raw": preprocess_raw,
    "clahe": preprocess_clahe,
    "gamma": preprocess_gamma,
    "gray": preprocess_gray,
}

# === TFLite LOADER ===
def load_tflite(model_path):
    try:
        import ai_edge_litert.interpreter as interp_mod
    except ImportError:
        import tflite_runtime.interpreter as interp_mod
    interpreter = interp_mod.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter

def classify_single(interpreter, face_img, preprocess_fn):
    """Classify a single face, return raw sigmoid output."""
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    
    processed = preprocess_fn(face_img)
    single = np.array(processed, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)
    interpreter.set_tensor(inp_idx, single)
    interpreter.invoke()
    pred = interpreter.get_tensor(out_idx)
    return float(pred.flatten()[0])

# === VIDEO EVALUATION ===
def eval_video_raw(ssd_net, interpreter, video_path, preprocess_fn, 
                   quality_thresh=0.0, threshold=0.5):
    """Evaluate a video, return per-frame sigmoid values."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frame_probs = []
    frame_qualities = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_INTERVAL == 0:
            faces, _ = detect_faces(ssd_net, frame)
            for face in faces:
                q = face_quality_score(face)
                if q < quality_thresh:
                    continue
                prob = classify_single(interpreter, face, preprocess_fn)
                frame_probs.append(prob)
                frame_qualities.append(q)
            if len(frame_probs) >= MAX_FRAMES_PER_VIDEO:
                break
        idx += 1
    cap.release()
    
    if not frame_probs:
        return None
    
    probs = np.array(frame_probs)
    qualities = np.array(frame_qualities)
    
    # Video-level prediction with threshold
    mean_prob = float(np.mean(probs))
    pred = 1 if mean_prob > threshold else 0
    
    return {
        "n_frames": len(probs),
        "mean_prob": mean_prob,
        "max_prob": float(np.max(probs)),
        "min_prob": float(np.min(probs)),
        "std_prob": float(np.std(probs)),
        "mean_quality": float(np.mean(qualities)),
        "pred": pred,
    }

# === DISCOVER SUBJECTS ===
def discover_subjects():
    subjects = {}
    for zip_name in sorted(os.listdir(UTA_RLDD_DIR)):
        if not zip_name.endswith('.zip'):
            continue
        zip_path = os.path.join(UTA_RLDD_DIR, zip_name)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if not name.endswith(('.mp4', '.mov', '.MOV', '.avi')):
                    continue
                parts = name.split('/')
                if len(parts) >= 3:
                    subject_id = parts[1]
                    video_name = parts[-1]
                    level = video_name.split('.')[0]
                    if subject_id not in subjects:
                        subjects[subject_id] = []
                    subjects[subject_id].append({
                        "zip": zip_name,
                        "path_in_zip": name,
                        "level": int(level) if level.isdigit() else -1,
                        "label": 1 if level == "10" else 0,
                    })
    return subjects

# === MAIN ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("DriveSafe AI Pipeline Optimization")
    print("=" * 70)
    
    # Discover subjects
    print("\nDiscovering subjects...")
    subjects = discover_subjects()
    subject_ids = sorted(subjects.keys())[:24]
    print(f"Using {len(subject_ids)} subjects")
    
    # Load SSD
    print("Loading SSD face detector...")
    ssd_net = load_ssd()
    
    # === PHASE 1: Raw sigmoid distribution analysis ===
    print("\n" + "=" * 70)
    print("PHASE 1: Sigmoid Distribution Analysis (drowsiness_robust.tflite)")
    print("=" * 70)
    
    interpreter = load_tflite(MODELS["drowsiness_robust"])
    
    all_probs_by_class = {0: [], 1: []}  # label -> list of probs
    
    for subject in subject_ids[:6]:  # Quick sample with 6 subjects
        for video_info in subjects[subject]:
            tmp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(UTA_RLDD_DIR, video_info["zip"])
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extract(video_info["path_in_zip"], tmp_dir)
                video_path = os.path.join(tmp_dir, video_info["path_in_zip"])
                if os.path.exists(video_path):
                    result = eval_video_raw(ssd_net, interpreter, video_path, preprocess_raw)
                    if result:
                        label = video_info["label"]
                        all_probs_by_class[label].append(result["mean_prob"])
                        level_str = {0: "alert", 1: "drowsy"}.get(video_info["level"], "?")
                        print(f"  Subject {subject} Level={video_info['level']:2d} ({level_str:12s}) "
                              f"mean_prob={result['mean_prob']:.4f} std={result['std_prob']:.4f} "
                              f"quality={result['mean_quality']:.3f} frames={result['n_frames']}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Print distribution
    for label, name in [(0, "ALERT"), (1, "DROWSY")]:
        probs = all_probs_by_class[label]
        if probs:
            print(f"\n  {name} distribution: mean={np.mean(probs):.4f} "
                  f"std={np.std(probs):.4f} min={np.min(probs):.4f} max={np.max(probs):.4f} "
                  f"median={np.median(probs):.4f}")
    
    # === PHASE 2: Model comparison (all 6 models) ===
    print("\n" + "=" * 70)
    print("PHASE 2: Model Comparison (all 6 TFLite models, threshold=0.5)")
    print("=" * 70)
    
    model_results = {}
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name}: file not found")
            continue
        
        print(f"\n  Testing {model_name} ({os.path.getsize(model_path)/1024/1024:.1f} MB)...")
        interp = load_tflite(model_path)
        
        tp = fp = fn = tn = 0
        all_probs = []
        
        for subject in subject_ids:
            for video_info in subjects[subject]:
                tmp_dir = tempfile.mkdtemp()
                try:
                    zip_path = os.path.join(UTA_RLDD_DIR, video_info["zip"])
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extract(video_info["path_in_zip"], tmp_dir)
                    video_path = os.path.join(tmp_dir, video_info["path_in_zip"])
                    if os.path.exists(video_path):
                        result = eval_video_raw(ssd_net, interp, video_path, preprocess_raw)
                        if result:
                            label = video_info["label"]
                            pred = result["pred"]
                            all_probs.append((result["mean_prob"], label))
                            if pred == 1 and label == 1: tp += 1
                            elif pred == 1 and label == 0: fp += 1
                            elif pred == 0 and label == 1: fn += 1
                            else: tn += 1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        
        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        far = fp / max(fp + tn, 1)  # false alarm rate on alert videos
        
        model_results[model_name] = {
            "size_mb": os.path.getsize(model_path) / 1024 / 1024,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "far": far,
            "probs": all_probs,
        }
        
        print(f"    Acc={acc*100:.1f}% Prec={prec*100:.1f}% Rec={rec*100:.1f}% "
              f"F1={f1:.3f} FAR={far*100:.1f}% (TP={tp} FP={fp} FN={fn} TN={tn})")
    
    # === PHASE 3: Threshold sweep on best model ===
    print("\n" + "=" * 70)
    print("PHASE 3: Threshold Sweep (best model from Phase 2)")
    print("=" * 70)
    
    # Find best model by F1
    best_model = max(model_results.keys(), key=lambda k: model_results[k]["f1"])
    print(f"Best model: {best_model} (F1={model_results[best_model]['f1']:.3f})")
    
    best_probs = model_results[best_model]["probs"]
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    threshold_results = []
    
    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for prob, label in best_probs:
            pred = 1 if prob > thresh else 0
            if pred == 1 and label == 1: tp += 1
            elif pred == 1 and label == 0: fp += 1
            elif pred == 0 and label == 1: fn += 1
            else: tn += 1
        
        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        far = fp / max(fp + tn, 1)
        
        threshold_results.append({
            "threshold": thresh, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "far": far,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        
        marker = ""
        if rec >= 0.95 and f1 > 0.4:
            marker = " <-- GOOD"
        if rec >= 1.0 and far < 0.5:
            marker = " <-- BEST"
        
        print(f"  thr={thresh:.2f} Acc={acc*100:5.1f}% Prec={prec*100:5.1f}% "
              f"Rec={rec*100:5.1f}% F1={f1:.3f} FAR={far*100:5.1f}% "
              f"(TP={tp:2d} FP={fp:2d} FN={fn:2d} TN={tn:2d}){marker}")
    
    # === PHASE 4: Preprocessing comparison on best model ===
    print("\n" + "=" * 70)
    print("PHASE 4: Preprocessing Comparison (best model, threshold=0.5)")
    print("=" * 70)
    
    interp_best = load_tflite(MODELS[best_model])
    
    for pp_name, pp_fn in PREPROCESSING.items():
        tp = fp = fn = tn = 0
        for subject in subject_ids[:12]:  # Quick sample
            for video_info in subjects[subject]:
                tmp_dir = tempfile.mkdtemp()
                try:
                    zip_path = os.path.join(UTA_RLDD_DIR, video_info["zip"])
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extract(video_info["path_in_zip"], tmp_dir)
                    video_path = os.path.join(tmp_dir, video_info["path_in_zip"])
                    if os.path.exists(video_path):
                        result = eval_video_raw(ssd_net, interp_best, video_path, pp_fn)
                        if result:
                            label = video_info["label"]
                            pred = result["pred"]
                            if pred == 1 and label == 1: tp += 1
                            elif pred == 1 and label == 0: fp += 1
                            elif pred == 0 and label == 1: fn += 1
                            else: tn += 1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        
        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        far = fp / max(fp + tn, 1)
        print(f"  {pp_name:8s}: Acc={acc*100:5.1f}% Prec={prec*100:5.1f}% "
              f"Rec={rec*100:5.1f}% F1={f1:.3f} FAR={far*100:5.1f}%")
    
    # === PHASE 5: Face quality filtering sweep ===
    print("\n" + "=" * 70)
    print("PHASE 5: Face Quality Filter Sweep (best model)")
    print("=" * 70)
    
    quality_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for q_thresh in quality_thresholds:
        tp = fp = fn = tn = 0
        skipped = 0
        for subject in subject_ids[:12]:
            for video_info in subjects[subject]:
                tmp_dir = tempfile.mkdtemp()
                try:
                    zip_path = os.path.join(UTA_RLDD_DIR, video_info["zip"])
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extract(video_info["path_in_zip"], tmp_dir)
                    video_path = os.path.join(tmp_dir, video_info["path_in_zip"])
                    if os.path.exists(video_path):
                        result = eval_video_raw(ssd_net, interp_best, video_path, 
                                               preprocess_raw, quality_thresh=q_thresh)
                        if result:
                            label = video_info["label"]
                            pred = result["pred"]
                            if pred == 1 and label == 1: tp += 1
                            elif pred == 1 and label == 0: fp += 1
                            elif pred == 0 and label == 1: fn += 1
                            else: tn += 1
                        else:
                            skipped += 1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        
        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        far = fp / max(fp + tn, 1)
        print(f"  q>={q_thresh:.1f}: Acc={acc*100:5.1f}% Prec={prec*100:5.1f}% "
              f"Rec={rec*100:5.1f}% F1={f1:.3f} FAR={far*100:5.1f}% "
              f"skipped={skipped}")
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\nBest model: {best_model} ({model_results[best_model]['size_mb']:.1f} MB)")
    print(f"Default performance: Acc={model_results[best_model]['accuracy']*100:.1f}% "
          f"Recall={model_results[best_model]['recall']*100:.1f}% "
          f"FAR={model_results[best_model]['far']*100:.1f}%")
    
    # Find optimal threshold (maximize F1 while keeping recall >= 95%)
    valid = [t for t in threshold_results if t["recall"] >= 0.95]
    if valid:
        best_thresh = max(valid, key=lambda t: t["f1"])
        print(f"\nOptimal threshold: {best_thresh['threshold']:.2f}")
        print(f"  Accuracy:  {best_thresh['accuracy']*100:.1f}%")
        print(f"  Precision: {best_thresh['precision']*100:.1f}%")
        print(f"  Recall:    {best_thresh['recall']*100:.1f}%")
        print(f"  F1:        {best_thresh['f1']:.3f}")
        print(f"  FAR:       {best_thresh['far']*100:.1f}%")
    
    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "optimization_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "size_mb", "accuracy", "precision", "recall", "f1", "far", "tp", "fp", "fn", "tn"])
        for name, r in model_results.items():
            w.writerow([name, f"{r['size_mb']:.1f}", f"{r['accuracy']:.4f}", 
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                        f"{r['far']:.4f}", r['tp'], r['fp'], r['fn'], r['tn']])
    
    csv_path2 = os.path.join(OUTPUT_DIR, "threshold_sweep.csv")
    with open(csv_path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "accuracy", "precision", "recall", "f1", "far", "tp", "fp", "fn", "tn"])
        for t in threshold_results:
            w.writerow([f"{t['threshold']:.2f}", f"{t['accuracy']:.4f}",
                        f"{t['precision']:.4f}", f"{t['recall']:.4f}", f"{t['f1']:.4f}",
                        f"{t['far']:.4f}", t['tp'], t['fp'], t['fn'], t['tn']])
    
    print(f"\nResults saved to {csv_path}")
    print(f"Threshold sweep saved to {csv_path2}")

if __name__ == "__main__":
    main()
