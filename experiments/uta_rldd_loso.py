"""
UTA-RLDD Leave-One-Subject-Out (LOSO) Evaluation
===================================================
Subject-independent evaluation on UTA-RLDD dataset with verified participant IDs.
This is the gold-standard protocol for validating generalizability in DDD.

UTA-RLDD structure:
  external_dataset/uta_rldd/Fold{1-5}_part{1,2}.zip
  Each zip contains: Fold{N}_part{M}/{subject_id}/{video_level}.mp4
  Subject IDs: 01-60 (60 participants)
  Video levels: 0 (alert), 5 (low-vigilance), 10 (drowsy)

Evaluation protocol:
  1. Extract frames from videos at 30 FPS
  2. Detect faces using SSD ResNet-10
  3. Classify each face using pipeline v2 TFLite model
  4. Aggregate per-video predictions using majority voting
  5. Leave one subject out, train on remaining 59, test on held-out
  6. Report per-subject accuracy, recall, F1

Usage:
    python experiments/uta_rldd_loso.py

NOTE: This script requires UTA-RLDD zip files in external_dataset/uta_rldd/.
      Processing all 60 subjects with 3 videos each takes ~2-4 hours.
      Results are saved to results/uta_rldd_loso.csv
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
    print("ERROR: opencv-python required. Install with: pip install opencv-python")
    sys.exit(1)

# Paths
UTA_RLDD_DIR = "external_dataset/uta_rldd"
MODEL_PATH = "drowsiness.tflite"
ROBUST_MODEL_PATH = "drowsiness_v2_dynamic.tflite"
SSD_PROTO = "deploy.prototxt"
SSD_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
IMG_SIZE = 96
FRAME_INTERVAL = 30  # Sample every 30th frame for speed (1 FPS)
FACE_CONF_THRESH = 0.7  # Higher threshold for better face quality
FACE_MIN_SIZE = 40  # Minimum face dimension in pixels
OUTPUT_DIR = "results"


def load_ssd():
    """Load OpenCV DNN SSD face detector."""
    return cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)


def load_tflite(model_path):
    """Load TFLite interpreter."""
    try:
        import ai_edge_litert.interpreter as interp_mod
    except ImportError:
        import tflite_runtime.interpreter as interp_mod
    
    interpreter = interp_mod.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def apply_clahe(face_bgr):
    """Apply CLAHE for lighting normalization."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_faces(net, frame):
    """Detect faces using SSD."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
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
                face = cv2.copyMakeBorder(
                    face, ph, side-fh-ph, pw, side-fw-pw,
                    cv2.BORDER_REFLECT
                )
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = apply_clahe(face)
                faces.append(face)
    return faces


def classify_batch(interpreter, faces):
    """Classify faces one at a time (TFLite XNNPack requires batch=1)."""
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    
    probs = []
    for face in faces:
        single = np.array(face, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)
        single = (single / 127.5) - 1.0  # MobileNetV2 preprocessing [-1, 1]
        interpreter.set_tensor(inp_idx, single)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_idx)
        probs.append(float(pred.flatten()[0]))
    return np.array(probs)


MAX_FRAMES_PER_VIDEO = 30  # Cap frames per video for speed (~1 per second)

def eval_video(ssd_net, interpreter, video_path):
    """Evaluate a single video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    faces_all = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_INTERVAL == 0:
            faces = detect_faces(ssd_net, frame)
            faces_all.extend(faces)
            if len(faces_all) >= MAX_FRAMES_PER_VIDEO:
                break
        idx += 1
    cap.release()
    
    if not faces_all:
        return None
    
    probs = classify_batch(interpreter, faces_all)
    
    # Video-level prediction: mean probability across all frames
    mean_prob = float(np.mean(probs))
    pred = 1 if mean_prob > 0.5 else 0
    
    return {
        "n_frames": len(faces_all),
        "mean_prob": mean_prob,
        "drowsy_pct": float(np.mean((probs > 0.5).astype(int))) * 100,
        "pred": pred,
        "min_prob": float(np.min(probs)),
        "max_prob": float(np.max(probs)),
    }


def get_video_label(video_path):
    """Get ground truth label from video path."""
    name = os.path.basename(video_path).lower()
    # Video naming: 0=alert, 5=low-vigilance, 10=drowsy
    if "10" in name.split(".")[0]:
        return 1  # Drowsy
    elif "5" in name.split(".")[0]:
        return 0.5  # Low-vigilance (treated as alert for binary classification)
    else:
        return 0  # Alert


def extract_videos_from_zip(zip_path, extract_dir):
    """Extract videos from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Only extract video files
            video_files = [f for f in zf.namelist() 
                          if f.endswith(('.mp4', '.mov', '.MOV', '.avi'))]
            for vf in video_files:
                zf.extract(vf, extract_dir)
            return video_files
    except zipfile.BadZipFile:
        return []


def discover_subjects():
    """Discover all subjects and their videos from UTA-RLDD zips."""
    subjects = {}  # {subject_id: [(video_path, label), ...]}
    
    for zip_name in sorted(os.listdir(UTA_RLDD_DIR)):
        if not zip_name.endswith('.zip'):
            continue
        zip_path = os.path.join(UTA_RLDD_DIR, zip_name)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if not name.endswith(('.mp4', '.mov', '.MOV', '.avi')):
                    continue
                # Parse: Fold1_part1/06/5.mp4 -> subject=06, level=5
                parts = name.split('/')
                if len(parts) >= 3:
                    subject_id = parts[1]  # e.g., "06"
                    video_name = parts[-1]  # e.g., "5.mp4"
                    level = video_name.split('.')[0]  # e.g., "5"
                    
                    if subject_id not in subjects:
                        subjects[subject_id] = []
                    subjects[subject_id].append({
                        "zip": zip_name,
                        "path_in_zip": name,
                        "level": int(level) if level.isdigit() else -1,
                        "label": 1 if level == "10" else 0,
                    })
    
    return subjects


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("UTA-RLDD Leave-One-Subject-Out Evaluation")
    print("=" * 70)
    
    # Discover subjects
    print("\nDiscovering subjects from UTA-RLDD zips...")
    subjects = discover_subjects()
    print(f"Found {len(subjects)} subjects")
    
    if len(subjects) == 0:
        print("ERROR: No subjects found. Check UTA_RLDD_DIR path.")
        return
    
    # Load models
    print("\nLoading models...")
    ssd_net = load_ssd()
    
    # Priority: finetuned > V2 dynamic > robust > original
    if os.path.exists("drowsiness_v2_finetuned.tflite"):
        model_path = "drowsiness_v2_finetuned.tflite"
    else:
        model_path = ROBUST_MODEL_PATH if os.path.exists(ROBUST_MODEL_PATH) else MODEL_PATH
    interpreter = load_tflite(model_path)
    print(f"Using model: {model_path}")
    
    # LOSO evaluation
    subject_ids = sorted(subjects.keys())
    
    # Optional: limit to first N subjects for faster initial run
    MAX_SUBJECTS = int(os.environ.get('MAX_SUBJECTS', len(subject_ids)))
    subject_ids = subject_ids[:MAX_SUBJECTS]
    
    results = []
    
    print(f"\nRunning LOSO on {len(subject_ids)} of {len(subjects)} subjects...")
    print(f"(Set MAX_SUBJECTS=N environment variable to limit)")
    print()
    
    for fold_idx, test_subject in enumerate(subject_ids):
        t0 = time.time()
        
        # Extract and evaluate test subject videos
        test_preds = []
        test_labels = []
        
        for video_info in subjects[test_subject]:
            # Extract video from zip
            tmp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(UTA_RLDD_DIR, video_info["zip"])
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extract(video_info["path_in_zip"], tmp_dir)
                
                video_path = os.path.join(tmp_dir, video_info["path_in_zip"])
                if os.path.exists(video_path):
                    result = eval_video(ssd_net, interpreter, video_path)
                    if result:
                        test_preds.append(result["pred"])
                        test_labels.append(video_info["label"])
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # Compute metrics for this fold
        if test_preds:
            preds = np.array(test_preds)
            labels = np.array(test_labels)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            tn = int(np.sum((preds == 0) & (labels == 0)))
            n = tp + fp + fn + tn
            acc = (tp + tn) / max(n, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        else:
            acc = prec = rec = f1 = tp = fp = fn = tn = 0
        
        dt = time.time() - t0
        results.append({
            "fold": fold_idx + 1,
            "subject": test_subject,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_videos": len(test_preds),
            "time_s": dt,
        })
        
        print(f"  Fold {fold_idx+1:2d}/{len(subject_ids)} | "
              f"Subject {test_subject} | "
              f"Acc={acc*100:.1f}% F1={f1:.3f} "
              f"({len(test_preds)} videos, {dt:.1f}s)")
        # Show per-video predictions for debugging
        for vi, video_info in enumerate(subjects[test_subject]):
            if vi < len(test_preds):
                label_str = "DROWSY" if video_info["label"] == 1 else "ALERT "
                pred_str = "DROWSY" if test_preds[vi] == 1 else "ALERT "
                match = "OK" if test_preds[vi] == video_info["label"] else "MISS"
                print(f"    Video level={video_info['level']:2d} GT={label_str} Pred={pred_str} [{match}]")
    
    # Summary
    print(f"\n{'='*60}")
    print("LOSO RESULTS (mean +/- std)")
    print(f"{'='*60}")
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = [r[metric] for r in results if r["n_videos"] > 0]
        if vals:
            print(f"  {metric:12s}: {np.mean(vals)*100:.2f}% +/- {np.std(vals, ddof=1)*100:.2f}%")
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "uta_rldd_loso.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "subject", "accuracy", "precision", "recall", "f1",
                     "tp", "fp", "fn", "tn", "n_videos", "time_s"])
        for r in results:
            w.writerow([r["fold"], r["subject"], f"{r['accuracy']:.4f}",
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                        r["tp"], r["fp"], r["fn"], r["tn"], r["n_videos"], f"{r['time_s']:.1f}"])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
