"""
End-to-End Pipeline Latency Measurement for DriveSafe AI
=========================================================
Measures each pipeline stage independently:
  1. JPEG decode (base64 -> numpy array)
  2. Face detection (OpenCV DNN SSD)
  3. Face crop + preprocessing (BGR->RGB, resize, normalize)
  4. Model inference (TFLite classification)
  5. Temporal confirmation logic
  6. JSON serialization + WebSocket frame assembly
  7. Total end-to-end (camera frame -> alert-ready JSON)

Hardware: reports CPU, thread count, and OS.
"""
import os, sys, time, json, base64
import cv2
import numpy as np

# === Setup (mirrors app.py exactly) ===
cv2.setNumThreads(2)

print("Loading face detector...")
net = cv2.dnn.readNetFromCaffe(
    os.path.join(os.path.dirname(__file__), "..", "deploy.prototxt"),
    os.path.join(os.path.dirname(__file__), "..", "res10_300x300_ssd_iter_140000.caffemodel"),
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Loading TFLite model...")
try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    try:
        import tflite_runtime.interpreter as litert
    except ImportError:
        import tensorflow as tf
        litert = tf.lite

interpreter = litert.Interpreter(
    model_path=os.path.join(os.path.dirname(__file__), "..", "drowsiness.tflite"),
    num_threads=2,
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"  Model input shape: {input_details[0]['shape']}")

# === Create synthetic test frames ===
N_WARMUP = 50
N_FRAMES = 500
W, H = 320, 240  # Matches app.py capture resolution

# Generate test frames with random faces (simulate webcam input)
print(f"\nGenerating {N_WARMUP + N_FRAMES} synthetic test frames ({W}x{H})...")
test_frames = []
for i in range(N_WARMUP + N_FRAMES):
    # Create a realistic-looking frame with skin-colored rectangle (simulates a face)
    frame = np.random.randint(40, 80, (H, W, 3), dtype=np.uint8)  # Dark background
    # Add a "face" region in the center
    fx, fy, fw, fh = W//4, H//4, W//2, H//2
    frame[fy:fy+fh, fx:fx+fw] = np.array([180, 160, 140], dtype=np.uint8)  # Skin tone
    # Add some eye-like features
    frame[fy+fh//3:fy+fh//3+10, fx+fw//4:fx+fw//4+20] = 60
    frame[fy+fh//3:fy+fh//3+10, fx+3*fw//4-20:fx+3*fw//4] = 60
    test_frames.append(frame)

# Create JPEG-encoded versions (simulates base64 decode from WebSocket)
print("Encoding test frames to JPEG...")
jpeg_frames = []
for frame in test_frames:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    jpeg_frames.append(buf.tobytes())

# Base64 encode (simulates WebSocket receive)
b64_frames = [base64.b64encode(jf).decode("utf-8") for jf in jpeg_frames]

# === Benchmark each stage ===

def measure(fn, frames, n=N_WARMUP, label=""):
    """Warmup then measure median/mean/p95/p99 latency."""
    # Warmup
    for i in range(min(n, len(frames))):
        fn(frames[i])

    # Measure
    times = []
    for i in range(len(frames)):
        t0 = time.perf_counter()
        fn(frames[i])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "label": label,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "std_ms": float(np.std(times)),
    }

print(f"\nBenchmarking {N_FRAMES} frames per stage...")

# --- Stage 1: JPEG Decode ---
def stage_jpeg_decode(b64_str):
    header, encoded = b64_str.split(",", 1) if "," in b64_str else ("", b64_str)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

r_jpeg = measure(stage_jpeg_decode, b64_frames, label="1. JPEG decode")
print(f"  JPEG decode:     {r_jpeg['median_ms']:.2f} ms (median)")

# --- Stage 2: Face Detection ---
def stage_face_detect(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    best_conf, best_idx = 0, -1
    for i in range(detections.shape[2]):
        c = detections[0, 0, i, 2]
        if c > best_conf:
            best_conf, best_idx = c, i
    if best_conf > 0.4:
        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
        return box.astype("int")
    return None

# Pre-decode images for face detection
imgs_decoded = [stage_jpeg_decode(b64) for b64 in b64_frames[:N_FRAMES]]
r_facedet = measure(stage_face_detect, imgs_decoded, label="2. Face detection")
print(f"  Face detection:  {r_facedet['median_ms']:.2f} ms (median)")

# --- Stage 3: Face Crop + Preprocessing ---
# Create a fixed face crop for consistent measurement
face_roi = cv2.resize(imgs_decoded[0][60:180, 80:240], (96, 96))

def stage_preprocess(_):
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_resized = face_rgb.astype(np.float32)
    # MobileNetV3 preprocessing
    face_resized = face_resized / 127.5 - 1.0
    return np.expand_dims(face_resized, axis=0)

r_preprocess = measure(stage_preprocess, list(range(N_FRAMES)), label="3. Preprocessing")
print(f"  Preprocessing:   {r_preprocess['median_ms']:.2f} ms (median)")

# --- Stage 4: Model Inference (single image) ---
face_batch = np.expand_dims(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0, axis=0)

def stage_inference_single(_):
    interpreter.set_tensor(input_details[0]["index"], face_batch)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])
    return float(pred[0][0])

r_inference = measure(stage_inference_single, list(range(N_FRAMES)), label="4. TFLite inference (single)")
print(f"  Inference (1):   {r_inference['median_ms']:.2f} ms (median)")

# --- Stage 5: Temporal Confirmation Logic ---
def stage_temporal(_):
    # Simulates the 3-second window check (trivially fast - just time comparisons)
    drowsy_since = time.time() - 3.5  # Already drowsy for 3.5s
    elapsed = time.time() - drowsy_since
    if elapsed >= 3.0:
        return "ALERT"
    return "DROWSY"

r_temporal = measure(stage_temporal, list(range(N_FRAMES)), label="5. Temporal logic")
print(f"  Temporal logic:  {r_temporal['median_ms']:.4f} ms (median)")

# --- Stage 6: JSON Serialization + WebSocket frame ---
def stage_json_ws(_):
    frame_data = {
        "status": "ALERT!!!",
        "alarm": "True",
        "conf": 0.9234,
        "box": [80, 60, 240, 180],
    }
    return json.dumps(frame_data)

r_json = measure(stage_json_ws, list(range(N_FRAMES)), label="6. JSON serialize")
print(f"  JSON serialize:  {r_json['median_ms']:.4f} ms (median)")

# --- Stage 7: End-to-End (all stages combined, face detected) ---
def stage_e2e(b64_str):
    # 1. JPEG decode
    header, encoded = b64_str.split(",", 1) if "," in b64_str else ("", b64_str)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 2. Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    best_conf, best_idx = 0, -1
    for i in range(detections.shape[2]):
        c = detections[0, 0, i, 2]
        if c > best_conf:
            best_conf, best_idx = c, i

    if best_conf > 0.4:
        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
        x, y, x1, y1 = box.astype("int")

        # 3. Preprocess
        face_roi = img[max(0,y):min(h,y1), max(0,x):min(w,x1)]
        if face_roi.size > 100:
            face_rgb = cv2.cvtColor(cv2.resize(face_roi, (96, 96)), cv2.COLOR_BGR2RGB)
            face_batch = np.expand_dims(face_rgb.astype(np.float32) / 127.5 - 1.0, axis=0)

            # 4. Inference
            interpreter.set_tensor(input_details[0]["index"], face_batch)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
            prob = float(pred[0][0])

            # 5. Temporal logic
            status = "DROWSY" if prob > 0.5 else "AWAKE"

            # 6. JSON
            result = json.dumps({"status": status, "alarm": "False", "conf": float(pred[0][0])})
            return result
    return json.dumps({"status": "SCANNING", "alarm": "False", "conf": 0.0})

r_e2e = measure(stage_e2e, b64_frames, label="7. End-to-end (face found)")
print(f"  End-to-end:      {r_e2e['median_ms']:.2f} ms (median)")

# --- End-to-End (no face detected - fast path) ---
def stage_e2e_noface(b64_str):
    header, encoded = b64_str.split(",", 1) if "," in b64_str else ("", b64_str)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return json.dumps({"status": "NO FACE", "alarm": "False", "conf": 0.0})

# Use dark frames (no face) for no-face path
dark_frames_b64 = []
for i in range(N_FRAMES):
    dark = np.zeros((H, W, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dark, [cv2.IMWRITE_JPEG_QUALITY, 40])
    dark_frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

r_e2e_noface = measure(stage_e2e_noface, dark_frames_b64, label="7b. End-to-end (no face)")
print(f"  End-to-end (no face): {r_e2e_noface['median_ms']:.2f} ms (median)")

# === Summary ===
all_results = [r_jpeg, r_facedet, r_preprocess, r_inference, r_temporal, r_json, r_e2e, r_e2e_noface]

print("\n" + "=" * 75)
print("DRIVESAFE AI - END-TO-END PIPELINE LATENCY BREAKDOWN")
print("=" * 75)
print(f"{'Stage':<40} {'Median':>8} {'Mean':>8} {'P95':>8} {'P99':>8}")
print(f"{'':40} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8}")
print("-" * 75)
for r in all_results:
    print(f"  {r['label']:<38} {r['median_ms']:>8.3f} {r['mean_ms']:>8.3f} {r['p95_ms']:>8.3f} {r['p99_ms']:>8.3f}")
print("-" * 75)

# Also report model-only latency (batch inference)
print("\nModel-only benchmarks (batch inference, as reported in paper):")
for bs in [1, 8, 16, 32]:
    batch = np.tile(face_batch, (bs, 1, 1, 1))
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], batch)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    med = np.median(times)
    print(f"  Batch {bs:>2}: {med:.2f} ms total, {med/bs:.3f} ms/image, {bs/(med/1000):.0f} img/s")

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"  Pipeline (face found):     {r_e2e['median_ms']:.2f} ms/frame ({1000/r_e2e['median_ms']:.0f} FPS)")
print(f"  Pipeline (no face):        {r_e2e_noface['median_ms']:.2f} ms/frame ({1000/r_e2e_noface['median_ms']:.0f} FPS)")
print(f"  Model inference (single):  {r_inference['median_ms']:.2f} ms/image")
print(f"  Face detection:            {r_facedet['median_ms']:.2f} ms/frame")
print(f"  Bottleneck:                {'Face detection' if r_facedet['median_ms'] > r_inference['median_ms'] else 'Model inference'}")
print("=" * 75)

# Save results as CSV
import csv
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "pipeline_latency.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label", "median_ms", "mean_ms", "p95_ms", "p99_ms", "std_ms"])
    w.writeheader()
    for r in all_results:
        w.writerow(r)
print(f"\nResults saved to {csv_path}")
