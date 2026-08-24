import tensorflow as tf
import numpy as np
import os
import time
from collections import defaultdict

# 1. ENGINE SETUP (LiteRT Standard)
try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    try:
        import tflite_runtime.interpreter as litert
    except ImportError:
        from tensorflow import lite as litert

model_path = 'drowsiness.tflite'
if not os.path.exists(model_path):
    print(f"Error: Model '{model_path}' not found.")
    exit()

# Optimization: num_threads should match your physical CPU cores (usually 4 or 8)
interpreter = litert.Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. DATASET CONFIGURATION
IMG_SIZE = (96, 96) 
BATCH_SIZE = 32 
DATASET_PATH = 'dataset/train_cropped'

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = val_ds.class_names
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

# Resize input tensor to handle full batches at once
interpreter.resize_tensor_input(input_details[0]['index'], [BATCH_SIZE, 96, 96, 3])
interpreter.allocate_tensors()

# 3. WARMUP (Crucial for < 10ms Accuracy)
print("\n🔥 Warming up CPU caches...")
warmup_data = np.zeros((BATCH_SIZE, 96, 96, 3), dtype=np.float32)
for _ in range(5):
    interpreter.set_tensor(input_details[0]['index'], warmup_data)
    interpreter.invoke()

# 4. VECTORIZED EVALUATION
print(f"🚀 Benchmarking validation set in Turbo Mode...")
correct, total = 0, 0
confusion = defaultdict(lambda: defaultdict(int))

# We use perf_counter for microsecond precision
start_time = time.perf_counter()

for images, labels in val_ds:
    current_batch_size = images.shape[0]
    
    # Preprocess entire batch at once (Vectorized)
    # MobileNetV3 requires scaling to [-1, 1]
    preprocessed_batch = preprocess_input(images.numpy()).astype(np.float32)
    labels_np = labels.numpy()

    # Dynamic handle for the final (smaller) batch
    if current_batch_size != BATCH_SIZE:
        interpreter.resize_tensor_input(input_details[0]['index'], [current_batch_size, 96, 96, 3])
        interpreter.allocate_tensors()

    # THE SPEED SECRET: One call for 32 images
    interpreter.set_tensor(input_details[0]['index'], preprocessed_batch)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Log metrics
    for i in range(current_batch_size):
        pred = predictions[i]
        # Robust handling for Binary or Categorical TFLite exports
        pred_class = np.argmax(pred) if len(pred) > 1 else (1 if pred[0] > 0.5 else 0)
        true_class = np.argmax(labels_np[i])
        
        confusion[true_class][pred_class] += 1
        if pred_class == true_class:
            correct += 1
        total += 1

end_time = time.perf_counter()

# 5. FINAL REPORT
total_time_ms = (end_time - start_time) * 1000
avg_latency = total_time_ms / total

print("\n" + "="*55)
print(f"✅ FINAL ACCURACY:  { (correct / total) * 100:.2f}%")
print(f"📊 TOTAL IMAGES:    {total}")
print(f"⏱️  TURBO LATENCY:   {avg_latency:.2f} ms per image")
print(f"📦 THROUGHPUT:      {int(1000/avg_latency)} images / second")
print("="*55)

print("\n--- CONFUSION MATRIX ---")
print(f"{'':>15} | Pred {class_names[0]:>10} | Pred {class_names[1]:>10}")
for idx, name in enumerate(class_names):
    print(f"Actual {name:>8} | {confusion[idx][0]:>15} | {confusion[idx][1]:>15}")

# Cross-check: verify confusion matrix sums to total
cm_sum = sum(confusion[i][j] for i in range(len(class_names)) for j in range(len(class_names)))
if cm_sum != total:
    print(f"\n⚠️  WARNING: Confusion matrix sum ({cm_sum}) != total ({total}).")
    print(f"   This means {total - cm_sum} images were excluded or miscounted.")