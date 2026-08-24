"""
Temporal Window Ablation — Simulated Prediction Sequence
=========================================================
Generates realistic frame-level prediction sequences and measures how
different temporal confirmation windows affect false alarms, miss rate,
detection delay, precision, recall, and F1.

This does NOT require a camera or dataset — it uses statistically
representative simulated predictions based on the repository's reported
metrics (97.24% accuracy, 1.9% FAR, 4.0% miss rate).

Usage:
    python experiments/temporal_ablation_simulated.py

Outputs:
    - results/temporal_ablation.csv
    - results/temporal_ablation.tex
    - Console report
"""

import csv
import os
import numpy as np

OUTPUT_DIR = "results"
np.random.seed(42)

# --- Simulate realistic frame sequence ---
# Based on repository metrics:
# - 97.24% accuracy on 2,974 images
# - 1,782 active (TN=1748, FP=34) → 34/1782 = 1.9% FAR
# - 1,192 fatigue (TP=1144, FN=48) → 48/1192 = 4.0% miss rate
#
# Simulate a 5-minute driving session at 30 FPS = 9,000 frames
# Mix of alert segments (majority) with intermittent drowsy episodes

FPS = 30
DURATION_SEC = 300  # 5 minutes
TOTAL_FRAMES = FPS * DURATION_SEC  # 9,000

# Generate ground truth: mostly alert, with 3 drowsy episodes
ground_truth = np.zeros(TOTAL_FRAMES, dtype=int)  # 0=alert, 1=fatigue

# Episode 1: frames 1500-2100 (20 seconds of drowsiness)
ground_truth[1500:2100] = 1
# Episode 2: frames 4500-5100 (20 seconds)
ground_truth[4500:5100] = 1
# Episode 3: frames 7500:7800 (10 seconds)
ground_truth[7500:7800] = 1

# Generate predictions based on reported error rates
predictions = np.zeros(TOTAL_FRAMES, dtype=int)

for i in range(TOTAL_FRAMES):
    if ground_truth[i] == 0:  # Actually alert
        # 98.1% correct (1.9% false alarm rate)
        predictions[i] = 0 if np.random.random() > 0.019 else 1
    else:  # Actually fatigue
        # 96.0% correct (4.0% miss rate)
        predictions[i] = 1 if np.random.random() > 0.04 else 0

# Compute statistics
total_frames = len(predictions)
actual_alert = np.sum(ground_truth == 0)
actual_fatigue = np.sum(ground_truth == 1)
predicted_fatigue = np.sum(predictions == 1)
predicted_alert = np.sum(predictions == 0)

print(f"Simulated session: {DURATION_SEC}s at {FPS} FPS = {TOTAL_FRAMES} frames")
print(f"Ground truth: {actual_alert} alert, {actual_fatigue} fatigue "
      f"({actual_fatigue/total_frames*100:.1f}% fatigue)")
print(f"Predictions: {predicted_fatigue} fatigue, {predicted_alert} alert")
print()

# --- Temporal window ablation ---
WINDOWS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

results = []

for window_sec in WINDOWS:
    window_frames = int(window_sec * FPS)

    # Simulate temporal confirmation
    alerts = np.zeros(TOTAL_FRAMES, dtype=int)
    consecutive = 0
    alert_active = False

    for i in range(TOTAL_FRAMES):
        if predictions[i] == 1:  # Model predicts fatigue
            consecutive += 1
            if consecutive >= window_frames and not alert_active:
                alert_active = True
        else:
            consecutive = 0
            alert_active = False
        alerts[i] = int(alert_active)

    # Compute metrics
    tp = np.sum((alerts == 1) & (ground_truth == 1))
    fp = np.sum((alerts == 1) & (ground_truth == 0))
    tn = np.sum((alerts == 0) & (ground_truth == 0))
    fn = np.sum((alerts == 0) & (ground_truth == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    far = fp / max(fp + tn, 1)
    miss = fn / max(fn + tp, 1)

    # False alarms per minute
    alarm_minutes = DURATION_SEC / 60.0
    false_alarms_per_min = fp / alarm_minutes

    # Detection delay: average frames from fatigue onset to first alert
    delays = []
    in_episode = False
    episode_start = -1
    first_alert_in_episode = -1

    for i in range(TOTAL_FRAMES):
        if ground_truth[i] == 1 and not in_episode:
            in_episode = True
            episode_start = i
            first_alert_in_episode = -1
        elif ground_truth[i] == 0 and in_episode:
            in_episode = False
            if first_alert_in_episode > 0:
                delays.append(first_alert_in_episode - episode_start)

        if in_episode and alerts[i] == 1 and first_alert_in_episode == -1:
            first_alert_in_episode = i

    avg_delay_frames = np.mean(delays) if delays else 0
    avg_delay_sec = avg_delay_frames / FPS

    results.append({
        "window_sec": window_sec,
        "window_frames": window_frames,
        "total_alerts": int(np.sum(alerts)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "miss_rate": miss,
        "false_alarms_per_min": false_alarms_per_min,
        "avg_detection_delay_sec": avg_delay_sec,
    })

    print(f"Window {window_sec:4.1f}s ({window_frames:3d} fr): "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
          f"FAR={far:.4f} FA/min={false_alarms_per_min:.2f} "
          f"Delay={avg_delay_sec:.2f}s")

# --- Write CSV ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, "temporal_ablation.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nCSV saved to {csv_path}")

# --- Write LaTeX table ---
tex_path = os.path.join(OUTPUT_DIR, "temporal_ablation.tex")
with open(tex_path, "w") as f:
    f.write("% Auto-generated by temporal_ablation_simulated.py\n")
    f.write("% Simulated prediction sequence based on repository-reported metrics\n")
    f.write("% 9,000 frames (300s at 30 FPS), 3 drowsy episodes\n")
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Temporal Window Ablation Results (Simulated)}\n")
    f.write("\\label{tab:temporal_ablation}\n")
    f.write("\\begin{tabular}{ccccccc}\n\\toprule\n")
    f.write("Window & Frames & Precision & Recall & F1 & FA/min & Delay (s) \\\\\n\\midrule\n")
    for r in results:
        f.write(f"{r['window_sec']:4.1f} & {r['window_frames']:3d} "
                f"& {r['precision']:.3f} & {r['recall']:.3f} "
                f"& {r['f1']:.3f} & {r['false_alarms_per_min']:.2f} "
                f"& {r['avg_detection_delay_sec']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\vspace{2pt}\n")
    f.write("\\footnotesize{Simulated predictions based on repository-reported "
            "metrics (97.24\\% accuracy, 1.9\\% FAR, 4.0\\% miss rate). "
            "Results require validation on live camera data.}\n")
    f.write("\\end{table}\n")
print(f"LaTeX table saved to {tex_path}")

# --- Summary ---
print("\n" + "=" * 70)
print("KEY FINDINGS (simulated)")
print("=" * 70)

# Find optimal F1
best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
best = results[best_f1_idx]
print(f"Best F1: Window = {best['window_sec']}s, F1 = {best['f1']:.3f}")
print(f"  Precision = {best['precision']:.3f}, Recall = {best['recall']:.3f}")
print(f"  False alarms/min = {best['false_alarms_per_min']:.2f}")
print(f"  Detection delay = {best['avg_detection_delay_sec']:.2f}s")
print()

# Compare 0s vs 3s
r0 = results[0]
r3 = [r for r in results if r["window_sec"] == 3.0][0]
print(f"Frame-level (0s) vs 3s window:")
print(f"  FAR: {r0['false_alarm_rate']:.4f} → {r3['false_alarm_rate']:.4f} "
      f"({(1-r3['false_alarm_rate']/max(r0['false_alarm_rate'],1e-8))*100:.1f}% reduction)")
print(f"  F1:  {r0['f1']:.3f} → {r3['f1']:.3f}")
print(f"  Delay: {r0['avg_detection_delay_sec']:.2f}s → {r3['avg_detection_delay_sec']:.2f}s")
