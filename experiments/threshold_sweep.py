"""Three-Zone Threshold Sweep Ablation
Run: python experiments/threshold_sweep.py
Requires: akahana dataset (same as main training)
Output: results/threshold_sweep.csv + console visualization
"""
import numpy as np
import os, sys

# --- Config ---
BASE_MODEL = os.path.join("..", "drowsiness_v2.keras") if os.path.exists("../drowsiness_v2.keras") else "drowsiness_v2.keras"
RESULTS_DIR = "../results" if os.path.exists("../results") else "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    import tensorflow as tf
    import tensorflow.keras as tfk

    # Load model
    print("Loading model...")
    m = tfk.models.load_model(BASE_MODEL)

    # Load validation data (same split as training)
    DATA_DIR = os.path.join("..", "Dataset", "augmented_data") if os.path.exists("../Dataset/augmented_data") else os.path.join("Dataset", "augmented_data")

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(rescale=1.0/127.5 - 1.0)
    val = gen.flow_from_directory(
        DATA_DIR, target_size=(96,96), batch_size=64,
        class_mode="binary", shuffle=False, subset="validation",
        seed=123, validation_split=0.2
    )

    # Get all predictions
    print("Running inference on validation set...")
    probs = m.predict(val, verbose=0).flatten()
    labels = val.classes
    n = len(probs)
    print(f"Validation samples: {n}")
    print(f"Class distribution: {labels.sum()}/{n} fatigue ({labels.sum()/n*100:.1f}%)")

    # Sweep thresholds
    thetas_a = np.arange(0.01, 0.20, 0.01)
    thetas_f = np.arange(0.80, 0.99, 0.01)

    results = []
    best_score = 0
    best_theta = (0, 0)

    print(f"Sweeping {len(thetas_a)} x {len(thetas_f)} = {len(thetas_a)*len(thetas_f)} threshold combinations...")
    print(f"{'theta_a':>8} {'theta_f':>8} {'coverage':>9} {'correct':>8} {'product':>8} {'F1':>6} {'FA%':>6}")
    print("-" * 60)

    for ta in thetas_a:
        for tf_val in thetas_f:
            # Classify
            preds = np.where(probs < ta, 0, np.where(probs > tf_val, 1, -1))

            # Coverage: fraction not refused
            covered = preds != -1
            coverage = covered.sum() / n

            if coverage < 0.5:
                continue

            # Correctness on covered samples
            correct = (preds[covered] == labels[covered]).sum() / covered.sum()

            # Product metric
            product = coverage * correct

            # F1 on covered
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # False alarm rate (wrong active predictions as fatigue)
            active_labels = labels == 0
            fa = ((preds == 1) & active_labels).sum() / active_labels.sum() * 100 if active_labels.sum() > 0 else 0

            results.append((ta, tf_val, coverage, correct, product, f1, fa))

            if product > best_score:
                best_score = product
                best_theta = (ta, tf_val)

    # Print top 10
    results.sort(key=lambda x: x[4], reverse=True)
    print(f"Top 10 threshold combinations (by coverage x correctness):")
    print(f"{'theta_a':>8} {'theta_f':>8} {'coverage':>9} {'correct':>8} {'product':>8} {'F1':>6} {'FA%':>6}")
    print("-" * 60)
    for r in results[:10]:
        print(f"{r[0]:8.2f} {r[1]:8.2f} {r[2]:8.1%} {r[3]:8.1%} {r[4]:8.4f} {r[5]:6.3f} {r[6]:6.2f}")

    print(f"Best: theta_a={best_theta[0]:.2f}, theta_f={best_theta[1]:.2f}, product={best_score:.4f}")

    # Save CSV
    import csv
    out = os.path.join(RESULTS_DIR, "threshold_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_a", "theta_f", "coverage", "correctness", "product", "f1", "false_alarm_pct"])
        w.writerows(results)
    print(f"Results saved to {out}")

    # Print ASCII visualization of product heatmap
    print(f"Coverage x Correctness Heatmap (top region):")
    print(f"     ", end="")
    for tf_val in thetas_f[::2]:
        print(f"{tf_val:.2f}", end="  ")
    print()
    for ta in thetas_a[::2]:
        print(f"{ta:.2f} ", end="")
        for tf_val in thetas_f[::2]:
            match = [r for r in results if abs(r[0]-ta)<0.005 and abs(r[1]-tf_val)<0.005]
            if match:
                score = match[0][4]
                if score > 0.9: print("##", end="  ")
                elif score > 0.8: print("**", end="  ")
                elif score > 0.7: print("..", end="  ")
                else: print("  ", end="  ")
            else:
                print("--", end="  ")
        print()
    print("Legend: ## = product>0.9, ** = >0.8, .. = >0.7, -- = <0.5 coverage")

if __name__ == "__main__":
    main()
