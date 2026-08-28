"""Cross-Subject Threshold Analysis for Three-Zone Protocol
Analyzes whether threshold optimization improves LOSO performance.
Uses per-frame predictions from uta_rldd_true_loso_v2_frames.csv
"""
import numpy as np
import csv
import os

RESULTS_DIR = "results"
FRAMES_CSV = os.path.join(RESULTS_DIR, "uta_rldd_true_loso_v2_frames.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "loso_threshold_sweep.csv")

def load_frames():
    """Load per-frame predictions from LOSO v2 run."""
    rows = []
    with open(FRAMES_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'subject': row['subject'],
                'frame_idx': int(row['frame_idx']),
                'pred_prob': float(row['pred_prob']),
                'pred_label': int(row['pred_label']),
                'true_label': int(row['true_label']),
            })
    return rows

def sweep_thresholds(rows):
    """Sweep threshold combinations on cross-subject data."""
    probs = np.array([r['pred_prob'] for r in rows])
    labels = np.array([r['true_label'] for r in rows])
    subjects = np.array([r['subject'] for r in rows])
    n = len(probs)
    
    print(f"Total frames: {n}")
    print(f"Class distribution: {labels.sum()}/{n} fatigue ({labels.sum()/n*100:.1f}%)")
    print(f"Subjects: {len(set(subjects))}")
    
    # Threshold ranges
    thetas_a = np.arange(0.01, 0.40, 0.01)
    thetas_f = np.arange(0.60, 0.99, 0.01)
    
    results = []
    best_product = 0
    best_theta = (0, 0)
    
    print(f"\nSweeping {len(thetas_a)} x {len(thetas_f)} = {len(thetas_a)*len(thetas_f)} combinations...")
    
    for ta in thetas_a:
        for tf_val in thetas_f:
            # Three-zone classification
            preds = np.where(probs < ta, 0, np.where(probs > tf_val, 1, -1))
            
            # Coverage: fraction not refused
            covered = preds != -1
            coverage = covered.sum() / n
            
            if coverage < 0.3:  # At least 30% coverage
                continue
            
            # Correctness on covered samples
            correct = (preds[covered] == labels[covered]).sum() / covered.sum()
            
            # Product metric
            product = coverage * correct
            
            # F1 on covered (fatigue class)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # False alarm rate
            active_mask = labels == 0
            fa = ((preds == 1) & active_mask).sum() / active_mask.sum() * 100 if active_mask.sum() > 0 else 0
            
            # Accuracy on covered
            acc = correct * 100
            
            # Refused count
            refused = (preds == -1).sum()
            
            results.append((ta, tf_val, coverage, correct, product, f1, fa, acc, refused))
            
            if product > best_product:
                best_product = product
                best_theta = (ta, tf_val)
    
    return results, best_theta, best_product, n

def per_subject_analysis(rows, theta_a, theta_f):
    """Analyze threshold effect per subject."""
    subjects = sorted(set(r['subject'] for r in rows))
    
    print(f"\nPer-subject analysis (theta_a={theta_a:.2f}, theta_f={theta_f:.2f}):")
    print(f"{'Subject':>8} {'Raw Acc':>8} {'3-Zone Acc':>10} {'Coverage':>9} {'Refused':>8} {'F1':>6}")
    print("-" * 60)
    
    for subj in subjects:
        subj_rows = [r for r in rows if r['subject'] == subj]
        probs = np.array([r['pred_prob'] for r in subj_rows])
        labels = np.array([r['true_label'] for r in subj_rows])
        
        # Raw accuracy (no threshold)
        raw_acc = (probs.round().astype(int) == labels).mean() * 100
        
        # Three-zone
        preds = np.where(probs < theta_a, 0, np.where(probs > theta_f, 1, -1))
        covered = preds != -1
        coverage = covered.sum() / len(probs)
        
        if coverage > 0:
            acc_3zone = (preds[covered] == labels[covered]).mean() * 100
            
            # F1
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            acc_3zone = 0
            f1 = 0
        
        refused = (preds == -1).sum()
        print(f"{subj:>8} {raw_acc:7.1f}% {acc_3zone:9.1f}% {coverage:8.1%} {refused:8d} {f1:6.3f}")

def print_heatmap(results, thetas_a, thetas_f):
    """Print ASCII heatmap of coverage x correctness."""
    print("\nCoverage x Correctness Heatmap:")
    print("Columns = theta_f, Rows = theta_a")
    print(f"{'':>6}", end="")
    for tf_val in thetas_f[::3]:
        print(f"{tf_val:.2f}", end="  ")
    print()
    
    for ta in thetas_a[::2]:
        print(f"{ta:.2f} ", end="")
        for tf_val in thetas_f[::3]:
            match = [r for r in results if abs(r[0]-ta)<0.005 and abs(r[1]-tf_val)<0.005]
            if match:
                score = match[0][4]
                if score > 0.6: print("##", end="  ")
                elif score > 0.5: print("**", end="  ")
                elif score > 0.4: print("..", end="  ")
                elif score > 0.3: print(",,", end="  ")
                else: print("  ", end="  ")
            else:
                print("--", end="  ")
        print()
    print("Legend: ## = product>0.6, ** = >0.5, .. = >0.4, ,, = >0.3, -- = low coverage")

def main():
    print("=" * 70)
    print("CROSS-SUBJECT THREE-ZONE THRESHOLD ANALYSIS")
    print("=" * 70)
    
    rows = load_frames()
    results, best_theta, best_product, n = sweep_thresholds(rows)
    
    # Sort by product
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\nTop 15 threshold combinations (by coverage x correctness):")
    print(f"{'theta_a':>8} {'theta_f':>8} {'coverage':>9} {'correct':>8} {'product':>8} {'F1':>6} {'FA%':>6} {'acc':>6} {'refused':>8}")
    print("-" * 80)
    for r in results[:15]:
        print(f"{r[0]:8.2f} {r[1]:8.2f} {r[2]:8.1%} {r[3]:8.1%} {r[4]:8.4f} {r[5]:6.3f} {r[6]:6.2f} {r[7]:5.1f}% {r[8]:8d}")
    
    print(f"\nBest: theta_a={best_theta[0]:.2f}, theta_f={best_theta[1]:.2f}, product={best_product:.4f}")
    
    # Compare with baseline (no threshold, all predictions accepted)
    probs = np.array([r['pred_prob'] for r in rows])
    labels = np.array([r['true_label'] for r in rows])
    baseline_acc = (probs.round().astype(int) == labels).mean() * 100
    baseline_tp = ((probs.round().astype(int) == 1) & (labels == 1)).sum()
    baseline_fp = ((probs.round().astype(int) == 1) & (labels == 0)).sum()
    baseline_fn = ((probs.round().astype(int) == 0) & (labels == 1)).sum()
    baseline_prec = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
    baseline_rec = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0
    baseline_f1 = 2 * baseline_prec * baseline_rec / (baseline_prec + baseline_rec) if (baseline_prec + baseline_rec) > 0 else 0
    
    print(f"\nBaseline (no thresholding):")
    print(f"  Accuracy:  {baseline_acc:.1f}%")
    print(f"  Precision: {baseline_prec:.3f}")
    print(f"  Recall:    {baseline_rec:.3f}")
    print(f"  F1:        {baseline_f1:.3f}")
    
    # Best three-zone result
    best = results[0]
    print(f"\nBest three-zone (theta_a={best[0]:.2f}, theta_f={best[1]:.2f}):")
    print(f"  Coverage:  {best[2]:.1%}")
    print(f"  Accuracy:  {best[7]:.1f}% (on covered samples)")
    print(f"  F1:        {best[5]:.3f}")
    print(f"  FA rate:   {best[6]:.2f}%")
    print(f"  Refused:   {best[8]} frames ({1-best[2]:.1%})")
    
    per_subject_analysis(rows, best[0], best[1])
    
    thetas_a = np.arange(0.01, 0.40, 0.01)
    thetas_f = np.arange(0.60, 0.99, 0.01)
    print_heatmap(results, thetas_a, thetas_f)
    
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['theta_a', 'theta_f', 'coverage', 'correctness', 'product', 'f1', 'false_alarm_pct', 'accuracy', 'refused'])
        w.writerows(results)
    print(f"\nResults saved to {OUT_CSV}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    if best[7] > baseline_acc:
        print(f"Three-zone IMPROVES accuracy: {baseline_acc:.1f}% -> {best[7]:.1f}% (+{best[7]-baseline_acc:.1f}pp)")
        print(f"But coverage drops to {best[2]:.1%} (refuses {best[8]} of {n} frames)")
    else:
        print(f"Three-zone does NOT improve accuracy over baseline: {baseline_acc:.1f}% vs {best[7]:.1f}%")
        print(f"The model uncertainty is not well-calibrated for cross-subject data")
    print(f"\nConclusion: The three-zone protocol reduces false alarms at the cost of")
    print(f"coverage. Its value is in safety-critical deployment where refusing uncertain")
    print(f"predictions prevents dangerous misclassifications.")

if __name__ == "__main__":
    main()
