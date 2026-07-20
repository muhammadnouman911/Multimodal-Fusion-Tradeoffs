import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'datasets'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'configs'))

from config import ExperimentConfig
try:
    from datasets.featLoader import LoadData
except ImportError:
    from featLoader import LoadData

def compute_entropy(labels, num_classes=70):
    counts = np.bincount(labels, minlength=num_classes)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def audit_folds(seed=42):
    print("=" * 70)
    print(f"RUNNING FOLD AUDIT & STRATIFICATION ANALYSIS (Seed {seed})")
    print("=" * 70)

    config = ExperimentConfig()
    ds_en = LoadData(config.train_csv_en, config.train_feats_dir, schema="train", lang_label=0)
    print(f"Total English Samples Loaded: {len(ds_en)} across {len(np.unique(ds_en.labels))} speakers")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    fold_stats = []
    max_entropy = np.log2(70)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(ds_en)), ds_en.labels)):
        train_labels = ds_en.labels[train_idx]
        val_labels = ds_en.labels[val_idx]

        train_entropy = compute_entropy(train_labels)
        val_entropy = compute_entropy(val_labels)

        train_counts = np.bincount(train_labels, minlength=70)
        val_counts = np.bincount(val_labels, minlength=70)

        min_val_samples = val_counts.min()
        max_val_samples = val_counts.max()
        zero_val_classes = np.sum(val_counts == 0)

        fold_stats.append({
            "Fold": fold_idx,
            "Train Samples": len(train_idx),
            "Val Samples": len(val_idx),
            "Train Entropy": f"{train_entropy:.4f} / {max_entropy:.4f}",
            "Val Entropy": f"{val_entropy:.4f} / {max_entropy:.4f}",
            "Min Val Class Count": min_val_samples,
            "Max Val Class Count": max_val_samples,
            "Zero Sample Classes (Val)": zero_val_classes
        })

    df_stats = pd.DataFrame(fold_stats)
    print("\nPER-FOLD STRATIFICATION AUDIT SUMMARY:")
    print(df_stats.to_string(index=False))

    audit_out = os.path.join(PROJECT_ROOT, "results", "fold_audit_report.csv")
    os.makedirs(os.path.dirname(audit_out), exist_ok=True)
    df_stats.to_csv(audit_out, index=False)
    print(f"\nAudit saved to {audit_out}")
    print("=" * 70)

if __name__ == "__main__":
    audit_folds(seed=42)
