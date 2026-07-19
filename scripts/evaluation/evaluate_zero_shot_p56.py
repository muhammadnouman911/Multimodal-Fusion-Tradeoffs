"""
Zero-Shot P5/P6 Evaluator (Enhanced)
======================================
Loads the best checkpoint from train_zero_shot.py and evaluates it
on the V3 German labeled dataset to produce clean, bias-free P5/P6.

Produces:
- P5 (cross-lingual multimodal accuracy)
- P6 (cross-lingual audio-only accuracy)
- Modality gap, Language gap
- Per-speaker accuracy
- Confusion matrix (saved as JSON + PNG)
- IEEE-style summary

Usage:
    python evaluate_zero_shot_p56.py --ckpt <path> --out_dir results/zero_shot_hybrid_seed42_fold0
"""
import os, sys, json, argparse
import numpy as np
import torch


import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'configs'))

from config import ExperimentConfig, CHALLENGE_ROOT
from utils.featLoader import LoadData
from models.multibranch import MultiBranchFOP
from models.fop import FOP

# V3 German dataset paths
# V3 German dataset paths
V3_GERMAN_CSV      = os.path.join(CHALLENGE_ROOT, "v3", "csv", "v3_train_German.csv")
V3_GERMAN_FEAT_DIR = os.path.join(CHALLENGE_ROOT, "v3")

# Urdu dataset paths
URDU_CSV      = os.path.join(CHALLENGE_ROOT, "Train", "comp", "v1_train_Urdu.csv")
URDU_FEAT_DIR = os.path.join(CHALLENGE_ROOT, "Train", "train(1)")

# English training path for label mapping
EN_TRAIN_CSV = os.path.join(CHALLENGE_ROOT, "Train", "comp", "v1_train_English.csv")


def compute_per_speaker_accuracy(preds, labels):
    """Compute per-speaker accuracy from numpy arrays."""
    unique_spk = np.unique(labels)
    per_spk = {}
    for spk in unique_spk:
        mask = labels == spk
        acc  = (preds[mask] == labels[mask]).mean() * 100
        per_spk[int(spk)] = round(float(acc), 2)
    return per_spk


def compute_confusion_matrix(preds, labels, n_classes):
    """Compute NxN confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, labels):
        cm[int(t), int(p)] += 1
    return cm


def evaluate(ckpt_path: str, dataset_choice: str = "german", out_dir: str = None):
    config = ExperimentConfig()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[P5/P6] Loading checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)

    # Resolve checkpoint metadata
    state_dict    = ckpt["model_state"]
    config.fusion = ckpt.get("fusion", config.fusion)
    ckpt_p3       = ckpt.get("P3", "N/A")
    ckpt_p4       = ckpt.get("P4", "N/A")
    ckpt_epoch    = ckpt.get("epoch", "N/A")
    ckpt_fold     = ckpt.get("fold", "N/A")
    ckpt_seed     = ckpt.get("seed", "N/A")

    print(f"[P5/P6] Checkpoint metadata: Seed={ckpt_seed} Fold={ckpt_fold} Ep={ckpt_epoch} "
          f"P3={ckpt_p3:.1f}% P4={ckpt_p4:.1f}%" if isinstance(ckpt_p3, float) else
          f"[P5/P6] Checkpoint metadata loaded.")

    # ── Load Target (unseen language) dataset ──────────────────────────────────
    if dataset_choice == "german":
        if not os.path.exists(V3_GERMAN_CSV):
            alt_csv  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "polysim-main_", "data", "v3_train_German.csv"))
            alt_feat = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "polysim-main_", "features"))
            if os.path.exists(alt_csv):
                eval_csv, eval_feat_dir = alt_csv, alt_feat
                print(f"[P5/P6] Using fallback path: {eval_csv}")
            else:
                raise FileNotFoundError(f"Could not locate V3 German CSV at {V3_GERMAN_CSV} or {alt_csv}")
        else:
            eval_csv, eval_feat_dir = V3_GERMAN_CSV, V3_GERMAN_FEAT_DIR
    elif dataset_choice == "urdu":
        eval_csv, eval_feat_dir = URDU_CSV, URDU_FEAT_DIR
    else:
        raise ValueError(f"Unknown dataset_choice: {dataset_choice}")

    # ── Load English mapping to align labels ──────────────────────────────────
    import pandas as pd
    en_df = pd.read_csv(EN_TRAIN_CSV)
    identity_to_label = dict(zip(en_df["identity"], en_df["label"]))

    print(f"[P5/P6] Loading {dataset_choice.capitalize()} dataset: {eval_csv}")
    eval_ds = LoadData(eval_csv, eval_feat_dir, schema="train", lang_label=1)
    
    # OVERRIDE LABELS WITH ENGLISH LABEL MAP
    eval_df = pd.read_csv(eval_csv)
    mapped_labels = [identity_to_label[uid] for uid in eval_df["identity"] if uid in identity_to_label]
    
    # Filter out identities not present in English (if any)
    valid_idx = [i for i, uid in enumerate(eval_df["identity"]) if uid in identity_to_label]
    eval_ds.face_feats = eval_ds.face_feats[valid_idx]
    eval_ds.audio_feats = eval_ds.audio_feats[valid_idx]
    eval_ds.labels = np.array(mapped_labels, dtype=np.int64)

    n_samples = len(eval_ds.labels)
    n_spk     = len(np.unique(eval_ds.labels))
    print(f"[P5/P6] {dataset_choice.capitalize()} dataset: {n_samples} samples | {n_spk} unique speakers (Aligned to English labels)")

    g_face  = torch.from_numpy(eval_ds.face_feats).float().to(config.device)
    g_audio = torch.from_numpy(eval_ds.audio_feats).float().to(config.device)
    g_lbls  = torch.from_numpy(eval_ds.labels).long()

    # ── Build model ────────────────────────────────────────────────────────────
    face_dim  = ckpt.get("face_dim",  g_face.shape[1])
    audio_dim = ckpt.get("audio_dim", g_audio.shape[1])
    model_type = ckpt.get("model_type", "multibranch_fop")
    
    if model_type == "fop":
        # Override config fusion to match checkpoint
        config.fusion = ckpt.get("fusion", "linear")
        print(f"[P5/P6] Instantiating FOP model with fusion: {config.fusion}")
        model = FOP(config, face_dim=face_dim, audio_dim=audio_dim)
    else:
        print("[P5/P6] Instantiating MultiBranchFOP model")
        model = MultiBranchFOP(config, face_dim=face_dim, audio_dim=audio_dim)
        
    model.load_state_dict(state_dict, strict=False)
    model = model.to(config.device).eval()

    print("[P5/P6] Model loaded. Running inference...")

    # Helper to get logits from any model output format
    def get_logits(out):
        if isinstance(out, dict):
            return out["fusion_logits"]
        elif isinstance(out, tuple):
            return out[1]
        else:
            return out

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        # P5: Multimodal (face + audio)
        out_p5  = model(g_face, g_audio.to(config.device))
        logits_p5 = get_logits(out_p5)
        preds_p5 = logits_p5.argmax(1).cpu().numpy()
        p5       = (preds_p5 == g_lbls.numpy()).mean() * 100

        # P6: Audio-only (face zeroed out)
        out_p6  = model(torch.zeros_like(g_face), g_audio.to(config.device))
        logits_p6 = get_logits(out_p6)
        preds_p6 = logits_p6.argmax(1).cpu().numpy()
        p6       = (preds_p6 == g_lbls.numpy()).mean() * 100

    g_lbls_np = g_lbls.numpy()
    modality_gap  = abs(p5 - p6)
    mean_score    = (p5 + p6) / 2.0
    # Language gap: how much P5/P6 differs from the English-trained best P4
    en_p4         = float(ckpt_p4) if isinstance(ckpt_p4, (int, float)) else None
    language_gap  = (en_p4 - p5) if en_p4 is not None else None

    # ── Per-speaker accuracy ───────────────────────────────────────────────────
    per_spk_p5 = compute_per_speaker_accuracy(preds_p5, g_lbls_np)
    per_spk_p6 = compute_per_speaker_accuracy(preds_p6, g_lbls_np)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    # Since model is trained on 70 English speakers, cm should be 70x70 to hold all predictions
    n_classes = 70 
    cm_p5 = compute_confusion_matrix(preds_p5, g_lbls_np, n_classes).tolist()
    cm_p6 = compute_confusion_matrix(preds_p6, g_lbls_np, n_classes).tolist()

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"ZERO-SHOT CROSS-LINGUAL EVALUATION (Unseen {dataset_choice.capitalize()})")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Trained on: English-only | Fold={ckpt_fold} | Seed={ckpt_seed}")
    print("=" * 65)
    print(f"  P5 (Cross-Lingual Multimodal):       {p5:.4f}%")
    print(f"  P6 (Cross-Lingual Audio-Only):        {p6:.4f}%")
    print(f"  Modality Gap  (P5 - P6):             {modality_gap:.4f}%")
    if language_gap is not None:
        print(f"  Language Gap  (EnP4 - P5):           {language_gap:.4f}%")
    print(f"  Mean Score    (P5 + P6) / 2:         {mean_score:.4f}%")
    print(f"  {dataset_choice.capitalize()} Speakers:                     {n_spk}")
    print(f"  {dataset_choice.capitalize()} Samples:                      {n_samples}")
    print("=" * 65)

    # Per-speaker summary
    spk_p5_vals = list(per_spk_p5.values())
    spk_p6_vals = list(per_spk_p6.values())
    print(f"\nPer-Speaker P5: Mean={np.mean(spk_p5_vals):.2f}% | Min={np.min(spk_p5_vals):.2f}% | Max={np.max(spk_p5_vals):.2f}%")
    print(f"Per-Speaker P6: Mean={np.mean(spk_p6_vals):.2f}% | Min={np.min(spk_p6_vals):.2f}% | Max={np.max(spk_p6_vals):.2f}%")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "checkpoint": ckpt_path,
        "ckpt_epoch": ckpt_epoch,
        "ckpt_fold":  ckpt_fold,
        "ckpt_seed":  ckpt_seed,
        "ckpt_P3":    ckpt_p3,
        "ckpt_P4":    ckpt_p4,
        "P5": round(p5, 4),
        "P6": round(p6, 4),
        "modality_gap":  round(modality_gap, 4),
        "language_gap":  round(language_gap, 4) if language_gap is not None else None,
        "mean_score":    round(mean_score, 4),
        "dataset":       dataset_choice,
        "n_eval_spk":    n_spk,
        "n_eval_samp":   n_samples,
        "per_speaker_P5": per_spk_p5,
        "per_speaker_P6": per_spk_p6,
        "confusion_matrix_P5": cm_p5,
        "confusion_matrix_P6": cm_p6,
    }

    if out_dir:
        res_path = os.path.join(out_dir, "p56_results.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[P5/P6] Results saved to: {res_path}")

        # IEEE-style text summary
        summary_txt = os.path.join(out_dir, "ieee_p56_summary.txt")
        with open(summary_txt, "w") as f:
            f.write("=" * 65 + "\n")
            f.write("IEEE ZERO-SHOT CROSS-LINGUAL EVALUATION SUMMARY\n")
            f.write("=" * 65 + "\n")
            f.write(f"Model Checkpoint : {os.path.basename(ckpt_path)}\n")
            f.write(f"Training Language: English Only (Zero-Shot Protocol)\n")
            f.write(f"Test Language    : {dataset_choice.capitalize()} (Unseen at Train Time)\n")
            f.write(f"Fold             : {ckpt_fold} | Seed: {ckpt_seed}\n")
            f.write(f"Best English P3  : {ckpt_p3}%\n")
            f.write(f"Best English P4  : {ckpt_p4}%\n")
            f.write("-" * 65 + "\n")
            f.write(f"P5 (Cross-Lingual Multimodal)      : {p5:.4f}%\n")
            f.write(f"P6 (Cross-Lingual Audio-Only)       : {p6:.4f}%\n")
            f.write(f"Modality Gap (|P5 - P6|)           : {modality_gap:.4f}%\n")
            if language_gap is not None:
                f.write(f"Language Gap (EnP4 - P5)           : {language_gap:.4f}%\n")
            f.write(f"Mean Score ((P5+P6)/2)             : {mean_score:.4f}%\n")
            f.write("-" * 65 + "\n")
            f.write(f"{dataset_choice.capitalize()} speakers evaluated: {n_spk}\n")
            f.write(f"{dataset_choice.capitalize()} samples evaluated : {n_samples}\n")
            f.write(f"Per-Spk P5: Mean={np.mean(spk_p5_vals):.2f}% Min={np.min(spk_p5_vals):.2f}% Max={np.max(spk_p5_vals):.2f}%\n")
            f.write(f"Per-Spk P6: Mean={np.mean(spk_p6_vals):.2f}% Min={np.min(spk_p6_vals):.2f}% Max={np.max(spk_p6_vals):.2f}%\n")
            f.write("=" * 65 + "\n")
        print(f"[P5/P6] IEEE summary saved to: {summary_txt}")

    return p5, p6


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str, required=True, help="Path to zero-shot checkpoint")
    p.add_argument("--dataset", type=str, default="german", choices=["german", "urdu"], help="Dataset to evaluate on")
    p.add_argument("--out_dir", type=str, default=None,  help="Directory to save results")
    args = p.parse_args()
    evaluate(args.ckpt, dataset_choice=args.dataset, out_dir=args.out_dir)
