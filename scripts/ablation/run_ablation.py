import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd


import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'configs'))

from config import ExperimentConfig, CHALLENGE_ROOT
from utils.featLoader import LoadData
from models.fop import FOP

# ─── SAM Optimizer ────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021)."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two-step: first_step/second_step")

# ─── SWA Helpers ──────────────────────────────────────────────────────────────
def swa_update(swa_model, model, swa_n):
    """Incrementally update SWA average."""
    for swa_p, p in zip(swa_model.parameters(), model.parameters()):
        swa_p.data = (swa_p.data * swa_n + p.data) / (swa_n + 1)

def swa_bn_update(swa_model, loader, device):
    """Update SWA BatchNorm statistics."""
    swa_model.train()
    with torch.no_grad():
        for audio, face, labels, _ in loader:
            face, audio = face.to(device), audio.to(device)
            swa_model(face, audio)
    swa_model.eval()

# ─── Utilities ────────────────────────────────────────────────────────────────
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics_from_cm(cm):
    cm = np.array(cm)
    total_samples = cm.sum()
    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    precisions, recalls = [], []
    for i in range(len(cm)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    mp = np.mean(precisions); mr = np.mean(recalls)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    return {"accuracy": accuracy * 100, "precision": mp * 100,
            "recall": mr * 100, "f1": f1 * 100, "confusion_matrix": cm.tolist()}

def evaluate_model(model, ds, device, modality="both"):
    model.eval()
    vf = torch.from_numpy(ds.face_feats).float().to(device)
    va = torch.from_numpy(ds.audio_feats).float().to(device)
    vl = torch.from_numpy(ds.labels).long()
    if modality == "audio_only":
        vf = torch.zeros_like(vf)
    with torch.no_grad():
        out = model(vf, va)
        logits = out["fusion_logits"] if isinstance(out, dict) else out
        preds = logits.argmax(dim=1).cpu().numpy()
    cm = np.zeros((70, 70), dtype=np.int64)
    for t, p in zip(vl.numpy(), preds):
        cm[t, p] += 1
    return calculate_metrics_from_cm(cm)

# ─── Core Training Function ───────────────────────────────────────────────────
def train_ablation(name, grl_alpha, dropout_face, dropout_audio, epochs, seed, device,
                   train_loader, ds_val_en, ds_ur, face_dim, audio_dim,
                   fusion="linear", use_sam=False, use_swa=False, swa_start=100):
    """Train a single ablation config with optional SAM/SWA and fusion type."""
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING: {name}", flush=True)
    print(f"  GRL: {grl_alpha} | Dropout Face: {dropout_face} | Audio: {dropout_audio}", flush=True)
    print(f"  Fusion: {fusion} | SAM: {use_sam} | SWA: {use_swa} (start={swa_start})", flush=True)
    print(f"{'='*60}", flush=True)

    set_seeds(seed)

    config = ExperimentConfig()
    config.seed = seed
    config.max_epochs = epochs
    config.grl_alpha = grl_alpha
    config.modality_dropout_face = dropout_face
    config.modality_dropout_audio = dropout_audio
    config.device = device
    config.fusion = fusion

    model = FOP(config, face_dim=face_dim, audio_dim=audio_dim).to(device)

    # ── Optimizer setup ──────────────────────────────────────────────────────
    if use_sam:
        base_opt = lambda params, **kw: optim.AdamW(params, **kw)
        optimizer = SAM(model.parameters(), base_opt, rho=0.05, lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # ── SWA model (shadow copy) ──────────────────────────────────────────────
    swa_model = None
    swa_n = 0
    if use_swa:
        import copy
        swa_model = copy.deepcopy(model).to(device)
        for p in swa_model.parameters():
            p.data.zero_()

    best_p4 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # Anneal GRL alpha (linear 0→alpha over epochs 10–30)
        current_grl_alpha = 0.0
        if epoch >= 10:
            anneal_factor = min(1.0, (epoch - 10) / 20.0)
            current_grl_alpha = grl_alpha * anneal_factor
        if hasattr(model, 'grl_alpha'):
            model.grl_alpha = current_grl_alpha
            if hasattr(model, 'grl'):
                model.grl.alpha = current_grl_alpha

        epoch_loss = 0.0
        n_batches = 0
        for audio, face, labels, lang_labels in train_loader:
            audio, face = audio.to(device), face.to(device)
            labels, lang_labels = labels.to(device), lang_labels.to(device)

            if use_sam:
                # SAM first step
                out = model(face, audio)
                loss_id = criterion(out["fusion_logits"], labels)
                loss_lang = criterion(out["lang_logits"], lang_labels) if out["lang_logits"] is not None else 0.0
                loss = loss_id + (current_grl_alpha * loss_lang if isinstance(loss_lang, torch.Tensor) else 0.0)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # SAM second step
                out2 = model(face, audio)
                loss_id2 = criterion(out2["fusion_logits"], labels)
                loss_lang2 = criterion(out2["lang_logits"], lang_labels) if out2["lang_logits"] is not None else 0.0
                loss2 = loss_id2 + (current_grl_alpha * loss_lang2 if isinstance(loss_lang2, torch.Tensor) else 0.0)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
                epoch_loss += loss.item()
            else:
                optimizer.zero_grad()
                out = model(face, audio)
                loss_id = criterion(out["fusion_logits"], labels)
                loss_lang = criterion(out["lang_logits"], lang_labels) if out["lang_logits"] is not None else None
                loss = loss_id + (current_grl_alpha * loss_lang if loss_lang is not None else 0.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            n_batches += 1

        # SWA update after swa_start
        if use_swa and swa_model is not None and epoch >= swa_start:
            swa_update(swa_model, model, swa_n)
            swa_n += 1

        # Validation (use main model during training)
        val_res = evaluate_model(model, ds_val_en, device, modality="audio_only")
        p4 = val_res["accuracy"]

        if p4 > best_p4:
            best_p4 = p4
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val P4: {p4:.2f}% (Best: {best_p4:.2f}%)", flush=True)

    # ── Final evaluation ─────────────────────────────────────────────────────
    # For SWA: update BN stats then evaluate SWA model
    if use_swa and swa_model is not None and swa_n > 0:
        print(f"  Updating SWA BatchNorm statistics...", flush=True)
        swa_bn_update(swa_model, train_loader, device)
        eval_model = swa_model
        print(f"  Evaluating SWA model (averaged over {swa_n} checkpoints)", flush=True)
    else:
        # Load best validation checkpoint
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        eval_model = model

    print(f"  Running final P3/P4/P5/P6 evaluations...", flush=True)
    p3_metrics = evaluate_model(eval_model, ds_val_en, device, modality="both")
    p4_metrics = evaluate_model(eval_model, ds_val_en, device, modality="audio_only")
    p5_metrics = evaluate_model(eval_model, ds_ur,    device, modality="both")
    p6_metrics = evaluate_model(eval_model, ds_ur,    device, modality="audio_only")

    lang_gap_p4p5 = p4_metrics["accuracy"] - p5_metrics["accuracy"]   # LG_modality
    lang_gap_p3p5 = p3_metrics["accuracy"] - p5_metrics["accuracy"]   # LG_language (cleaner)
    mod_gap       = p5_metrics["accuracy"] - p6_metrics["accuracy"]

    print(f"  P3={p3_metrics['accuracy']:.2f}% P4={p4_metrics['accuracy']:.2f}% "
          f"P5={p5_metrics['accuracy']:.2f}% P6={p6_metrics['accuracy']:.2f}%", flush=True)
    print(f"  LG(P4-P5)={lang_gap_p4p5:.2f}% | LG(P3-P5)={lang_gap_p3p5:.2f}% | ModGap={mod_gap:.2f}%", flush=True)

    return {
        "P3": p3_metrics, "P4": p4_metrics,
        "P5": p5_metrics, "P6": p6_metrics,
        "language_gap_p4p5": lang_gap_p4p5,
        "language_gap_p3p5": lang_gap_p3p5,
        "modality_gap": mod_gap
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold",   type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--run",    type=str, default="all",
                   help="Which experiments to run: all | extended | d_only | prime_only")
    args = p.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "results/ablation_study"
    os.makedirs(out_dir, exist_ok=True)

    config = ExperimentConfig()

    # ── Load data once ───────────────────────────────────────────────────────
    print("Loading English dataset...", flush=True)
    ds_en = LoadData(config.train_csv_en, config.train_feats_dir, schema="train", lang_label=0)
    print(f"  English: {len(ds_en)} samples loaded", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = list(skf.split(np.arange(len(ds_en)), ds_en.labels))[args.fold]

    weights = compute_sample_weight("balanced", ds_en.labels[train_idx])
    sampler = WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights), True)
    train_loader = DataLoader(Subset(ds_en, train_idx), batch_size=64, sampler=sampler)

    class ValSubset: pass
    ds_val_en = ValSubset()
    ds_val_en.face_feats  = ds_en.face_feats[val_idx]
    ds_val_en.audio_feats = ds_en.audio_feats[val_idx]
    ds_val_en.labels      = ds_en.labels[val_idx]

    print("Loading Urdu dataset...", flush=True)
    ds_ur = LoadData(config.train_csv_ur, os.path.join(CHALLENGE_ROOT, "Train", "train(1)"),
                     schema="train", lang_label=1)
    en_df  = pd.read_csv(config.train_csv_en)
    identity_to_label = dict(zip(en_df["identity"], en_df["label"]))
    ur_df  = pd.read_csv(config.train_csv_ur)
    mapped_labels = [identity_to_label[uid] for uid in ur_df["identity"] if uid in identity_to_label]
    valid_idx_ur  = [i for i, uid in enumerate(ur_df["identity"]) if uid in identity_to_label]
    ds_ur.face_feats  = ds_ur.face_feats[valid_idx_ur]
    ds_ur.audio_feats = ds_ur.audio_feats[valid_idx_ur]
    ds_ur.labels      = np.array(mapped_labels, dtype=np.int64)
    print(f"  Urdu: {len(ds_ur.labels)} samples loaded", flush=True)

    a_s, f_s, _, _ = next(iter(train_loader))
    face_dim  = f_s.shape[1]
    audio_dim = a_s.shape[1]
    print(f"  Face dim: {face_dim} | Audio dim: {audio_dim}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DATA LOADED. Run mode: {args.run}", flush=True)
    print(f"{'='*60}", flush=True)

    common = dict(epochs=args.epochs, seed=args.seed, device=device,
                  train_loader=train_loader, ds_val_en=ds_val_en, ds_ur=ds_ur,
                  face_dim=face_dim, audio_dim=audio_dim)

    all_results = {}

    # ── Load prior results if they exist ────────────────────────────────────
    prior_path = os.path.join(out_dir, "ablation_results.json")
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} prior results from {prior_path}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT D  — FOP_MAV + MultiBranch Fusion ONLY
    #   Controls: No GRL, No Dropout, No SAM, No SWA
    #   Only variable vs. baseline: fusion = "hybrid"
    # ════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "extended", "d_only"):
        res_d = train_ablation(
            "Exp D: FOP_MAV + MultiBranch Fusion",
            grl_alpha=0.0, dropout_face=0.0, dropout_audio=0.0,
            fusion="hybrid", use_sam=False, use_swa=False, **common
        )
        all_results["Experiment_D_MultiBranchFusion"] = res_d
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  [Saved Exp D results]", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A'  — FOP_MAV + Modality Dropout + SAM + SWA
    #   Resolves optimizer confound for Experiment A
    # ════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "extended", "prime_only"):
        res_a_prime = train_ablation(
            "Exp A': FOP_MAV + Dropout + SAM/SWA",
            grl_alpha=0.0, dropout_face=0.4, dropout_audio=0.1,
            fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
        )
        all_results["Experiment_A_prime_Dropout_SAM_SWA"] = res_a_prime
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  [Saved Exp A' results]", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B'  — FOP_MAV + GRL + SAM + SWA
    #   Resolves optimizer confound for Experiment B
    # ════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "extended", "prime_only"):
        res_b_prime = train_ablation(
            "Exp B': FOP_MAV + GRL + SAM/SWA",
            grl_alpha=0.3, dropout_face=0.0, dropout_audio=0.0,
            fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
        )
        all_results["Experiment_B_prime_GRL_SAM_SWA"] = res_b_prime
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  [Saved Exp B' results]", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C'  — FOP_MAV + GRL + Dropout + SAM + SWA
    #   Resolves optimizer confound for Experiment C
    # ════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "extended", "prime_only"):
        res_c_prime = train_ablation(
            "Exp C': FOP_MAV + GRL + Dropout + SAM/SWA",
            grl_alpha=0.3, dropout_face=0.4, dropout_audio=0.1,
            fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
        )
        all_results["Experiment_C_prime_GRL_Dropout_SAM_SWA"] = res_c_prime
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  [Saved Exp C' results]", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT D'  — FOP_MAV + MultiBranch Fusion + SAM + SWA
    #   Resolves optimizer confound for Experiment D
    # ════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "extended", "prime_only"):
        res_d_prime = train_ablation(
            "Exp D': FOP_MAV + MultiBranch Fusion + SAM/SWA",
            grl_alpha=0.0, dropout_face=0.0, dropout_audio=0.0,
            fusion="hybrid", use_sam=True, use_swa=True, swa_start=100, **common
        )
        all_results["Experiment_D_prime_MultiBranchFusion_SAM_SWA"] = res_d_prime
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  [Saved Exp D' results]", flush=True)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("EXTENDED ABLATION STUDY COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Results saved to {out_dir}/ablation_results.json", flush=True)

    print(f"\n{'Config':<45} {'P3':>7} {'P4':>7} {'P5':>7} {'P6':>7} "
          f"{'LG(P4-P5)':>10} {'LG(P3-P5)':>10} {'ModGap':>8}", flush=True)
    print("-" * 110, flush=True)
    for exp_name, res in all_results.items():
        lg_p4p5 = res.get("language_gap_p4p5", res.get("language_gap", float("nan")))
        lg_p3p5 = res.get("language_gap_p3p5", float("nan"))
        print(
            f"{exp_name:<45} "
            f"{res['P3']['accuracy']:>7.2f} {res['P4']['accuracy']:>7.2f} "
            f"{res['P5']['accuracy']:>7.2f} {res['P6']['accuracy']:>7.2f} "
            f"{lg_p4p5:>10.2f} {lg_p3p5:>10.2f} {res['modality_gap']:>8.2f}",
            flush=True
        )

if __name__ == "__main__":
    main()
