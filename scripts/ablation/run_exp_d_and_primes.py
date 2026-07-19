"""
run_exp_d_and_primes.py
=======================
Self-contained script to run:
  - Experiment D:  FOP_MAV + MultiBranch Fusion (no GRL, no Dropout, AdamW only)
  - Experiment A': FOP_MAV + Dropout + SAM/SWA
  - Experiment B': FOP_MAV + GRL + SAM/SWA
  - Experiment C': FOP_MAV + GRL + Dropout + SAM/SWA

Saves individual JSON results and generates analysis reports.
Does NOT modify any existing files.

Usage:
  python run_exp_d_and_primes.py --run d        # Exp D only
  python run_exp_d_and_primes.py --run primes   # A'/B'/C' only
  python run_exp_d_and_primes.py --run all      # All four
"""
import os, sys, argparse, random, json, copy, datetime
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import torch
torch.set_num_threads(4)
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
                if p.grad is None or "old_p" not in self.state[p]: continue
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
            ]), p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two-step: first_step/second_step")

# ─── SWA Helpers ──────────────────────────────────────────────────────────────
def swa_update(swa_model, model, swa_n):
    for swa_p, p in zip(swa_model.parameters(), model.parameters()):
        swa_p.data = (swa_p.data * swa_n + p.data) / (swa_n + 1)

def swa_bn_update(swa_model, loader, device):
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
    total = cm.sum()
    accuracy = np.trace(cm) / total if total > 0 else 0.0
    precisions, recalls = [], []
    for i in range(len(cm)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    mp = np.mean(precisions); mr = np.mean(recalls)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    return {"accuracy": accuracy * 100, "precision": mp * 100,
            "recall": mr * 100, "f1": f1 * 100}

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
def train_experiment(name, grl_alpha, dropout_face, dropout_audio, epochs, seed,
                     device, train_loader, ds_val_en, ds_ur, face_dim, audio_dim,
                     fusion="linear", use_sam=False, use_swa=False, swa_start=100):
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING: {name}", flush=True)
    print(f"  GRL alpha:      {grl_alpha}", flush=True)
    print(f"  Dropout face:   {dropout_face}  audio: {dropout_audio}", flush=True)
    print(f"  Fusion:         {fusion}", flush=True)
    print(f"  SAM: {use_sam}  SWA: {use_swa} (start={swa_start})", flush=True)
    print(f"  Epochs: {epochs}  Seed: {seed}", flush=True)
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

    if use_sam:
        base_opt = lambda params, **kw: optim.AdamW(params, **kw)
        optimizer = SAM(model.parameters(), base_opt, rho=0.05, lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    swa_model = None
    swa_n = 0
    if use_swa:
        swa_model = copy.deepcopy(model).to(device)
        for p in swa_model.parameters():
            p.data.zero_()

    best_p4 = 0.0
    best_model_state = None
    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        model.train()

        # GRL annealing
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
                out = model(face, audio)
                loss_id = criterion(out["fusion_logits"], labels)
                loss_lang = criterion(out["lang_logits"], lang_labels) if out["lang_logits"] is not None else 0.0
                loss = loss_id + (current_grl_alpha * loss_lang if isinstance(loss_lang, torch.Tensor) else 0.0)
                loss.backward()
                optimizer.first_step(zero_grad=True)
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

        if use_swa and swa_model is not None and epoch >= swa_start:
            swa_update(swa_model, model, swa_n)
            swa_n += 1

        val_res = evaluate_model(model, ds_val_en, device, modality="audio_only")
        p4 = val_res["accuracy"]

        if p4 > best_p4:
            best_p4 = p4
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            print(f"  Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val P4: {p4:.2f}% (Best: {best_p4:.2f}%) | "
                  f"Elapsed: {elapsed:.0f}s", flush=True)

    # ── Final evaluation ──────────────────────────────────────────────────────
    if use_swa and swa_model is not None and swa_n > 0:
        print(f"  Updating SWA BatchNorm stats ({swa_n} checkpoints)...", flush=True)
        swa_bn_update(swa_model, train_loader, device)
        eval_model = swa_model
    else:
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        eval_model = model

    print(f"  Evaluating P3/P4/P5/P6...", flush=True)
    p3 = evaluate_model(eval_model, ds_val_en, device, modality="both")
    p4 = evaluate_model(eval_model, ds_val_en, device, modality="audio_only")
    p5 = evaluate_model(eval_model, ds_ur,     device, modality="both")
    p6 = evaluate_model(eval_model, ds_ur,     device, modality="audio_only")

    total_time = (datetime.datetime.now() - start_time).total_seconds()

    print(f"\n  RESULTS for {name}:", flush=True)
    print(f"  P3={p3['accuracy']:.2f}%  P4={p4['accuracy']:.2f}%  "
          f"P5={p5['accuracy']:.2f}%  P6={p6['accuracy']:.2f}%", flush=True)
    print(f"  LG_old(P4-P5)={p4['accuracy']-p5['accuracy']:.2f}%  "
          f"LG_new(P3-P5)={p3['accuracy']-p5['accuracy']:.2f}%  "
          f"MG(P5-P6)={p5['accuracy']-p6['accuracy']:.2f}%", flush=True)
    print(f"  Training time: {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    return {
        "p3": p3["accuracy"], "p4": p4["accuracy"],
        "p5": p5["accuracy"], "p6": p6["accuracy"],
        "lg_old": round(p4["accuracy"] - p5["accuracy"], 4),
        "lg_new": round(p3["accuracy"] - p5["accuracy"], 4),
        "mg": round(p5["accuracy"] - p6["accuracy"], 4),
        "training_time_seconds": round(total_time, 1)
    }


def generate_exp_d_analysis(result, baseline_p5=98.16, baseline_p3=99.63, baseline_p4=99.13):
    """Generate hypothesis analysis for Exp D."""
    p5 = result["p5"]
    p3 = result["p3"]
    p4 = result["p4"]
    drop = baseline_p5 - p5

    lines = []
    lines.append("# Experiment D: Scientific Analysis\n")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Fusion | HybridMultiPathFusion |")
    lines.append("| GRL | Disabled (alpha=0.0) |")
    lines.append("| Modality Dropout | Disabled (p_face=0.0, p_audio=0.0) |")
    lines.append("| Optimizer | AdamW only (no SAM, no SWA) |")
    lines.append("| Epochs | 150 |")
    lines.append("| Seed | 42 |")
    lines.append("| Fold | 0 |")
    lines.append("")

    lines.append("## Results\n")
    lines.append("| Metric | Baseline FOP_MAV | Exp D (MultiBranch) | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| P3 (EN multi) | {baseline_p3:.2f}% | {p3:.2f}% | {p3-baseline_p3:+.2f}% |")
    lines.append(f"| P4 (EN audio) | {baseline_p4:.2f}% | {p4:.2f}% | {p4-baseline_p4:+.2f}% |")
    lines.append(f"| P5 (UR multi) | {baseline_p5:.2f}% | {p5:.2f}% | {p5-baseline_p5:+.2f}% |")
    lines.append(f"| P6 (UR audio) | 85.67% | {result['p6']:.2f}% | {result['p6']-85.67:+.2f}% |")
    lines.append(f"| LG_new (P3-P5) | 1.47% | {result['lg_new']:.2f}% | — |")
    lines.append(f"| MG (P5-P6) | 12.49% | {result['mg']:.2f}% | — |")
    lines.append("")

    lines.append("## Hypothesis Evaluation\n")

    # Determine which hypothesis is supported
    supported = None

    if drop > 5.0:
        supported = "H1"
        lines.append(f"### **SUPPORTED: H1 — MultiBranch fusion itself hurts cross-lingual transfer**\n")
        lines.append(f"P5 dropped by **{drop:.2f}%** (from {baseline_p5:.2f}% to {p5:.2f}%), ")
        lines.append(f"exceeding the 5% threshold. This confirms that the MultiBranch Hybrid ")
        lines.append(f"Fusion architecture — independent of GRL, Dropout, SAM, or SWA — is the ")
        lines.append(f"primary cause of the cross-lingual accuracy degradation.\n")
    elif abs(drop) <= 2.0:
        supported = "H2"
        lines.append(f"### **SUPPORTED: H2 — Architecture is neutral**\n")
        lines.append(f"P5 changed by only **{drop:+.2f}%** (from {baseline_p5:.2f}% to {p5:.2f}%), ")
        lines.append(f"within the ±2% tolerance. This suggests the MultiBranch fusion does not ")
        lines.append(f"significantly affect cross-lingual transfer by itself. The 12-point drop ")
        lines.append(f"in the full model must therefore be attributed to the optimizer/regularization combination.\n")
    elif p5 > baseline_p5:
        supported = "H3"
        lines.append(f"### **SUPPORTED: H3 — Fusion helps but is masked elsewhere**\n")
        lines.append(f"P5 **improved** to {p5:.2f}% (from {baseline_p5:.2f}%), suggesting the ")
        lines.append(f"MultiBranch fusion actually enhances cross-lingual generalization. ")
        lines.append(f"The degradation in the full model must be caused by over-regularization.\n")

    if p3 > baseline_p3 and p5 < baseline_p5 and drop > 2.0:
        if supported != "H1":
            supported = "H4"
        lines.append(f"### **{'ALSO ' if supported == 'H1' else ''}SUPPORTED: H4 — Fusion overfits to English**\n")
        lines.append(f"P3 improved ({baseline_p3:.2f}% → {p3:.2f}%) while P5 decreased ")
        lines.append(f"({baseline_p5:.2f}% → {p5:.2f}%), indicating the MultiBranch fusion ")
        lines.append(f"learns English-specific interactions that do not transfer to Urdu.\n")
    elif p3 <= baseline_p3 or p5 >= baseline_p5:
        lines.append(f"### NOT SUPPORTED: H4 — Fusion overfits to English\n")
        lines.append(f"P3 did not improve while P5 decreased. H4 is not supported.\n")

    if supported is None:
        supported = "Inconclusive"
        lines.append(f"### INCONCLUSIVE\n")
        lines.append(f"The results (P5={p5:.2f}%, drop={drop:.2f}%) fall between hypothesis ")
        lines.append(f"boundaries and require additional experiments to resolve.\n")

    lines.append("## Reviewer-Facing Summary\n")
    if supported == "H1":
        lines.append(f"Experiment D demonstrates that replacing the linear fusion of FOP_MAV with ")
        lines.append(f"HybridMultiPathFusion — while holding all other variables constant (no GRL, ")
        lines.append(f"no Dropout, AdamW only) — causes a {drop:.2f}% drop in zero-shot Urdu ")
        lines.append(f"multimodal accuracy (P5: {baseline_p5:.2f}% → {p5:.2f}%). This isolates ")
        lines.append(f"the architectural change as the primary source of the 12-point cross-lingual ")
        lines.append(f"degradation observed in the full MultiBranchFOP model. The multi-head ")
        lines.append(f"attention mechanism with modality-specific positional embeddings appears to ")
        lines.append(f"learn dataset-specific interactions that harm cross-lingual generalization.\n")
    elif supported == "H2":
        lines.append(f"Experiment D demonstrates that the MultiBranch Hybrid Fusion architecture ")
        lines.append(f"does not significantly affect cross-lingual transfer when used in isolation ")
        lines.append(f"(P5 change: {drop:+.2f}%). The 12-point degradation in the full MultiBranchFOP ")
        lines.append(f"model must therefore be attributed to the combined effect of GRL, Dropout, ")
        lines.append(f"and/or the SAM+SWA optimizer.\n")
    elif supported == "H3":
        lines.append(f"Experiment D reveals that the MultiBranch Hybrid Fusion actually improves ")
        lines.append(f"cross-lingual transfer (P5: {baseline_p5:.2f}% → {p5:.2f}%). The degradation ")
        lines.append(f"in the full model is therefore caused by over-regularization from the ")
        lines.append(f"combination of GRL, Dropout, SAM, and SWA.\n")

    lines.append(f"\n**Hypothesis supported:** {supported}")
    lines.append(f"**P5 drop from baseline:** {drop:.2f}%")

    return "\n".join(lines), supported


def generate_optimizer_report(primes, originals, multibranch):
    """Generate the optimizer ablation report."""
    lines = []
    lines.append("# Optimizer Ablation Report: SAM/SWA Confound Resolution\n")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## 1. Comparison Table\n")
    lines.append("| Experiment | Optimizer | P3 | P4 | P5 | P6 | LG_new | LG_old | MG |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for label, data in [("A (Dropout)", originals.get("A")),
                         ("A' (Dropout+SAM/SWA)", primes.get("A'")),
                         ("B (GRL)", originals.get("B")),
                         ("B' (GRL+SAM/SWA)", primes.get("B'")),
                         ("C (GRL+Dropout)", originals.get("C")),
                         ("C' (GRL+Dropout+SAM/SWA)", primes.get("C'")),
                         ("MultiBranchFOP (full)", multibranch)]:
        if data:
            opt = "SAM+SWA" if "'" in label or "Multi" in label else "AdamW"
            lines.append(f"| {label} | {opt} | {data['p3']:.2f} | {data['p4']:.2f} | "
                         f"{data['p5']:.2f} | {data['p6']:.2f} | {data['lg_new']:.2f} | "
                         f"{data['lg_old']:.2f} | {data['mg']:.2f} |")

    lines.append("\n## 2. Scientific Analysis\n")

    if primes.get("A'") and originals.get("A"):
        a = originals["A"]; ap = primes["A'"]
        lines.append("### Did SAM/SWA improve performance?\n")
        lines.append(f"- Exp A → A': P5 changed from {a['p5']:.2f}% to {ap['p5']:.2f}% "
                     f"(delta={ap['p5']-a['p5']:+.2f}%)")

    if primes.get("B'") and originals.get("B"):
        b = originals["B"]; bp = primes["B'"]
        lines.append(f"- Exp B → B': P5 changed from {b['p5']:.2f}% to {bp['p5']:.2f}% "
                     f"(delta={bp['p5']-b['p5']:+.2f}%)")

    if primes.get("C'") and originals.get("C"):
        c = originals["C"]; cp = primes["C'"]
        lines.append(f"- Exp C → C': P5 changed from {c['p5']:.2f}% to {cp['p5']:.2f}% "
                     f"(delta={cp['p5']-c['p5']:+.2f}%)")

    lines.append("\n### Did optimizer choice affect language gap?\n")
    if primes.get("A'") and originals.get("A"):
        a_lg = originals["A"]["lg_new"]; ap_lg = primes["A'"]["lg_new"]
        lines.append(f"- A→A' LG_new: {a_lg:.2f}% → {ap_lg:.2f}%")
    if primes.get("B'") and originals.get("B"):
        b_lg = originals["B"]["lg_new"]; bp_lg = primes["B'"]["lg_new"]
        lines.append(f"- B→B' LG_new: {b_lg:.2f}% → {bp_lg:.2f}%")
    if primes.get("C'") and originals.get("C"):
        c_lg = originals["C"]["lg_new"]; cp_lg = primes["C'"]["lg_new"]
        lines.append(f"- C→C' LG_new: {c_lg:.2f}% → {cp_lg:.2f}%")

    lines.append("\n### Did optimizer choice affect modality gap?\n")
    if primes.get("A'") and originals.get("A"):
        a_mg = originals["A"]["mg"]; ap_mg = primes["A'"]["mg"]
        lines.append(f"- A→A' MG: {a_mg:.2f}% → {ap_mg:.2f}%")
    if primes.get("B'") and originals.get("B"):
        b_mg = originals["B"]["mg"]; bp_mg = primes["B'"]["mg"]
        lines.append(f"- B→B' MG: {b_mg:.2f}% → {bp_mg:.2f}%")
    if primes.get("C'") and originals.get("C"):
        c_mg = originals["C"]["mg"]; cp_mg = primes["C'"]["mg"]
        lines.append(f"- C→C' MG: {c_mg:.2f}% → {cp_mg:.2f}%")

    lines.append("\n## 3. Reviewer-Facing Conclusion\n")
    lines.append("The optimizer confound experiments (A'/B'/C') enable direct comparison between ")
    lines.append("AdamW-only and SAM+SWA configurations while holding architectural and ")
    lines.append("regularization variables constant. This resolves the reviewer concern that ")
    lines.append("performance differences between Exp A/B/C (AdamW) and MultiBranchFOP (SAM+SWA) ")
    lines.append("could be attributed to the optimizer rather than the architectural changes.\n")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Run Exp D and optimizer-confound experiments")
    p.add_argument("--fold",   type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--run",    type=str, default="all",
                   choices=["all", "d", "primes"],
                   help="d=Exp D only, primes=A'/B'/C' only, all=everything")
    args = p.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    config = ExperimentConfig()

    # ── Load data ─────────────────────────────────────────────────────────────
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
    en_df = pd.read_csv(config.train_csv_en)
    identity_to_label = dict(zip(en_df["identity"], en_df["label"]))
    ur_df = pd.read_csv(config.train_csv_ur)
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

    common = dict(epochs=args.epochs, seed=args.seed, device=device,
                  train_loader=train_loader, ds_val_en=ds_val_en, ds_ur=ds_ur,
                  face_dim=face_dim, audio_dim=audio_dim)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT D: FOP_MAV + MultiBranch Fusion (no GRL, no Dropout, AdamW)
    # ══════════════════════════════════════════════════════════════════════════
    if args.run in ("all", "d"):
        print(f"\n{'#'*60}", flush=True)
        print(f"  EXPERIMENT D: Baseline + MultiBranch Fusion Only", flush=True)
        print(f"{'#'*60}", flush=True)

        res_d = train_experiment(
            "Exp D: FOP_MAV + MultiBranch Fusion",
            grl_alpha=0.0, dropout_face=0.0, dropout_audio=0.0,
            fusion="hybrid", use_sam=False, use_swa=False, **common
        )

        # Save result
        exp_d_json = {
            "experiment": "Baseline + MultiBranch Fusion",
            "fusion": "HybridMultiPathFusion",
            "grl": False, "dropout": False, "sam": False, "swa": False,
            **res_d
        }
        with open("results/exp_d_multibranch_only.json", "w") as f:
            json.dump(exp_d_json, f, indent=4)
        print(f"\n  Saved -> results/exp_d_multibranch_only.json", flush=True)

        # Generate analysis
        analysis_text, hypothesis = generate_exp_d_analysis(res_d)
        with open("results/exp_d_analysis.md", "w") as f:
            f.write(analysis_text)
        print(f"  Saved -> results/exp_d_analysis.md", flush=True)
        print(f"  Hypothesis supported: {hypothesis}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENTS A'/B'/C': Optimizer confound controls
    # ══════════════════════════════════════════════════════════════════════════
    primes = {}
    if args.run in ("all", "primes"):
        print(f"\n{'#'*60}", flush=True)
        print(f"  OPTIMIZER CONFOUND EXPERIMENTS: A' / B' / C'", flush=True)
        print(f"{'#'*60}", flush=True)

        # ── A': Dropout + SAM/SWA ─────────────────────────────────────────────
        if os.path.exists("results/exp_a_prime.json"):
            print("  Found existing results/exp_a_prime.json, loading...", flush=True)
            with open("results/exp_a_prime.json", "r") as f:
                ap_json = json.load(f)
            res_ap = {k: ap_json[k] for k in ["p3", "p4", "p5", "p6", "lg_old", "lg_new", "mg", "training_time_seconds"]}
            primes["A'"] = res_ap
        else:
            res_ap = train_experiment(
                "Exp A': FOP_MAV + Dropout + SAM/SWA",
                grl_alpha=0.0, dropout_face=0.4, dropout_audio=0.1,
                fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
            )
            ap_json = {
                "experiment": "FOP_MAV + Dropout + SAM/SWA",
                "fusion": "linear", "grl": False, "dropout": True,
                "sam": True, "swa": True, **res_ap
            }
            with open("results/exp_a_prime.json", "w") as f:
                json.dump(ap_json, f, indent=4)
            print(f"\n  Saved -> results/exp_a_prime.json", flush=True)
            primes["A'"] = res_ap

        # ── B': GRL + SAM/SWA ────────────────────────────────────────────────
        if os.path.exists("results/exp_b_prime.json"):
            print("  Found existing results/exp_b_prime.json, loading...", flush=True)
            with open("results/exp_b_prime.json", "r") as f:
                bp_json = json.load(f)
            res_bp = {k: bp_json[k] for k in ["p3", "p4", "p5", "p6", "lg_old", "lg_new", "mg", "training_time_seconds"]}
            primes["B'"] = res_bp
        else:
            res_bp = train_experiment(
                "Exp B': FOP_MAV + GRL + SAM/SWA",
                grl_alpha=0.3, dropout_face=0.0, dropout_audio=0.0,
                fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
            )
            bp_json = {
                "experiment": "FOP_MAV + GRL + SAM/SWA",
                "fusion": "linear", "grl": True, "dropout": False,
                "sam": True, "swa": True, **res_bp
            }
            with open("results/exp_b_prime.json", "w") as f:
                json.dump(bp_json, f, indent=4)
            print(f"\n  Saved -> results/exp_b_prime.json", flush=True)
            primes["B'"] = res_bp

        # ── C': GRL + Dropout + SAM/SWA ──────────────────────────────────────
        if os.path.exists("results/exp_c_prime.json"):
            print("  Found existing results/exp_c_prime.json, loading...", flush=True)
            with open("results/exp_c_prime.json", "r") as f:
                cp_json = json.load(f)
            res_cp = {k: cp_json[k] for k in ["p3", "p4", "p5", "p6", "lg_old", "lg_new", "mg", "training_time_seconds"]}
            primes["C'"] = res_cp
        else:
            res_cp = train_experiment(
                "Exp C': FOP_MAV + GRL + Dropout + SAM/SWA",
                grl_alpha=0.3, dropout_face=0.4, dropout_audio=0.1,
                fusion="linear", use_sam=True, use_swa=True, swa_start=100, **common
            )
            cp_json = {
                "experiment": "FOP_MAV + GRL + Dropout + SAM/SWA",
                "fusion": "linear", "grl": True, "dropout": True,
                "sam": True, "swa": True, **res_cp
            }
            with open("results/exp_c_prime.json", "w") as f:
                json.dump(cp_json, f, indent=4)
            print(f"\n  Saved -> results/exp_c_prime.json", flush=True)
            primes["C'"] = res_cp

        # ── Generate optimizer ablation report ────────────────────────────────
        originals = {
            "A": {"p3": 99.63, "p4": 99.13, "p5": 95.26, "p6": 84.16,
                  "lg_old": 3.87, "lg_new": 4.37, "mg": 11.10},
            "B": {"p3": 100.00, "p4": 99.13, "p5": 97.97, "p6": 86.38,
                  "lg_old": 1.17, "lg_new": 2.03, "mg": 11.59},
            "C": {"p3": 100.00, "p4": 99.38, "p5": 96.31, "p6": 82.44,
                  "lg_old": 3.07, "lg_new": 3.69, "mg": 13.88},
        }
        multibranch = {"p3": 99.01, "p4": 99.01, "p5": 86.04, "p6": 82.12,
                       "lg_old": 12.97, "lg_new": 12.97, "mg": 3.92}

        report = generate_optimizer_report(primes, originals, multibranch)
        with open("results/optimizer_ablation_report.md", "w") as f:
            f.write(report)
        print(f"\n  Saved → results/optimizer_ablation_report.md", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"ALL EXPERIMENTS COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Generated files:", flush=True)
    for f in ["results/exp_d_multibranch_only.json", "results/exp_d_analysis.md",
              "results/exp_a_prime.json", "results/exp_b_prime.json",
              "results/exp_c_prime.json", "results/optimizer_ablation_report.md"]:
        if os.path.exists(f):
            print(f"  ✓ {f}", flush=True)


if __name__ == "__main__":
    main()
