"""
FOP Baseline Reproduction (IEEE Paper Section 5.3)
===================================================
Trains the official FOP model (Nawaz et al., ICASSP 2022) on English only.
Uses standard Linear/Gated fusion, standard Cross-Entropy (no ArcFace, no GRL, no SAM).

Usage:
    python train_baseline_fop.py --fusion linear
    python train_baseline_fop.py --fusion gated
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'configs'))

from config import ExperimentConfig
from utils.featLoader import LoadData
from models.fop import FOP

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", default="linear", choices=["linear", "gated"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    return p.parse_args()

def main():
    args = parse_args()
    config = ExperimentConfig()
    config.model_type = "fop"
    config.fusion = args.fusion
    config.seed = args.seed
    config.max_epochs = args.epochs
    config.zero_shot_mode = True
    
    # Disable advanced loss components for FOP baseline
    config.arcface_s = 0.0
    config.grl_alpha = 0.0
    config.proxy_loss_weight = 0.0
    config.center_loss_weight = 0.0
    config.modality_dropout_face = 0.0
    config.modality_dropout_audio = 0.0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = LoadData(config.train_csv_en, config.train_feats_dir, schema="train", lang_label=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.arange(len(ds)), ds.labels))
    train_idx, val_idx = folds[0]

    weights = compute_sample_weight("balanced", ds.labels[train_idx])
    sampler = WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights), True)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=64, sampler=sampler)

    a_s, f_s, _, _ = next(iter(train_loader))
    model = FOP(config, face_dim=f_s.shape[1], audio_dim=a_s.shape[1]).to(config.device)

    import torch.optim as optim
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs("checkpoints/baseline", exist_ok=True)
    best_p4 = 0.0

    for epoch in range(config.max_epochs):
        model.train()
        for audio, face, labels, _ in train_loader:
            audio, face, labels = audio.to(config.device), face.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            out = model(face, audio)
            
            # FOP model forward output details
            if isinstance(out, dict):
                logits = out["fusion_logits"]
            elif isinstance(out, tuple):
                logits = out[1]
            else:
                logits = out
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        dev = config.device
        vf = torch.from_numpy(ds.face_feats[val_idx]).float().to(dev)
        va = torch.from_numpy(ds.audio_feats[val_idx]).float().to(dev)
        vl = torch.from_numpy(ds.labels[val_idx]).long().to(dev)
        
        with torch.no_grad():
            out = model(vf, va)
            logits = out["fusion_logits"] if isinstance(out, dict) else (out[1] if isinstance(out, tuple) else out)
            p3 = (logits.argmax(1) == vl).float().mean().item() * 100
            
            out2 = model(torch.zeros_like(vf), va)
            logits2 = out2["fusion_logits"] if isinstance(out2, dict) else (out2[1] if isinstance(out2, tuple) else out2)
            p4 = (logits2.argmax(1) == vl).float().mean().item() * 100
            
        print(f"FOP-{args.fusion} | Ep {epoch:03d} | P3={p3:.1f}% P4={p4:.1f}%")
        
        if p4 > best_p4:
            best_p4 = p4
            torch.save({
                "model_state": model.state_dict(),
                "fusion": args.fusion,
                "model_type": "fop"
            }, f"checkpoints/baseline/fop_{args.fusion}_seed{args.seed}_best.pt")

    print(f"\nFOP-{args.fusion} Best P4={best_p4:.2f}%")

if __name__ == "__main__":
    main()
