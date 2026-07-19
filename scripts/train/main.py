"""
POLY-SIM Grand Challenge 2026 — Training Entry Point
=====================================================

Strategy:
  - Train on 85% of the labelled English training data.
  - Evaluate P3/P4 on held-out English val split.
  - Evaluate P5/P6 on full Urdu dataset (same speakers, different lang).
  - Monitor Unseen Mean (P5+P6) as the primary optimization metric.

Usage:
    cd "D:\\POLY-SIM Grand Challenge 2026\\solution"
    python main.py
"""

import logging

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'configs'))

import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import ExperimentConfig
from utils.featLoader import LoadData
from utils.trainer    import Trainer
from utils.evaluator  import Evaluator
from utils.earlystop  import EarlyStopping
from models.fop         import FOP
from models.multibranch import MultiBranchFOP


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logger(config) -> logging.Logger:
    logger = logging.getLogger("POLY-SIM")
    logger.setLevel(config.log_level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger


def save_checkpoint(model, optimizer, config, epoch, metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch":           epoch,
            "metrics":         metrics,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config":          vars(config),
        },
        save_path,
    )


def make_train_loader(config) -> DataLoader:
    # ── IEEE Zero-Shot Mode: English ONLY ─────────────────────────────────
    # Urdu is held out as the UNSEEN test language.
    # Not a single Urdu sample is loaded at any point in training.
    if config.zero_shot_mode:
        ds_en = LoadData(
            csv_paths=config.train_csv_en,
            feat_dir=config.train_feats_dir,
            schema="train",
            lang_label=0
        )

        # Class-balanced sampling: corrects for speaker imbalance
        # (English train has min=2, max=391 samples per speaker)
        import numpy as np
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", ds_en.labels)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            ds_en,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=(config.device == "cuda"),
            drop_last=False,
        )

    # ── Legacy bilingual mode (competition submission only) ────────────────
    # DO NOT use this mode for IEEE paper experiments.
    import numpy as np
    ds_en = LoadData(
        csv_paths=config.train_csv_en,
        feat_dir=config.train_feats_dir,
        schema="train",
        lang_label=0
    )
    ds_ur = LoadData(
        csv_paths=config.train_csv_ur,
        feat_dir=config.train_feats_dir,
        schema="train",
        lang_label=1
    )
    combined_ds = torch.utils.data.ConcatDataset([ds_en, ds_ur])
    weights = [1.0/len(ds_en)] * len(ds_en) + [1.0/len(ds_ur)] * len(ds_ur)
    sampler = WeightedRandomSampler(torch.tensor(weights).float(),
                                    num_samples=len(weights), replacement=True)
    return DataLoader(
        combined_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
    )


def make_val_dataset_seen(config) -> LoadData:
    return LoadData(
        csv_paths=config.train_csv_en,
        feat_dir=config.train_feats_dir,
        schema="train",
        lang_label=0
    )


def make_val_dataset_unseen(config) -> LoadData:
    urdu_csv = os.path.join(
        config.train_csv_dir,
        f"v1_train_{config.unseen_lang}.csv",
    )
    return LoadData(
        csv_paths=urdu_csv,
        feat_dir=config.train_feats_dir,
        schema="train",
        lang_label=1
    )


def build_model(config, face_dim: int, audio_dim: int):
    mtype = config.model_type.lower()
    if mtype == "fop":
        return FOP(config, face_dim=face_dim, audio_dim=audio_dim)
    elif mtype == "multibranch":
        return MultiBranchFOP(config, face_dim=face_dim, audio_dim=audio_dim)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main_seed(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger = setup_logger(config)

    logger.info("=" * 62)
    logger.info("POLY-SIM Grand Challenge 2026 — Training")
    logger.info("=" * 62)
    logger.info("Version=%s | Seen=%s | Unseen=%s | #Classes=%d",
                config.version, config.seen_lang, config.unseen_lang,
                config.resolved_num_classes)
    logger.info("Model=%s | Fusion=%s | Emb=%d | LR=%.4f | Batch=%d",
                config.model_type, config.fusion, config.embedding_dim,
                config.lr, config.batch_size)
    logger.info("Alpha(OPL)=%.3f | ModalityDropout F/A=%.2f/%.2f",
                config.alpha, config.modality_dropout_face, config.modality_dropout_audio)
    logger.info("ArcFace s=%.1f m=%.2f | GRL alpha=%.2f | Proxy=%.2f",
                config.arcface_s, config.arcface_m, config.grl_alpha, config.proxy_loss_weight)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading datasets …")

    train_loader  = make_train_loader(config)
    seen_val_ds   = make_val_dataset_seen(config)
    unseen_val_ds = make_val_dataset_unseen(config)

    logger.info("Train=%d | Seen-val=%d | Unseen-val=%d",
                len(train_loader.dataset), len(seen_val_ds), len(unseen_val_ds))

    # ── Feature dimensions ────────────────────────────────────────────────────
    audio_sample, face_sample, _, _ = next(iter(train_loader))
    audio_dim = audio_sample.shape[1]
    face_dim  = face_sample.shape[1]
    logger.info("Feature dims → audio=%d | face=%d", audio_dim, face_dim)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config, face_dim=face_dim, audio_dim=audio_dim)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model params: %.3f M", n_params)

    # ── Resume / Fine-tune ────────────────────────────────────────────────────
    start_epoch = 0
    # ── Teacher Model (for Distillation) ──────────────────────────────────────
    teacher = None
    if config.teacher_checkpoint and os.path.exists(config.teacher_checkpoint):
        logger.info("Loading Teacher model for distillation: %s", config.teacher_checkpoint)
        teacher = build_model(config, face_dim=face_dim, audio_dim=audio_dim)
        t_ckpt  = torch.load(config.teacher_checkpoint, map_location=config.device)
        teacher.load_state_dict(t_ckpt["model_state"])
        logger.info("Teacher model loaded and set to eval mode.")

    # ── Training components ───────────────────────────────────────────────────
    trainer       = Trainer(model, config, teacher=teacher)
    evaluator     = Evaluator(model, config)
    early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        mode="max",
    )

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Keep track of top 7 checkpoints
    top_checkpoints = []
    best_score = -float("inf")

    logger.info("=" * 62)
    logger.info("Starting training for up to %d epochs …", config.max_epochs)
    logger.info("=" * 62)

    for epoch in range(start_epoch, config.max_epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        train_loss = trainer.train_epoch(
            train_loader, alpha=config.alpha, epoch=epoch
        )

        # ── Evaluate P3–P6 ────────────────────────────────────────────────────
        p_acc = evaluator.p_accuracy(seen_val_ds, unseen_val_ds)

        logger.info(
            "Ep %03d | Loss %.4f | P3=%.1f P4=%.1f P5=%.1f P6=%.1f | Mean=%.2f",
            epoch, train_loss,
            p_acc["P3"], p_acc["P4"], p_acc["P5"], p_acc["P6"], p_acc["mean"],
        )
        
        # ── Step LR Scheduler / SWA ───────────────────────────────────────────
        unseen_mean = (p_acc["P5"] + p_acc["P6"]) / 2
        metric_score = unseen_mean if config.early_stop_metric == "unseen" else p_acc["mean"]
        
        if config.swa_start_epoch and epoch >= config.swa_start_epoch:
            trainer.swa_model.update_parameters(model)
            trainer.swa_scheduler.step()
        else:
            trainer.scheduler.step(metric_score)

        # ── Save top checkpoints ──────────────────────────────────────────────
        current_score = metric_score
        
        if len(top_checkpoints) < 7 or current_score > min([s for s, _, _ in top_checkpoints]):
            if len(top_checkpoints) == 7:
                top_checkpoints.sort(key=lambda x: x[0])
                _, _, worst_path = top_checkpoints.pop(0)
                if os.path.exists(worst_path): os.remove(worst_path)
            
            save_path = os.path.join(
                save_dir,
                f"{config.version}_{config.seen_lang}_{config.model_type}"
                f"_seed{config.seed}_alpha{config.alpha}_ep{epoch}_s{current_score:.2f}.pt",
            )
            save_checkpoint(model, trainer.optimizer, config, epoch, p_acc, save_path)
            top_checkpoints.append((current_score, epoch, save_path))
            
            if current_score > best_score:
                best_score = current_score
                logger.info("  ✓ New absolute best (%s=%.2f) — saved.", config.early_stop_metric, best_score)
            else:
                logger.info("  ✓ Top-7 checkpoint (%s=%.2f) — saved.", config.early_stop_metric, current_score)

        # ── Early stopping ────────────────────────────────────────────────────
        if config.early_stop and early_stopper.step(metric_score):
            logger.info("Early stop at epoch %d (best %s=%.2f).", epoch, config.early_stop_metric, early_stopper.best_score)
            break

    # ── Finalize SWA ──────────────────────────────────────────────────────────
    if config.swa_start_epoch:
        logger.info("Finalizing SWA (updating BN stats)...")
        trainer.swa_model.train()
        with torch.no_grad():
            for audio, face, _ in train_loader:
                audio = audio.to(config.device)
                face  = face.to(config.device)
                trainer.swa_model(face, audio)
        
        trainer.swa_model.eval()
        swa_evaluator = Evaluator(trainer.swa_model, config)
        swa_p_acc = swa_evaluator.p_accuracy(seen_val_ds, unseen_val_ds)
        logger.info(
            "SWA Final | P3=%.1f P4=%.1f P5=%.1f P6=%.1f | Mean=%.2f",
            swa_p_acc["P3"], swa_p_acc["P4"], swa_p_acc["P5"], swa_p_acc["P6"], swa_p_acc["mean"],
        )
        
        swa_save_path = os.path.join(
            save_dir,
            f"{config.version}_{config.seen_lang}_{config.model_type}"
            f"_seed{config.seed}_swa_s{swa_p_acc['mean']:.2f}.pt"
        )
        save_checkpoint(trainer.swa_model, trainer.optimizer, config, epoch, swa_p_acc, swa_save_path)
        logger.info("SWA model saved: %s", swa_save_path)

    logger.info("=" * 62)
    logger.info("Done. Best score = %.2f", best_score)
    logger.info("=" * 62)


def main():
    base_config = ExperimentConfig()
    for seed in base_config.seeds:
        base_config.seed = seed
        # Re-initialize or copy to ensure fresh state
        import copy
        current_cfg = copy.deepcopy(base_config)
        main_seed(current_cfg)

if __name__ == "__main__":
    main()
