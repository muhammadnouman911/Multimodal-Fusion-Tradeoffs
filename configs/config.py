import os
import logging
import torch
from dataclasses import dataclass, field

CHALLENGE_ROOT = r"O:\POLY-SIM Grand Challenge 2026"
TRAIN_FEATS_DIR = os.path.join(CHALLENGE_ROOT, "Train", "train(1)")
TRAIN_CSV_DIR   = os.path.join(CHALLENGE_ROOT, "Train", "comp")
SUBMIT_FEATS_DIR = os.path.join(CHALLENGE_ROOT, "Dev", "val(1)")
SUBMIT_CSV_DIR   = os.path.join(CHALLENGE_ROOT, "Dev", "comp")

@dataclass
class ExperimentConfig:
    seeds: list = field(default_factory=lambda: [43, 44, 45, 46])
    seed: int = 42 
    
    resume_from: str = os.path.join("checkpoints", "v17_Annealed_English_multibranch_seed42_alpha0.3_ep98_s99.96.pt") 
    audio_backbone: str = "wavlm"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0          

    train_feats_dir: str  = TRAIN_FEATS_DIR
    train_csv_dir: str    = TRAIN_CSV_DIR
    submit_feats_dir: str = SUBMIT_FEATS_DIR
    submit_csv_dir: str   = SUBMIT_CSV_DIR
    
    version: str    = "v17_Overdrive"

    # ── IEEE Paper Mode ───────────────────────────────────────────────────────
    # When True, Urdu is NEVER loaded during training.
    # All training scripts check this flag before loading any Urdu CSV.
    zero_shot_mode: bool = True
    paper_version: str = "v17_ZeroShot_IEEE"

    train_csv_en: str = os.path.join(TRAIN_CSV_DIR, "v1_train_English.csv")
    train_csv_ur: str = os.path.join(TRAIN_CSV_DIR, "v1_train_Urdu.csv")
    val_csv_ur: str   = os.path.join(TRAIN_CSV_DIR, "v1_val_Urdu.csv")

    model_type: str    = "multibranch"   
    fusion: str        = "hybrid" 
    embedding_dim: int = 1024

    # NOTE: These are competition-mode defaults.
    # The IEEE paper ablation experiments (scripts/ablation/run_ablation.py)
    # override these with: lr=1e-3, weight_decay=1e-4, batch_size=64.
    lr: float = 1e-4 
    batch_size: int    = 128
    max_epochs: int = 1000
    weight_decay: float = 5e-2 
    
    feat_noise_std: float = 0.01
    modality_dropout_face: float = 0.4
    modality_dropout_audio: float = 0.1
    mixup_alpha: float = 0.3 

    # Tier 1 & 2 Roadmap Params
    arcface_s: float = 32.0
    arcface_m: float = 0.2
    arcface_k: int = 3
    grl_alpha: float = 0.3
    proxy_loss_weight: float = 0.1
    center_loss_weight: float = 0.01
    
    # Tier 3 & 4 TTA & Calibration
    tta_samples: int = 5
    tta_noise_std: float = 0.01
    temperature: float = 1.0

    # Tier 2 — Item 6: Knowledge Distillation
    teacher_checkpoint: str = None  # Path to your best multimodal checkpoint
    distill_weight: float = 1.0
    distill_temp: float   = 2.0

    alpha: float = 0.3
    label_smoothing: float = 0.1

    early_stop: bool         = False
    early_stop_patience: int = 100
    early_stop_metric: str   = "mean"   
    early_stop_min_delta: float = 0.0

    val_split: float = 0.15
    swa_start_epoch: int = None

    @property
    def log_level(self): return logging.INFO
    @property
    def resolved_num_classes(self) -> int: return 70
    @property
    def seen_lang(self) -> str: return "English"
    @property
    def unseen_lang(self) -> str: return "Urdu"
    def submit_csv(self, lang: str) -> str: return os.path.join(self.submit_csv_dir, f"v1_val_{lang}.csv")
