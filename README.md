# When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification

## Overview
This repository contains the official PyTorch implementation and reproducible evaluation scripts for the paper:  
**"When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification"**

We investigate the zero-shot cross-lingual transferability of multimodal (audio-visual) speaker identification systems. Specifically, we demonstrate that while sophisticated cross-attention mechanisms (like MultiBranch Hybrid Fusion) improve within-language missing-modality robustness, they severely degrade cross-lingual generalization compared to simple linear fusion baselines. 

This repository provides a controlled experimental framework evaluating:
* **FOP_MAV**: A clean implementation of the FOP architecture with linear fusion.
* **GRL**: Gradient Reversal Layer for audio language disentanglement.
* **Modality Dropout**: Asymmetric missing-modality training.
* **MultiBranchFOP**: Full cross-attention architecture demonstrating the robustness vs. transfer trade-off.

---

## Pre-trained Checkpoints & Reviewer Verification

For paper reviewers and researchers seeking quick verification, pre-trained model weights are hosted on Hugging Face and can be downloaded directly:

🤗 **Hugging Face Model Weights & Checkpoints Repository:**  
👉 **[https://huggingface.co/datasets/MuhammadNouman911/Multimodal-Fusion-Tradeoffs](https://huggingface.co/datasets/MuhammadNouman911/Multimodal-Fusion-Tradeoffs)**

| Checkpoint File | Architecture | Description | Target Performance | Download Link |
|:---|:---|:---|:---|:---|
| `fop_linear_seed42_best.pt` | Linear Fusion Baseline (FOP_MAV) | Best baseline model trained on English, demonstrating high cross-lingual transfer. | **98.16%** Zero-Shot Urdu | [Hugging Face](https://huggingface.co/datasets/MuhammadNouman911/Multimodal-Fusion-Tradeoffs) |
| `v17_ZeroShot_IEEE_seed42_fold0_hybrid_ep153_p399.0_p499.0.pt` | MultiBranch Hybrid Fusion | Full cross-attention model exhibiting cross-lingual trade-off. | **86.04%** Zero-Shot Urdu | [Hugging Face](https://huggingface.co/datasets/MuhammadNouman911/Multimodal-Fusion-Tradeoffs) |

---

## Datasets: Full & Sample Datasets

### 1. MAV-Celeb Full Dataset
The full experiments utilize the **MAV-Celeb** multilingual audio-visual dataset:
* **Training**: 70 English speaker identities (multimodal).
* **Evaluation (Zero-Shot)**: The same 70 speaker identities in Urdu.
* **Features**: Pre-extracted FaceNet (512-d) and ECAPA-TDNN / WavLM (192-d) embeddings.

### 2. Sample Dataset (`datasets/sample_data/`)
> [!NOTE]  
> **Included Sample Dataset for Demonstration & Testing:**  
> A lightweight sample dataset containing **5 samples per subset** (English Train/Val and Urdu Train/Val with precomputed FaceNet and ECAPA embeddings) is pushed directly to `datasets/sample_data/`. This enables reviewers and users to verify dataloading, model instantiation, and pipeline execution out of the box without requiring the full multi-gigabyte dataset.

---

## Directory Structure

```text
Multimodal-Fusion-Tradeoffs/
├── checkpoints/              # Pre-trained model weights (Linear Baseline & MultiBranch)
├── configs/                  # Configuration files (config.py with hyperparameter overrides)
├── datasets/                 # Lightweight sample dataset for demonstration (5 samples each)
│   └── sample_data/          # CSV manifests & sample feature tensors (.npy)
├── results/                  # Generated figures, CSV summaries, & JSON diagnostic reports
│   ├── ablation_study/       # Multi-seed statistical summary tables
│   ├── figures/              # Publication figures (PDF & PNG)
│   └── metrics/              # Quantitative reports and audit outputs
├── scripts/                  # Execution & evaluation scripts
│   ├── ablation/             # Factorial ablation, mechanistic diagnostics & scaling scripts
│   ├── evaluation/           # Zero-shot evaluation pipeline (evaluate_zero_shot_p56.py)
│   ├── figures/              # Plot generation script (plot_advanced_figures.py)
│   ├── preprocessing/        # Fold auditing & data validation scripts
│   └── train/                # Training pipelines (main.py, train_baseline_fop.py)
├── src/                      # Source code
│   ├── datasets/             # Data loading and feature preprocessing (featLoader.py)
│   ├── fusion/               # Fusion module implementations
│   ├── losses/               # Disentanglement & classification loss functions
│   ├── models/               # Network architectures (fop.py, multibranch.py, model.py)
│   └── trainers/             # Training loops and early stopping (trainer.py, evaluator.py)
├── .gitignore                # Configured to include sample data & checkpoints while ignoring raw data
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muhammadnouman911/Multimodal-Fusion-Tradeoffs.git
cd Multimodal-Fusion-Tradeoffs
```

2. Create a virtual or conda environment and install dependencies:
```bash
conda create -n multibranchfop python=3.10 -y
conda activate multibranchfop
pip install -r requirements.txt
```

---

## Reproducibility & Execution Commands

### 1. Zero-Shot Evaluation (Pre-trained Checkpoints)
Evaluate the included pre-trained checkpoint on zero-shot Urdu performance:
```bash
python scripts/evaluation/evaluate_zero_shot_p56.py --config configs/config.py
```

### 2. Training Baseline (Linear Fusion)
To train the baseline linear-fusion FOP_MAV model on the English partition:
```bash
python scripts/train/train_baseline_fop.py --config configs/config.py
```

### 3. Full MultiBranchFOP Training
To train the full architecture (SAM+SWA, GRL, Modality Dropout, Cross-Attention):
```bash
python scripts/train/main.py --config configs/config.py
```

### 4. Ablation & Diagnostic Experiments
Run full factorial ablation studies and scaling analysis:
```bash
python scripts/ablation/run_ablation.py
python scripts/ablation/run_exp_d_and_primes.py
python scripts/ablation/mechanistic_diagnostics.py
python scripts/ablation/scaling_and_generalization.py
```

### 5. Publication Figure Generation
Generate high-resolution IEEE figures (t-SNE embeddings, attention heatmaps, radar chart, component contribution):
```bash
python scripts/figures/plot_advanced_figures.py
```

---

## Experimental Results Summary

**Zero-Shot Urdu Accuracy on MAV-Celeb (Fold 0)**

| Configuration | Optimizer | $P_5$ (Multimodal) | $P_6$ (Audio-Only) | Modality Gap | Language Gap |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FOP_MAV (Baseline)** | AdamW | **98.16%** | 85.67% | 12.49% | **1.47%** |
| + GRL (Exp B) | AdamW | 97.97% | **86.38%** | 11.59% | 2.03% |
| + Dropout (Exp A) | AdamW | 95.26% | 84.16% | 11.10% | 4.37% |
| **MultiBranch Fusion Only** (Exp D) | AdamW | 80.83% | 80.83% | **0.00%** | 17.94% |
| **MultiBranchFOP (Full)** | SAM+SWA | 86.04% | 82.12% | 3.92% | 12.97% |

*The primary finding is the performance drop in Experiment D relative to the baseline, confirming that cross-attention mechanisms can severely impair cross-lingual transferability despite improving within-distribution modality robustness.*

---

## Citation
If you use our evaluation framework or find this codebase helpful for your research, please cite our paper:
```bibtex
@inproceedings{nouman2026multimodal,
  title={When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification},
  author={Muhammad Nouman et al.},
  booktitle={IEEE Conference Presentation},
  year={2026}
}
```

---

## License
This project is licensed under the MIT License.
