# When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification

## Overview
This repository contains the official PyTorch implementation and reproducible evaluation scripts for the paper **"When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification"**.

We investigate the zero-shot cross-lingual transferability of multimodal (audio-visual) speaker identification systems. Specifically, we demonstrate that while sophisticated cross-attention mechanisms (like MultiBranch Hybrid Fusion) improve within-language missing-modality robustness, they severely degrade cross-lingual generalization compared to simple linear fusion baselines. 

This repository provides a controlled experimental framework evaluating:
* **FOP_MAV**: A clean re-implementation of the FOP architecture with linear fusion.
* **GRL**: Gradient Reversal Layer for audio language disentanglement.
* **Modality Dropout**: Asymmetric missing-modality training.
* **MultiBranchFOP**: Full cross-attention architecture demonstrating the robustness vs. transfer trade-off.

## Architecture

Our ablation study evaluates the **MultiBranchFOP** architecture, consisting of parallel FaceNet and WavLM embedding branches, optional domain-adversarial GRL, Asymmetric Modality Dropout, and a MultiBranch Hybrid Fusion module.

*(For detailed architectural diagrams, please see Figure 1 in the paper (`paper/pdf/IEEE-conference-template-062824.pdf`).)*

## Dataset: MAV-Celeb
The experiments utilize the **MAV-Celeb** multilingual audio-visual dataset. 
* **Training**: 70 English speaker identities (multimodal).
* **Evaluation (Zero-Shot)**: The same 70 speaker identities in Urdu.
* **Features**: Pre-extracted FaceNet (512-d) and WavLM (192-d) embeddings.

*Note: Datasets must be placed in the `Train/` and `Dev/` directories at the project root.*

## Folder Structure
```text
.
├── configs/            # Configuration files (hyperparameters, paths)
├── src/                # Core source code
│   ├── models/         # Architecture definitions (fop.py, multibranch.py)
│   ├── fusion/         # Fusion mechanisms
│   ├── losses/         # Loss functions
│   ├── trainers/       # Training loops and early stopping
│   ├── datasets/       # Dataloaders and feature loading
│   └── utils/          # Evaluation metrics and utilities
├── scripts/            # Execution scripts
│   ├── train/          # Main training scripts
│   ├── evaluation/     # Zero-shot evaluation
│   ├── ablation/       # Factorial ablation and isolation scripts
│   └── preprocessing/  # Data preparation
├── experiments/        # Log output directories for respective configs
├── checkpoints/        # Saved model weights
├── results/            # Generated tables, metrics, and figures
├── paper/              # LaTeX source, generated PDFs, and figures
└── unused/             # Legacy code, temporary logs, and unused iterations
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muhammadnouman911/Multimodal-Fusion-Tradeoffs.git
cd Multimodal-Fusion-Tradeoffs
```

2. Create a conda environment and install dependencies:
```bash
conda create -n multibranchfop python=3.10
conda activate multibranchfop
pip install -r requirements.txt
```

## Commands for Reproducibility

### 1. Training (English Baseline)
To train the baseline linear-fusion FOP_MAV model on the English partition:
```bash
python scripts/train/train_baseline_fop.py --config configs/config.py
```

### 2. Full MultiBranchFOP Training
To train the full architecture (SAM+SWA, GRL, Modality Dropout, Cross-Attention):
```bash
python scripts/train/main.py --config configs/config.py
```

### 3. Ablation Experiments
To run the full factorial ablation matrix (Experiments A, B, C, D, A', B', C'):
```bash
python scripts/ablation/run_ablation.py
python scripts/ablation/run_exp_d_and_primes.py
```

### 5. Advanced Figure Generation
To generate publication-grade IEEE figures (t-SNE embedding plots, attention heatmaps, radar chart, component contribution bar chart):
```bash
python scripts/figures/plot_advanced_figures.py
```

## Results

**Summary of Zero-Shot Urdu Accuracy on MAV-Celeb (Fold 0)**

| Configuration | Optimizer | $P_5$ (Multimodal) | $P_6$ (Audio-Only) | Modality Gap | Language Gap |
|---------------|-----------|--------------------|--------------------|--------------|--------------|
| **FOP_MAV (Baseline)** | AdamW | **98.16%** | 85.67% | 12.49% | **1.47%** |
| + GRL (Exp B) | AdamW | 97.97% | **86.38%** | 11.59% | 2.03% |
| + Dropout (Exp A) | AdamW | 95.26% | 84.16% | 11.10% | 4.37% |
| **MultiBranch Fusion Only** (Exp D) | AdamW | 80.83% | 80.83% | **0.00%** | 17.94% |
| **MultiBranchFOP (Full)** | SAM+SWA | 86.04% | 82.12% | 3.92% | 12.97% |

*The primary finding is the 17.33% performance collapse in Experiment D relative to the baseline, confirming that cross-attention mechanisms can severely impair cross-lingual transferability despite improving within-distribution modality robustness.*

## Citation
If you use our evaluation framework or find the analysis helpful, please cite our paper:
```bibtex
@inproceedings{anonymous2026when,
  title={When Multimodal Fusion Hurts: Understanding Cross-Lingual and Missing-Modality Trade-offs in Speaker Identification},
  author={Anonymous},
  booktitle={IEEE Conference on ...},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or issues regarding the code, please open a GitHub issue or contact the authors.
