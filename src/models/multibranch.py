import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EmbedBranch, GatedFusion, HybridMultiPathFusion, LSTMBottleneckFusion, ArcFaceLinear, SubCenterArcFaceLinear, GRL


class MultiBranchFOP(nn.Module):
    """
    Supreme Multi-branch model with 3 selectable Fusion Paths.
    """

    def __init__(self, config, face_dim: int, audio_dim: int):
        super().__init__()

        emb         = config.embedding_dim
        num_classes = config.resolved_num_classes

        # ── Embedding branches ────────────────────────────────────────────────
        self.face_branch  = EmbedBranch(face_dim,  emb)
        self.audio_branch = EmbedBranch(audio_dim, emb)

        # ── Fusion ────────────────────────────────────────────────────────────
        if config.fusion == "gated":
            self.fusion = GatedFusion(emb)
        elif config.fusion == "hybrid":
            self.fusion = HybridMultiPathFusion(emb)
        elif config.fusion == "temporal":
            self.fusion = LSTMBottleneckFusion(emb)
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        # ── Hybrid Classification Heads ──────────────────────────────────────
        # Standard Linear heads for stable Softmax (reclaiming v16 performance)
        self.face_logits  = nn.Linear(emb, num_classes)
        self.audio_logits = nn.Linear(emb, num_classes)
        self.fusion_logits = nn.Linear(emb, num_classes)

        # Sub-Center ArcFace heads for deep discrimination
        self.face_arcface  = SubCenterArcFaceLinear(emb, num_classes, k=config.arcface_k, s=config.arcface_s, m=config.arcface_m)
        self.audio_arcface = SubCenterArcFaceLinear(emb, num_classes, k=config.arcface_k, s=config.arcface_s, m=config.arcface_m)
        self.fusion_arcface = SubCenterArcFaceLinear(emb, num_classes, k=config.arcface_k, s=config.arcface_s, m=config.arcface_m)
        
        # ── Language Adversarial Head ─────────────────────────────────────────
        self.grl = GRL(alpha=config.grl_alpha)
        self.lang_classifier = nn.Sequential(
            nn.Linear(emb, emb // 2),
            nn.BatchNorm1d(emb // 2),
            nn.GELU(),
            nn.Linear(emb // 2, 2)
        )

        self.modality_dropout_face = config.modality_dropout_face
        self.modality_dropout_audio = config.modality_dropout_audio

    def forward(self, face: torch.Tensor, audio: torch.Tensor, labels: torch.Tensor = None) -> dict:
        face_e  = self.face_branch(face)
        audio_e = self.audio_branch(audio)

        # ── Asymmetric Modality Dropout ───────────────────────────────────────
        # Skewed toward dropping face (60%) to strengthen audio branch
        if self.training:
            r = torch.rand(1).item()
            if r < self.modality_dropout_face:
                face_e_drop = torch.zeros_like(face_e)
                audio_e_drop = audio_e
            elif r < (self.modality_dropout_face + self.modality_dropout_audio):
                face_e_drop = face_e
                audio_e_drop = torch.zeros_like(audio_e)
            else:
                face_e_drop = face_e
                audio_e_drop = audio_e
        else:
            face_e_drop = face_e
            audio_e_drop = audio_e

        streams = [face_e_drop, audio_e_drop]

        # Fusion
        if isinstance(self.fusion, HybridMultiPathFusion):
            fused, _, _ = self.fusion(*streams)
        elif isinstance(self.fusion, LSTMBottleneckFusion):
            fused = self.fusion(streams[0], streams[1])
        else:
            fused, _, _ = self.fusion(streams[0], streams[1])

        # ── Step 5: Multi-Objective Classification ─────────────────────────────
        # ArcFace stream
        face_af   = self.face_arcface(face_e, labels)
        audio_af  = self.audio_arcface(audio_e, labels)
        fusion_af = self.fusion_arcface(fused, labels)
        
        # Standard Logits stream (for stable Softmax)
        face_lo   = self.face_logits(face_e)
        audio_lo  = self.audio_logits(audio_e)
        fusion_lo = self.fusion_logits(fused)
        
        # ── Language Adversarial Logits ───────────────────────────────────────
        # Apply GRL to audio embeddings to force language-invariance
        lang_logits = self.lang_classifier(self.grl(audio_e))

        return {
            "face_logits": face_lo, "audio_logits": audio_lo, "fusion_logits": fusion_lo,
            "face_af": face_af, "audio_af": audio_af, "fusion_af": fusion_af,
            "face_embed": face_e, "audio_embed": audio_e, "fusion_embed": fused,
            "lang_logits": lang_logits
        }

def build_model(config, face_dim: int, audio_dim: int):
    return MultiBranchFOP(config, face_dim, audio_dim)
