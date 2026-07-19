import torch
import torch.nn as nn

from .model import EmbedBranch, GatedFusion, HybridMultiPathFusion, GRL


class FOP(nn.Module):
    """
    Fusion and Orthogonal Projection model (single fusion head).
    """

    def __init__(self, config, face_dim: int, audio_dim: int):
        super().__init__()

        emb_dim     = config.embedding_dim
        num_classes = config.resolved_num_classes

        self.face_branch  = EmbedBranch(face_dim,  emb_dim)
        self.audio_branch = EmbedBranch(audio_dim, emb_dim)

        if config.fusion == "gated":
            self.fusion   = GatedFusion(emb_dim)
            fusion_dim    = emb_dim
        elif config.fusion == "hybrid":
            self.fusion   = HybridMultiPathFusion(emb_dim)
            fusion_dim    = emb_dim
        elif config.fusion in ["concat", "linear"]:
            self.fusion   = None
            fusion_dim    = emb_dim * 2
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        self.classifier = nn.Linear(fusion_dim, num_classes)
        
        # ── Language Adversarial Head (GRL) ──────────────────────────────────
        self.grl_alpha = getattr(config, "grl_alpha", 0.0)
        if self.grl_alpha > 0:
            self.grl = GRL(alpha=self.grl_alpha)
            self.lang_classifier = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.BatchNorm1d(emb_dim // 2),
                nn.GELU(),
                nn.Linear(emb_dim // 2, 2)
            )
            
        self.modality_dropout_face = getattr(config, "modality_dropout_face", 0.0)
        self.modality_dropout_audio = getattr(config, "modality_dropout_audio", 0.0)

    def forward(self, face: torch.Tensor, audio: torch.Tensor):
        face_e  = self.face_branch(face)
        audio_e = self.audio_branch(audio)

        # ── Modality Dropout ──────────────────────────────────────────────────
        if self.training:
            # Face Dropout
            if self.modality_dropout_face > 0 and torch.rand(1).item() < self.modality_dropout_face:
                face_e = torch.zeros_like(face_e)
            # Audio Dropout
            if self.modality_dropout_audio > 0 and torch.rand(1).item() < self.modality_dropout_audio:
                audio_e = torch.zeros_like(audio_e)

        if self.fusion is None:
            fused = torch.cat([face_e, audio_e], dim=1)
        else:
            fused, _, _ = self.fusion(face_e, audio_e)

        logits = self.classifier(fused)
        
        # ── GRL Forward ────────────────────────────────────────────────────────
        lang_logits = None
        if self.grl_alpha > 0 and self.training:
            # Pass audio embedding through GRL to classify language
            lang_feat = self.grl(audio_e)
            lang_logits = self.lang_classifier(lang_feat)
            
        # Return dict to stay uniform with MultiBranchFOP
        return {
            "fusion_logits": logits,
            "face_embedding": face_e,
            "audio_embedding": audio_e,
            "lang_logits": lang_logits
        }
