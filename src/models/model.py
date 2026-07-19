import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Residual Blocks & Utilities
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.gelu = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.gelu(x + self.net(x))

class MSDLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_samples: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.linear = nn.Linear(in_dim, out_dim)
        self.temp = nn.Parameter(torch.ones(1) * 1.0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training: return self.linear(x) / self.temp
        logits = 0
        for i in range(self.num_samples): logits += self.linear(self.dropouts[i](x))
        return (logits / self.num_samples) / self.temp


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Embedding Branch
# ──────────────────────────────────────────────────────────────────────────────

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim: int, emb_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(feat_dim, emb_dim)
        self.bn   = nn.BatchNorm1d(emb_dim)
        self.res1 = ResidualBlock(emb_dim, dropout=dropout)
        self.res2 = ResidualBlock(emb_dim, dropout=dropout)
        self.res3 = ResidualBlock(emb_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3: x = x.mean(dim=1)
        x = self.input_drop(x)
        x = self.bn(F.gelu(self.proj(x)))
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        return F.normalize(x, dim=1, p=2)


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Modules
# ──────────────────────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    def __init__(self, emb_dim: int, mid_dim: int = 512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim * 2, mid_dim), nn.BatchNorm1d(mid_dim), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(mid_dim, emb_dim),
        )
        self.face_proj = nn.Linear(emb_dim, emb_dim); self.audio_proj = nn.Linear(emb_dim, emb_dim)
    def forward(self, face: torch.Tensor, audio: torch.Tensor):
        gate = torch.sigmoid(self.attention(torch.cat([face, audio], dim=1)))
        f_t = torch.tanh(self.face_proj(face)); a_t = torch.tanh(self.audio_proj(audio))
        fused = gate * f_t + (1.0 - gate) * a_t
        return F.normalize(fused, dim=1, p=2), f_t, a_t

class HybridMultiPathFusion(nn.Module):
    def __init__(self, emb_dim: int, max_modalities: int = 4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.modality_embed = nn.Parameter(torch.randn(1, max_modalities + 1, emb_dim)) # +1 for CLS
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True, dropout=0.2)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, *embeddings: torch.Tensor):
        B = embeddings[0].size(0)
        N = len(embeddings)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.stack(embeddings, dim=1) # (B, N, D)
        seq = torch.cat([cls_tokens, seq], dim=1) # (B, N+1, D)
        
        seq = seq + self.modality_embed[:, :N+1, :]
        
        attn_out, _ = self.attn(seq, seq, seq)
        fused = self.norm(attn_out[:, 0, :])
        
        return F.normalize(fused, dim=1, p=2), None, None

class LSTMBottleneckFusion(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, emb_dim); self.norm = nn.LayerNorm(emb_dim)
    def forward(self, face: torch.Tensor, audio: torch.Tensor):
        out, _ = self.lstm(torch.stack([face, audio], dim=1))
        fused = self.proj(out.mean(dim=1))
        return F.normalize(self.norm(face + fused), dim=1, p=2)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1 Components: ArcFace & GRL
# ──────────────────────────────────────────────────────────────────────────────

class ArcFaceLinear(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for deep face recognition.
    """
    def __init__(self, in_dim, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        
        # Precompute constants
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, x, labels=None):
        # x is (B, D), weights is (C, D)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        if labels is None or not self.training:
            return cosine * self.s
            
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Additive margin condition
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class SubCenterArcFaceLinear(nn.Module):
    """
    Sub-center ArcFace: Allows multiple centers (k) per class to handle intra-class variance.
    """
    def __init__(self, in_dim, n_classes, k=3, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes * k, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.k = k
        self.n_classes = n_classes
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)) # (B, C*K)
        
        if self.k > 1:
            cosine = cosine.view(-1, self.n_classes, self.k)
            cosine = cosine.max(dim=2)[0] # (B, C)
            
        if labels is None or not self.training:
            return cosine * self.s
            
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)
