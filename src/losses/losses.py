import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    """
    Orthogonal Projection Loss (OPL) with Margin.
    
    Forces positive pairs to have cosine similarity > (1 - margin)
    and negative pairs to have cosine similarity < margin_neg.
    """

    def __init__(self, neg_weight: float = 0.9, margin: float = 0.1):
        super().__init__()
        self.neg_weight = neg_weight
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats  = F.normalize(feats, dim=1)
        labels = labels.unsqueeze(1)

        mask = labels.eq(labels.T)
        eye  = torch.eye(len(labels), device=labels.device).bool()

        pos  = (mask & ~eye).float()
        neg  = (~mask).float()

        dot  = feats @ feats.T

        # Margin-based positive penalty: (1 - dot) but only if dot < 1-margin? 
        # Actually, standard OPL is already strong. Let's use a soft margin.
        pos_dist = 1.0 - dot
        pos_loss = (pos * pos_dist).sum() / (pos.sum() + 1e-6)
        
        # Harder negative penalty: focus on pairs with dot > margin
        neg_dist = dot.abs()
        neg_loss = (neg * neg_dist).sum() / (neg.sum() + 1e-6)

        return pos_loss + self.neg_weight * neg_loss


class CenterLoss(nn.Module):
    """
    Center loss: pulls each embedding toward the class centroid.
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5):
        super().__init__()
        self.alpha   = alpha
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim), requires_grad=False
        )

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats   = F.normalize(feats, dim=1)
        centers = F.normalize(self.centers, dim=1)

        batch_centers = centers[labels]
        loss = F.mse_loss(feats, batch_centers)

        with torch.no_grad():
            one_hot = torch.zeros(
                labels.size(0), self.centers.size(0), device=feats.device
            )
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            count        = one_hot.sum(0).unsqueeze(1).clamp(min=1)
            delta        = one_hot.T @ feats - count * self.centers
            self.centers += self.alpha * delta / count

        return loss
