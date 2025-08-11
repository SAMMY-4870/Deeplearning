from typing import Optional

import torch
import torch.nn as nn


class BCEWithLogitsLossSmooth(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]) if pos_weight else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets.clamp(self.label_smoothing, 1.0 - self.label_smoothing)
        return self.bce(logits, targets)


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * ce_loss
        return loss.mean()