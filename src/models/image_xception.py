from typing import Any

import timm
import torch
import torch.nn as nn


class ImageXceptionBinary(nn.Module):
    def __init__(self, name: str = "xception", pretrained: bool = True, in_chans: int = 3, drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, num_classes=0, drop_rate=drop_rate)
        feat_dim = self.backbone.num_features if hasattr(self.backbone, "num_features") else self.backbone.num_features
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits.squeeze(1)