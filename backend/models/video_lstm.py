from typing import Tuple

import timm
import torch
import torch.nn as nn


class VideoXceptionLSTM(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_size: int = 512, num_layers: int = 1):
        super().__init__()
        # Backbone returns pooled features if num_classes=0 with global_pool="avg"
        try:
            self.backbone = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="avg")
        except Exception:
            self.backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
        # Try to infer feature dimension from backbone if possible
        inferred_dim = getattr(self.backbone, "num_features", None)
        if isinstance(inferred_dim, int) and inferred_dim > 0:
            feature_dim = inferred_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time, channels, height, width)
        Returns:
            logits: Tensor of shape (batch, 1)
        """
        batch_size, time_steps = x.shape[0], x.shape[1]
        x = x.view(batch_size * time_steps, x.shape[2], x.shape[3], x.shape[4])
        feats = self.backbone(x)  # (B*T, F)
        feats = feats.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(feats)
        last_out = lstm_out[:, -1, :]
        logits = self.classifier(last_out)
        return logits


def build_video_model(hidden_size: int = 512, num_layers: int = 1) -> nn.Module:
    return VideoXceptionLSTM(hidden_size=hidden_size, num_layers=num_layers)


def load_weights_if_available(model: nn.Module, weights_path: str) -> Tuple[nn.Module, bool]:
    if not weights_path:
        return model, False
    try:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model, True
    except Exception:
        return model, False