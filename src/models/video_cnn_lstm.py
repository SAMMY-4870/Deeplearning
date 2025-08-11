from typing import Tuple

import timm
import torch
import torch.nn as nn


class VideoCnnLstmBinary(nn.Module):
    def __init__(
        self,
        backbone: str = "xception",
        pretrained: bool = True,
        in_chans: int = 3,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.frame_encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.feature_dim = self.frame_encoder.num_features if hasattr(self.frame_encoder, "num_features") else 2048
        self.temporal = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        classifier_in = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_in, 1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        b, t, c, h, w = video.shape
        video_2d = video.view(b * t, c, h, w)
        feats = self.frame_encoder(video_2d)  # (B*T, F)
        feats = feats.view(b, t, -1)  # (B, T, F)
        out, (hn, cn) = self.temporal(feats)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.squeeze(1)