from typing import Tuple

import timm
import torch
import torch.nn as nn


def build_xception_binary(num_classes: int = 1) -> nn.Module:
    """Create Xception model with a binary classifier head (logit output)."""
    try:
        model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    except Exception:
        model = timm.create_model("xception", pretrained=False, num_classes=num_classes)
    return model


def load_weights_if_available(model: nn.Module, weights_path: str) -> Tuple[nn.Module, bool]:
    if not weights_path:
        return model, False
    try:
        state = torch.load(weights_path, map_location="cpu")
        # Support loading a raw state_dict or a checkpoint dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model, True
    except Exception:
        return model, False