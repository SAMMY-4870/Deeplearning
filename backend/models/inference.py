import io
import math
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from backend.models.video_lstm import build_video_model, load_weights_if_available as load_video_w
from backend.models.xception_classifier import build_xception_binary, load_weights_if_available as load_image_w


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_transform_for_model(model_name: str = "xception"):
    config = timm.data.resolve_model_data_config(model_name)
    transform = timm.data.create_transform(**config, is_training=False)
    return transform


def load_image_model(weights_path: Optional[str] = None):
    device = _get_device()
    model = build_xception_binary()
    model, loaded = load_image_w(model, weights_path or "")
    model.eval()
    model.to(device)
    return model, loaded


def load_video_model(weights_path: Optional[str] = None):
    device = _get_device()
    model = build_video_model()
    model, loaded = load_video_w(model, weights_path or "")
    model.eval()
    model.to(device)
    return model, loaded


def predict_image_from_bytes(model, image_bytes: bytes) -> float:
    device = next(model.parameters()).device
    transform = _build_transform_for_model("xception")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def _sample_frame_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int32)
    if num_samples >= total_frames:
        return np.arange(total_frames, dtype=np.int32)
    step = total_frames / float(num_samples)
    indices = [int(step * i + step / 2.0) for i in range(num_samples)]
    indices = np.clip(indices, 0, max(0, total_frames - 1)).astype(np.int32)
    return np.unique(indices)


def predict_video_from_path(model, video_path: str, num_frames: int = 32) -> float:
    device = next(model.parameters()).device
    transform = _build_transform_for_model("xception")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _sample_frame_indices(total_frames, num_frames)
    if indices.size == 0:
        cap.release()
        raise RuntimeError("No frames found in video")

    frames = []
    current_index = 0
    target_ptr = 0
    next_target = indices[target_ptr]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_index == next_target:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
            frames.append(tensor)
            target_ptr += 1
            if target_ptr >= len(indices):
                break
            next_target = indices[target_ptr]
        current_index += 1

    cap.release()

    if not frames:
        raise RuntimeError("Failed to sample frames from video")

    batch = torch.cat(frames, dim=0)  # (T, C, H, W)
    batch = batch.unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        logits = model(batch)
        prob = torch.sigmoid(logits).item()
    return float(prob)