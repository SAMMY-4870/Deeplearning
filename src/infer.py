import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .data.transforms import build_eval_transforms
from .models.image_xception import ImageXceptionBinary
from .models.video_cnn_lstm import VideoCnnLstmBinary
from .utils import get_device, load_checkpoint


def load_image(path: str, size: int):
    img = Image.open(path).convert("RGB")
    tf = build_eval_transforms(size)
    return tf(img).unsqueeze(0)


def load_video(path: str, size: int, num_frames: int):
    import cv2
    from PIL import Image as PILImage
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def sample_indices(nf, total):
        step = max(total / float(nf), 1)
        idxs = [int(step * i + step / 2) for i in range(nf)]
        return [min(i, max(total - 1, 0)) for i in idxs]
    idxs = set(sample_indices(num_frames, total_frames))
    frames: List[PILImage] = []
    cur = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PILImage.fromarray(frame))
        cur += 1
    cap.release()
    if len(frames) == 0:
        frames = [PILImage.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(num_frames)]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    tf = build_eval_transforms(size)
    frames = [tf(f) for f in frames]
    video = torch.stack(frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)
    video = video.permute(0, 1, 2, 3, 4)
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--path", required=True, type=str, help="Path to an image or video file")
    parser.add_argument("--checkpoint", required=True, type=str, help="Model checkpoint path")
    args = parser.parse_args()

    cfg = None
    # Reuse loader to normalize output dir; we don't need overrides
    from .config import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    task = cfg["task_type"]
    if task == "image":
        size = int(cfg.get("image_size", 299))
        model = ImageXceptionBinary(name=cfg["model"].get("name", "xception"), pretrained=False).to(device)
        ms, _, _, _, _ = load_checkpoint(args.checkpoint)
        model.load_state_dict(ms, strict=True)
        model.eval()
        x = load_image(args.path, size).to(device)
        with torch.no_grad():
            logit = model(x).squeeze(0).item()
        prob_fake = float(1.0 / (1.0 + np.exp(-logit)))
    else:
        size = int(cfg.get("video", {}).get("image_size", 224))
        num_frames = int(cfg.get("video", {}).get("num_frames", 16))
        model = VideoCnnLstmBinary(
            backbone=cfg["model"].get("backbone", "xception"),
            pretrained=False,
            in_chans=int(cfg["model"].get("in_chans", 3)),
            lstm_hidden=int(cfg["model"].get("lstm_hidden", 512)),
            lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
            bidirectional=bool(cfg["model"].get("bidirectional", False)),
            dropout=float(cfg["model"].get("dropout", 0.3)),
        ).to(device)
        ms, _, _, _, _ = load_checkpoint(args.checkpoint)
        model.load_state_dict(ms, strict=True)
        model.eval()
        video = load_video(args.path, size, num_frames).to(device)
        with torch.no_grad():
            logit = model(video).squeeze(0).item()
        prob_fake = float(1.0 / (1.0 + np.exp(-logit)))

    print({"file": args.path, "prob_fake": prob_fake, "prediction": "fake" if prob_fake >= 0.5 else "real"})


if __name__ == "__main__":
    main()