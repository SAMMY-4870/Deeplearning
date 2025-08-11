import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data.image_dataset import ImageBinaryDataset
from .data.video_dataset import VideoBinaryDataset
from .data.transforms import build_eval_transforms
from .engine import validate
from .models.image_xception import ImageXceptionBinary
from .models.video_cnn_lstm import VideoCnnLstmBinary
from .utils import get_device, load_checkpoint
from .losses import BCEWithLogitsLossSmooth


def build_dataset(cfg, split: str):
    task = cfg["task_type"]
    num_workers = int(cfg.get("num_workers", 4))

    if task == "image":
        size = int(cfg.get("image_size", 299))
        tf = build_eval_transforms(size)
        ds = ImageBinaryDataset(cfg["DATA_ROOT"], cfg["paths"].get(split, split), transform=tf)
    elif task == "video":
        size = int(cfg.get("video", {}).get("image_size", 224))
        num_frames = int(cfg.get("video", {}).get("num_frames", 16))
        tf = build_eval_transforms(size)
        ds = VideoBinaryDataset(cfg["DATA_ROOT"], cfg["paths"].get(split, split), num_frames=num_frames, transform=tf)
    else:
        raise ValueError(f"Unknown task_type: {task}")
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 32)), shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def build_model(cfg):
    if cfg["task_type"] == "image":
        m = cfg["model"]
        return ImageXceptionBinary(name=m.get("name", "xception"), pretrained=False, in_chans=int(m.get("in_chans", 3)))
    else:
        m = cfg["model"]
        return VideoCnnLstmBinary(
            backbone=m.get("backbone", "xception"),
            pretrained=False,
            in_chans=int(m.get("in_chans", 3)),
            lstm_hidden=int(m.get("lstm_hidden", 512)),
            lstm_layers=int(m.get("lstm_layers", 1)),
            bidirectional=bool(m.get("bidirectional", False)),
            dropout=float(m.get("dropout", 0.3)),
        )


def main():
    cfg = load_config()
    device = get_device()

    split = "test" if cfg["paths"].get("test") else "val"
    loader = build_dataset(cfg, split)
    model = build_model(cfg).to(device)

    # Load checkpoint path override
    ckpt_override = None
    for item in cfg.keys():
        pass
    # Allow passing CHECKPOINT_PATH=... override
    # We look into overrides by environment: we already expanded via load_config; user sets CHECKPOINT_PATH=...
    ckpt_path = os.path.join(cfg["output_dir"], "best.pt")
    if "CHECKPOINT_PATH" in cfg:
        ckpt_path = cfg["CHECKPOINT_PATH"]

    ms, _, _, _, _ = load_checkpoint(ckpt_path)
    model.load_state_dict(ms, strict=True)

    criterion = BCEWithLogitsLossSmooth()
    stats = validate(model, loader, criterion, device, use_amp=bool(cfg.get("use_amp", True)))
    print(stats)


if __name__ == "__main__":
    main()