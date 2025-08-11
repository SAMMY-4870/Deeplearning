import os
from typing import Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import load_config
from .data.image_dataset import ImageBinaryDataset
from .data.video_dataset import VideoBinaryDataset
from .data.transforms import build_train_transforms, build_eval_transforms
from .engine import train_one_epoch, validate
from .losses import BCEWithLogitsLossSmooth
from .models.image_xception import ImageXceptionBinary
from .models.video_cnn_lstm import VideoCnnLstmBinary
from .utils import get_device, set_seed, save_checkpoint, load_checkpoint


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    task = cfg["task_type"]
    num_workers = int(cfg.get("num_workers", 4))

    if task == "image":
        size = int(cfg.get("image_size", 299))
        train_tf = build_train_transforms(size, cfg.get("augment", {}).get("hflip_prob", 0.5), cfg.get("augment", {}).get("color_jitter"))
        eval_tf = build_eval_transforms(size)
        train_ds = ImageBinaryDataset(cfg["DATA_ROOT"], cfg["paths"]["train"], transform=train_tf)
        val_ds = ImageBinaryDataset(cfg["DATA_ROOT"], cfg["paths"]["val"], transform=eval_tf)
    elif task == "video":
        size = int(cfg.get("video", {}).get("image_size", 224))
        num_frames = int(cfg.get("video", {}).get("num_frames", 16))
        train_tf = build_train_transforms(size, cfg.get("augment", {}).get("hflip_prob", 0.5))
        eval_tf = build_eval_transforms(size)
        train_ds = VideoBinaryDataset(cfg["DATA_ROOT"], cfg["paths"]["train"], num_frames=num_frames, transform=train_tf)
        val_ds = VideoBinaryDataset(cfg["DATA_ROOT"], cfg["paths"]["val"], num_frames=num_frames, transform=eval_tf)
    else:
        raise ValueError(f"Unknown task_type: {task}")

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model(cfg):
    task = cfg["task_type"]
    if task == "image":
        m = cfg["model"]
        return ImageXceptionBinary(name=m.get("name", "xception"), pretrained=bool(m.get("pretrained", True)), in_chans=int(m.get("in_chans", 3)), drop_rate=float(m.get("drop_rate", 0.2)))
    elif task == "video":
        m = cfg["model"]
        return VideoCnnLstmBinary(
            backbone=m.get("backbone", "xception"),
            pretrained=bool(m.get("pretrained", True)),
            in_chans=int(m.get("in_chans", 3)),
            lstm_hidden=int(m.get("lstm_hidden", 512)),
            lstm_layers=int(m.get("lstm_layers", 1)),
            bidirectional=bool(m.get("bidirectional", False)),
            dropout=float(m.get("dropout", 0.3)),
        )
    else:
        raise ValueError(f"Unknown task_type: {task}")


def main():
    cfg = load_config()
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    # Criterion
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    criterion = BCEWithLogitsLossSmooth(label_smoothing=label_smoothing)

    # Optimizer and scheduler
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["train"].get("epochs", 10)))

    # Resume if exists
    start_epoch = 0
    best_auc = 0.0
    ckpt_path = os.path.join(cfg["output_dir"], "checkpoint.pt")
    if os.path.isfile(ckpt_path):
        ms, osd, ssd, start_epoch, best_auc = load_checkpoint(ckpt_path)
        model.load_state_dict(ms)
        optimizer.load_state_dict(osd)
        scheduler.load_state_dict(ssd)
        print(f"Resumed from epoch {start_epoch} with best_auc={best_auc:.4f}")

    epochs = int(cfg["train"]["epochs"]) 
    use_amp = bool(cfg.get("use_amp", True))

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_stats = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print({"train": train_stats, "val": val_stats})

        # Save best by AUC
        current_auc = float(val_stats.get("auc", 0.0))
        if current_auc >= best_auc:
            best_auc = current_auc
            save_checkpoint(os.path.join(cfg["output_dir"], "best.pt"),
                            checkpoint_type(model, optimizer, scheduler, epoch + 1, best_auc))
        # Save last
        save_checkpoint(ckpt_path, checkpoint_type(model, optimizer, scheduler, epoch + 1, best_auc))

    print(f"Training complete. Best AUC: {best_auc:.4f}. Artifacts in {cfg['output_dir']}")


def checkpoint_type(model, optimizer, scheduler, epoch: int, best_metric: float):
    from .utils import Checkpoint
    return Checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        epoch=epoch,
        best_metric=best_metric,
    )


if __name__ == "__main__":
    main()