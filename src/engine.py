from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_binary_metrics
from .utils import AverageMeter


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, use_amp: bool = True):
    model.train()
    loss_meter = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(loader, desc="Train", leave=False):
        inputs, targets = batch
        targets = targets.to(device).float()
        inputs = inputs.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs)
            if logits.ndim == 1:
                logits = logits
            else:
                logits = logits.squeeze(1)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=inputs.size(0))

    return {"loss": loss_meter.avg}


def validate(model, loader: DataLoader, criterion, device, use_amp: bool = True):
    model.eval()
    loss_meter = AverageMeter()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            inputs, targets = batch
            targets = targets.to(device).float()
            inputs = inputs.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                if logits.ndim == 1:
                    logits = logits
                else:
                    logits = logits.squeeze(1)
                loss = criterion(logits, targets)

            loss_meter.update(loss.item(), n=inputs.size(0))
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(logits_np, labels_np)
    metrics.update({"loss": loss_meter.avg})
    return metrics