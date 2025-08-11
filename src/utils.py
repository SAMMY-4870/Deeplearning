import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    epoch: int
    best_metric: float


def save_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    torch.save({
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "scheduler_state": checkpoint.scheduler_state,
        "epoch": checkpoint.epoch,
        "best_metric": checkpoint.best_metric,
    }, path)


def load_checkpoint(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int, float]:
    data = torch.load(path, map_location="cpu")
    return (
        data.get("model_state", {}),
        data.get("optimizer_state", {}),
        data.get("scheduler_state", {}),
        int(data.get("epoch", 0)),
        float(data.get("best_metric", 0.0)),
    )


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count