from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)

    accuracy = float((preds == labels).mean())
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    try:
        f1 = float(f1_score(labels, preds))
    except Exception:
        f1 = float("nan")

    return {"accuracy": accuracy, "auc": auc, "f1": f1}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))