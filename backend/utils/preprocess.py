from typing import List

import numpy as np
import timm
from PIL import Image


def build_eval_transform(model_name: str = "xception"):
    cfg = timm.data.resolve_model_data_config(model_name)
    return timm.data.create_transform(**cfg, is_training=False)


def uniform_sample_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int32)
    if num_samples >= total_frames:
        return np.arange(total_frames, dtype=np.int32)
    step = total_frames / float(num_samples)
    indices = [int(step * i + step / 2.0) for i in range(num_samples)]
    indices = np.clip(indices, 0, max(0, total_frames - 1)).astype(np.int32)
    return np.unique(indices)