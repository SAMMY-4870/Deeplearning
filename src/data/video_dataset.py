import os
from typing import List, Tuple

import cv2
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_LABEL = {"real": 0, "fake": 1}


def sample_frame_indices(num_frames: int, total_frames: int) -> List[int]:
    if total_frames <= 0:
        return [0] * num_frames
    # Uniform sampling
    step = max(total_frames / float(num_frames), 1)
    indices = [int(step * i + step / 2) for i in range(num_frames)]
    indices = [min(idx, total_frames - 1) for idx in indices]
    return indices


class VideoBinaryDataset(Dataset):
    def __init__(self, root_dir: str, split: str, num_frames: int = 16, transform=None) -> None:
        self.transform = transform
        self.num_frames = num_frames
        split_dir = os.path.join(root_dir, split)
        self.samples: List[Tuple[str, int]] = []
        for cls, label in CLASS_TO_LABEL.items():
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(cls_dir, fname), label))
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        frames = self._load_video_frames(path, self.num_frames)
        if self.transform is not None:
            frames = [self.transform(f) for f in frames]
        # Stack to (T, C, H, W)
        import torch
        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor, float(label)

    @staticmethod
    def _load_video_frames(path: str, num_frames: int) -> List[Image.Image]:
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(num_frames, total_frames)
        collected: List[Image.Image] = []
        current_index = 0
        frame_map = set(indices)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in frame_map:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                collected.append(img)
                if len(collected) == num_frames:
                    break
            current_index += 1
        cap.release()

        # Pad if short
        if len(collected) == 0:
            # Create a black frame if video failed
            collected = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(num_frames)]
        elif len(collected) < num_frames:
            last = collected[-1]
            while len(collected) < num_frames:
                collected.append(last)
        return collected