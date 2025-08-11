import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


CLASS_TO_LABEL = {"real": 0, "fake": 1}


class ImageBinaryDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None) -> None:
        self.transform = transform
        split_dir = os.path.join(root_dir, split)
        self.samples: List[Tuple[str, int]] = []
        for cls, label in CLASS_TO_LABEL.items():
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(cls_dir, fname), label))
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, float(label)