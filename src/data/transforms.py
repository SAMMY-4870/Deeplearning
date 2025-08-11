from typing import Tuple

from torchvision import transforms


def build_train_transforms(image_size: int, hflip_prob: float = 0.5, color_jitter=None):
    cj = None
    if color_jitter:
        cj = transforms.ColorJitter(*color_jitter)
    t = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=hflip_prob),
    ]
    if cj:
        t.append(cj)
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(t)


def build_eval_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def apply_transforms_to_frames(frames, transform):
    # frames: list of PIL Images
    return [transform(f) for f in frames]