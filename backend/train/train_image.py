import argparse
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.models.xception_classifier import build_xception_binary


class ImageBinaryDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = []
        for label_name in ["real", "ai"]:
            label_dir = self.root / label_name
            if not label_dir.exists():
                continue
            for p in label_dir.rglob("*.jpg"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
            for p in label_dir.rglob("*.png"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
            for p in label_dir.rglob("*.jpeg"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
        cfg = timm.data.resolve_model_data_config("xception")
        self.transform = timm.data.create_transform(**cfg, is_training=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ImageBinaryDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = build_xception_binary()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / max(1, len(ds))
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset root with subfolders 'real' and 'ai'")
    parser.add_argument("--out", type=str, default="/workspace/checkpoints/xception_image.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)