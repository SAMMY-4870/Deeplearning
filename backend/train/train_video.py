import argparse
import os
from pathlib import Path

import cv2
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.models.video_lstm import build_video_model
from backend.utils.preprocess import build_eval_transform, uniform_sample_indices


class VideoBinaryDataset(Dataset):
    def __init__(self, root: str, num_frames: int = 32):
        self.root = Path(root)
        self.samples = []
        self.num_frames = num_frames
        for label_name in ["real", "ai"]:
            label_dir = self.root / label_name
            if not label_dir.exists():
                continue
            for p in label_dir.rglob("*.mp4"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
            for p in label_dir.rglob("*.mov"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
            for p in label_dir.rglob("*.avi"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
            for p in label_dir.rglob("*.mkv"):
                self.samples.append((str(p), 0 if label_name == "real" else 1))
        cfg = timm.data.resolve_model_data_config("xception")
        self.transform = timm.data.create_transform(**cfg, is_training=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = uniform_sample_indices(total, self.num_frames)
        frames = []
        i = 0
        ptr = 0
        target = indices[ptr] if len(indices) > 0 else -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i == target:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                x = self.transform(img)
                frames.append(x)
                ptr += 1
                if ptr >= len(indices):
                    break
                target = indices[ptr]
            i += 1
        cap.release()
        if not frames:
            # Fallback: create a dummy frame to avoid crashes
            dummy = torch.zeros(3, 299, 299)
            frames = [dummy for _ in range(self.num_frames)]
        x = torch.stack(frames, dim=0)  # (T, C, H, W)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


def collate_batch(batch):
    xs, ys = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    # Pad sequences to same length
    padded = []
    for x in xs:
        if x.shape[0] < max_t:
            pad = torch.zeros(max_t - x.shape[0], *x.shape[1:])
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    X = torch.stack(padded, dim=0)  # (B, T, C, H, W)
    Y = torch.stack(ys, dim=0)      # (B, 1)
    return X, Y


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VideoBinaryDataset(args.data, num_frames=args.num_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_batch)

    model = build_video_model()
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
    parser.add_argument("--out", type=str, default="/workspace/checkpoints/video_lstm.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)