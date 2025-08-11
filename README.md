# AI vs Real Media Detector (Image + Video)

A PyTorch project to classify whether an input image or video is AI-generated (fake) or real. Supports:
- Image classification using Xception backbone (`timm`)
- Video classification using CNN feature extractor + LSTM temporal model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Layout

You can use any dataset arranged as:

Images:
```
DATA_ROOT/
  train/
    real/
      img_001.jpg
      ...
    fake/
      img_101.jpg
      ...
  val/
    real/
    fake/
  test/
    real/
    fake/
```

Videos:
```
DATA_ROOT/
  train/
    real/
      vid_001.mp4
      ...
    fake/
      vid_201.mp4
      ...
  val/
    real/
    fake/
  test/
    real/
    fake/
```

Recommended public datasets: DFDC, FaceForensics++, Celeb-DF (follow their licenses and usage terms).

## Quick Start

Train an image model (Xception):
```bash
python -m src.train --config configs/image_xception.yaml DATA_ROOT=/path/to/images
```

Train a video model (CNN+LSTM):
```bash
python -m src.train --config configs/video_cnn_lstm.yaml DATA_ROOT=/path/to/videos
```

Evaluate a trained checkpoint:
```bash
python -m src.eval --config configs/image_xception.yaml CHECKPOINT_PATH=/path/to/checkpoint.pt DATA_ROOT=/path/to/images
```

Run inference on a single file:
```bash
# Image
python -m src.infer --config configs/image_xception.yaml --path /path/to/file.jpg --checkpoint /path/to/checkpoint.pt

# Video
python -m src.infer --config configs/video_cnn_lstm.yaml --path /path/to/file.mp4 --checkpoint /path/to/checkpoint.pt
```

## Configs

See `configs/` for examples. You can override any config key from the CLI using `KEY=VALUE` pairs.

## Notes
- Binary labels: `real` → 0, `fake` → 1
- Uses BCEWithLogits loss; metrics include accuracy, F1, and AUC
- Mixed precision supported via `use_amp: true`
- CPU/GPU auto-detection