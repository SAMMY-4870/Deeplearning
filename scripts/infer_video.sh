#!/usr/bin/env bash
set -euo pipefail
python -m src.infer --config configs/video_cnn_lstm.yaml --path "$1" --checkpoint "$2"