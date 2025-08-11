#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/video_cnn_lstm.yaml DATA_ROOT=${1:-./data/videos}