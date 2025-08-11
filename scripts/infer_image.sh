#!/usr/bin/env bash
set -euo pipefail
python -m src.infer --config configs/image_xception.yaml --path "$1" --checkpoint "$2"