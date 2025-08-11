#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/image_xception.yaml DATA_ROOT=${1:-./data/images}