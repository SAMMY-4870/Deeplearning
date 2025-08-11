import argparse
import os
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override key=val pairs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Parse KEY=VALUE overrides
    for item in args.overrides:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        set_deep(cfg, key, parse_value(val))

    # Normalize output dir
    if "output_dir" in cfg:
        cfg["output_dir"] = os.path.abspath(cfg["output_dir"])
        os.makedirs(cfg["output_dir"], exist_ok=True)

    # Normalize data root
    if "DATA_ROOT" in cfg:
        cfg["DATA_ROOT"] = os.path.abspath(cfg["DATA_ROOT"])  # noqa: N816

    return cfg


def set_deep(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_value(v: str) -> Any:
    # Try to parse YAML-style scalars
    try:
        return yaml.safe_load(v)
    except Exception:
        return v