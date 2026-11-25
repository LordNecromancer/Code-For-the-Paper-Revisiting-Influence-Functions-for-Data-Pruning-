"""Small shared helpers: seeding, config loading, JSON IO, device selection."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply `a.b.c=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)  # atomic: a crashed run never leaves a half-written result


def load_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def deep_copy(d: dict) -> dict:
    return copy.deepcopy(d)


def run_paths(run_dir: str | os.PathLike, seed: int) -> dict[str, Path]:
    """Canonical on-disk layout for one experiment family."""
    run = Path(run_dir)
    sd = run / f"seed{seed}"
    return {
        "run": run,
        "seed": sd,
        "score": sd / "score",
        "subsets": sd / "subsets",
        "results": run / "results",
    }
