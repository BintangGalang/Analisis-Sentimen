"""
src/utils/__init__.py
=====================
Package utilitas terpusat:
- load_config, set_seed, setup_logging (Core utils)
- logger, console, setup_logger (Rich logging)
- random_delay, parse_numeric_metric, read_jsonl, append_to_jsonl (Helper I/O)
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .logger import logger, console, setup_logger
from .helpers import (
    load_config as helper_load_config,
    random_delay,
    parse_numeric_metric,
    read_jsonl,
    append_to_jsonl
)

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Memuat konfigurasi proyek dari file YAML (configs/config.yaml atau config/config.yaml)."""
    return helper_load_config(str(config_path) if config_path else None)


def set_seed(seed: int = 42) -> None:
    """Mengatur seed global untuk seluruh generator bilangan acak agar reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(config: dict[str, Any] | None = None) -> None:
    """Mengatur konfigurasi logging global."""
    log_cfg = (config or {}).get("logging", {})
    level_str: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_str, logging.INFO)
    fmt: str = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.basicConfig(level=level, format=fmt)


__all__ = [
    "load_config",
    "set_seed",
    "setup_logging",
    "logger",
    "console",
    "setup_logger",
    "random_delay",
    "parse_numeric_metric",
    "read_jsonl",
    "append_to_jsonl"
]

