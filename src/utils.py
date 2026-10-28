"""
src/utils.py
============
Fungsi-fungsi utilitas inti yang digunakan di seluruh proyek.

Berisi:
    - load_config  : Memuat konfigurasi dari file YAML.
    - setup_logging: Mengatur logger global berdasarkan config.
    - set_seed     : Mengatur seed global untuk reproducibility
                     (random, numpy, torch, transformers).

Penggunaan:
    from src.utils import load_config, set_seed, setup_logging

    config = load_config("configs/config.yaml")
    setup_logging(config)
    set_seed(config["seed"])
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Loader Konfigurasi
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Memuat konfigurasi proyek dari file YAML.

    Semua parameter (path, seed, hyperparameter) wajib dibaca dari
    config; tidak ada nilai yang di-hardcode di modul lain.

    Args:
        config_path: Path ke file YAML. Default ``configs/config.yaml``.

    Returns:
        Dict berisi seluruh konfigurasi proyek.

    Raises:
        FileNotFoundError: Jika file config tidak ditemukan.
        yaml.YAMLError: Jika format YAML tidak valid.

    Contoh::

        config = load_config("configs/config.yaml")
        seed   = config["seed"]           # 42
        raw    = config["paths"]["raw_dir"]
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {config_path.resolve()}\n"
            "Pastikan Anda menjalankan skrip dari root proyek."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.debug("Konfigurasi berhasil dimuat dari: %s", config_path.resolve())
    return config


# ─────────────────────────────────────────────────────────────
# 2. Setup Logging
# ─────────────────────────────────────────────────────────────

def setup_logging(config: dict[str, Any] | None = None) -> None:
    """Mengatur konfigurasi logging global dari config.

    Jika ``config["logging"]["to_file"]`` bernilai True, log juga
    ditulis ke file di ``config["paths"]["logs_dir"]/run.log``.

    Args:
        config: Dict konfigurasi dari :func:`load_config`.
                Jika None, menggunakan level INFO ke konsol saja.

    Contoh::

        config = load_config()
        setup_logging(config)
    """
    log_cfg = (config or {}).get("logging", {})
    level_str: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_str, logging.INFO)
    fmt: str = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    formatter = logging.Formatter(fmt)

    # Console handler — paksa UTF-8 agar emoji/Unicode tidak crash di Windows
    import sys, io  # noqa: PLC0415, E401
    try:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        # Fallback jika stdout tidak punya .buffer (mis. saat di-redirect)
        stream = sys.stdout
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [console_handler]

    if config and log_cfg.get("to_file", False):
        logs_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logger.debug("Logging dikonfigurasi pada level %s.", level_str)



# ─────────────────────────────────────────────────────────────
# 3. Set Seed Global
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Mengatur seed global untuk reproducibility penuh.

    Menetapkan seed pada:
      * ``random`` (stdlib Python)
      * ``numpy``
      * ``torch`` (CPU & semua GPU)
      * ``transformers`` (via os.environ ``PYTHONHASHSEED``)

    Args:
        seed: Nilai seed. Default 42 (sesuai ``configs/config.yaml``).

    Catatan:
        Untuk reproducibility penuh pada CUDA, operasi non-deterministik
        dinonaktifkan (``torch.backends.cudnn.deterministic = True``).
        Ini bisa sedikit memperlambat training GPU.

    Contoh::

        from src.utils import load_config, set_seed
        config = load_config()
        set_seed(config["seed"])
    """
    # Python stdlib
    random.seed(seed)

    # Environment (memengaruhi hash Python dan beberapa library C)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (impor bersyarat agar proyek tetap bisa jalan tanpa GPU/torch)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning(
            "PyTorch tidak ditemukan; seed torch tidak di-set. "
            "Install torch jika akan fine-tuning IndoBERT."
        )

    # Transformers (HuggingFace) menggunakan seed dari torch & numpy secara otomatis,
    # tetapi kita set juga environment variable sebagai jaga-jaga.
    os.environ["TRANSFORMERS_SEED"] = str(seed)

    logger.info("Seed global ditetapkan: %d", seed)
