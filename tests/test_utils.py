"""
tests/test_utils.py
===================
Unit test dasar untuk src/utils.py.

Jalankan dengan:
    pytest tests/test_utils.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Pastikan root proyek ada di sys.path saat pytest dijalankan dari root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed


# ─────────────────────────────────────────────────────────────
# Fixture: path konfigurasi yang valid
# ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


# ─────────────────────────────────────────────────────────────
# Test: load_config
# ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    """Uji fungsi load_config."""

    def test_mengembalikan_dict(self) -> None:
        """load_config harus mengembalikan dict."""
        config = load_config(CONFIG_PATH)
        assert isinstance(config, dict), "Hasil load_config harus berupa dict."

    def test_kunci_wajib_ada(self) -> None:
        """Kunci-kunci wajib harus ada di config."""
        config = load_config(CONFIG_PATH)
        kunci_wajib = ["seed", "paths", "split", "labels", "bert", "ensemble"]
        for kunci in kunci_wajib:
            assert kunci in config, f"Kunci '{kunci}' tidak ditemukan di config."

    def test_seed_bernilai_42(self) -> None:
        """Seed default harus 42 sesuai project-rules."""
        config = load_config(CONFIG_PATH)
        assert config["seed"] == 42, "Seed harus 42."

    def test_file_tidak_ditemukan(self, tmp_path: Path) -> None:
        """FileNotFoundError harus dilempar jika file tidak ada."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "tidak_ada.yaml")


# ─────────────────────────────────────────────────────────────
# Test: set_seed
# ─────────────────────────────────────────────────────────────

class TestSetSeed:
    """Uji fungsi set_seed."""

    def test_reproducibility_numpy(self) -> None:
        """Dua pemanggilan set_seed dengan seed sama harus menghasilkan array identik."""
        set_seed(42)
        arr1 = np.random.rand(10)

        set_seed(42)
        arr2 = np.random.rand(10)

        np.testing.assert_array_equal(arr1, arr2,
            err_msg="Array numpy tidak identik setelah reset seed.")

    def test_reproducibility_random(self) -> None:
        """Modul random Python harus reproducible setelah set_seed."""
        import random

        set_seed(42)
        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        assert val1 == val2, "Nilai random.random() tidak identik setelah reset seed."

    def test_pythonhashseed_diset(self) -> None:
        """PYTHONHASHSEED harus di-set ke nilai seed."""
        set_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_seed_berbeda_menghasilkan_array_berbeda(self) -> None:
        """Seed yang berbeda harus menghasilkan array berbeda."""
        set_seed(1)
        arr1 = np.random.rand(10)

        set_seed(2)
        arr2 = np.random.rand(10)

        assert not np.array_equal(arr1, arr2), \
            "Seed berbeda seharusnya menghasilkan array berbeda."
