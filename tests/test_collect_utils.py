"""
tests/test_collect_utils.py
===========================
Unit test untuk src/collect/utils_collect.py.

Menguji fungsi-fungsi helper tanpa memanggil API eksternal.

Jalankan dengan:
    pytest tests/test_collect_utils.py -v
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect.utils_collect import (
    anonymize_id,
    build_standard_row,
    flag_spam_promo,
    is_official_account,
    save_raw,
    RateLimiter,
    STANDARD_COLUMNS,
)


# ─────────────────────────────────────────────────────────────
# Test: anonymize_id
# ─────────────────────────────────────────────────────────────

class TestAnonymizeId:
    def test_menghasilkan_string_64_karakter(self) -> None:
        result = anonymize_id("user123")
        assert len(result) == 64

    def test_id_sama_menghasilkan_hash_sama(self) -> None:
        assert anonymize_id("abc") == anonymize_id("abc")

    def test_id_berbeda_menghasilkan_hash_berbeda(self) -> None:
        assert anonymize_id("user1") != anonymize_id("user2")

    def test_menerima_integer(self) -> None:
        result = anonymize_id(12345)
        assert len(result) == 64


# ─────────────────────────────────────────────────────────────
# Test: is_official_account
# ─────────────────────────────────────────────────────────────

class TestIsOfficialAccount:
    def test_akun_resmi_terdeteksi(self) -> None:
        assert is_official_account("undiksha") is True
        assert is_official_account("humas.undiksha") is True

    def test_akun_publik_tidak_terdeteksi(self) -> None:
        assert is_official_account("mahasiswa_undiksha_2024") is False
        assert is_official_account("random_user") is False

    def test_case_insensitive(self) -> None:
        assert is_official_account("UNDIKSHA") is True


# ─────────────────────────────────────────────────────────────
# Test: flag_spam_promo
# ─────────────────────────────────────────────────────────────

class TestFlagSpamPromo:
    def test_promosi_terdeteksi(self) -> None:
        assert flag_spam_promo("Penerimaan Mahasiswa Baru 2025 dibuka!") is True
        assert flag_spam_promo("daftar sekarang dan raih beasiswamu") is True

    def test_opini_tidak_terdeteksi(self) -> None:
        assert flag_spam_promo("Dosen di Undiksha sangat helpful") is False

    def test_teks_kosong(self) -> None:
        assert flag_spam_promo("") is False


# ─────────────────────────────────────────────────────────────
# Test: build_standard_row
# ─────────────────────────────────────────────────────────────

class TestBuildStandardRow:
    def test_kunci_wajib_ada(self) -> None:
        row = build_standard_row(
            raw_id="123",
            platform="youtube",
            text="Kampus Undiksha bagus",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        for col in STANDARD_COLUMNS:
            assert col in row, f"Kolom '{col}' tidak ada di row."

    def test_id_dianonimkan(self) -> None:
        row = build_standard_row("secret_id", "youtube", "teks", None)
        assert row["id_anon"] != "secret_id"
        assert len(row["id_anon"]) == 64

    def test_teks_di_strip(self) -> None:
        row = build_standard_row("1", "youtube", "  spasi  ", None)
        assert row["text"] == "spasi"


# ─────────────────────────────────────────────────────────────
# Test: save_raw
# ─────────────────────────────────────────────────────────────

class TestSaveRaw:
    def test_menyimpan_file_csv(self, tmp_path: Path) -> None:
        df = pd.DataFrame([
            build_standard_row("1", "youtube", "Teks uji", None)
        ])
        out = save_raw(df, "youtube", tmp_path)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 1

    def test_nama_file_mengandung_platform(self, tmp_path: Path) -> None:
        df = pd.DataFrame([build_standard_row("1", "instagram", "Teks", None)])
        out = save_raw(df, "instagram", tmp_path)
        assert "instagram" in out.name


# ─────────────────────────────────────────────────────────────
# Test: RateLimiter
# ─────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_jeda_minimum_diterapkan(self) -> None:
        limiter = RateLimiter(min_interval=0.1)
        t0 = time.monotonic()
        with limiter:
            pass
        with limiter:
            pass
        elapsed = time.monotonic() - t0
        # Harus ada jeda minimal ~0.1 detik antara dua request
        assert elapsed >= 0.08, f"Jeda terlalu singkat: {elapsed:.3f}s"
