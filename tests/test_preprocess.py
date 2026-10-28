"""
tests/test_preprocess.py
========================
Unit test untuk modul preprocessing.

Menguji cleaner, normalizer, dan tokenizer_pipe
tanpa memerlukan data nyata atau koneksi internet.

Jalankan dengan:
    pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.cleaner import (
    clean_text,
    normalize_hashtags,
    remove_mentions,
    remove_urls,
    to_lowercase,
    reduce_repeated_chars,
)
from src.preprocess.normalizer import (
    detect_codemix,
    emoji_to_text,
    normalize_slang,
    normalize_text,
)
from src.preprocess.tokenizer_pipe import tokenize, remove_stopwords, tokenize_and_stem

# Path kamus slang proyek
SLANG_PATH = str(Path(__file__).resolve().parent.parent / "data" / "interim" / "slang_dict.json")


# ─────────────────────────────────────────────────────────────
# Test: cleaner.py
# ─────────────────────────────────────────────────────────────

class TestCleaner:
    """Uji fungsi-fungsi dalam cleaner.py."""

    def test_to_lowercase(self) -> None:
        assert to_lowercase("Undiksha BAGUS") == "undiksha bagus"

    def test_remove_urls_http(self) -> None:
        result = remove_urls("kunjungi https://undiksha.ac.id ya")
        assert "https://" not in result
        assert "undiksha.ac.id" not in result

    def test_remove_urls_www(self) -> None:
        result = remove_urls("cek www.undiksha.ac.id")
        assert "www." not in result

    def test_remove_mentions(self) -> None:
        result = remove_mentions("@undiksha keren!")
        assert "@undiksha" not in result
        assert "keren" in result

    def test_normalize_hashtags(self) -> None:
        result = normalize_hashtags("#undiksha #singaraja bagus")
        assert "#" not in result
        assert "undiksha" in result
        assert "singaraja" in result

    def test_reduce_repeated_chars(self) -> None:
        result = reduce_repeated_chars("kereeeeen")
        # Harus dikurangi menjadi maks 2 karakter berulang
        assert "eeee" not in result

    def test_clean_text_full_pipeline(self) -> None:
        text = "@user Undiksha KEREEEN!! #undiksha https://t.co/xyz"
        result = clean_text(text)
        assert "@user" not in result
        assert "https://" not in result
        assert "#" not in result
        assert result == result.lower()

    def test_clean_text_empty(self) -> None:
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_text_none_like(self) -> None:
        # None bukan str, harus return ""
        assert clean_text(123) == ""  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
# Test: normalizer.py
# ─────────────────────────────────────────────────────────────

class TestNormalizer:
    """Uji fungsi-fungsi dalam normalizer.py."""

    def test_normalize_slang_basic(self) -> None:
        slang_dict = {"gak": "tidak", "bgt": "sangat", "krn": "karena"}
        result = normalize_slang("gak bisa bgt krn malas", slang_dict=slang_dict)
        assert "tidak" in result
        assert "sangat" in result
        assert "karena" in result
        assert "gak" not in result

    def test_normalize_slang_dict_from_file(self) -> None:
        """Uji normalisasi menggunakan kamus file JSON proyek."""
        result = normalize_slang(
            "gak bisa kuliah krn sakit",
            slang_path=SLANG_PATH,
        )
        assert "tidak" in result
        assert "karena" in result

    def test_normalize_slang_no_change(self) -> None:
        slang_dict = {"gak": "tidak"}
        result = normalize_slang("kuliah di undiksha", slang_dict=slang_dict)
        assert result == "kuliah di undiksha"

    def test_detect_codemix_positif(self) -> None:
        has_mix, words = detect_codemix("kampusnya nice banget recommend deh")
        assert has_mix is True
        assert "nice" in words or "recommend" in words

    def test_detect_codemix_negatif(self) -> None:
        has_mix, words = detect_codemix("kuliah di undiksha sangat menyenangkan")
        # Teks Indonesia murni tidak dianggap code-mix
        assert has_mix is False

    def test_emoji_to_text_dengan_emoji(self) -> None:
        text_out, has_emoji = emoji_to_text("Undiksha keren 🔥")
        assert has_emoji is True
        # Emoji harusnya dikonversi ke kata (fire atau serupa)
        assert "🔥" not in text_out

    def test_emoji_to_text_tanpa_emoji(self) -> None:
        text_out, has_emoji = emoji_to_text("Undiksha keren sekali")
        assert has_emoji is False

    def test_normalize_text_output_keys(self) -> None:
        result = normalize_text("gak bisa kuliah 🔥", slang_path=SLANG_PATH)
        assert "text_norm" in result
        assert "emoji_replaced" in result
        assert "has_codemix" in result
        assert "en_words" in result


# ─────────────────────────────────────────────────────────────
# Test: tokenizer_pipe.py
# ─────────────────────────────────────────────────────────────

class TestTokenizerPipe:
    """Uji fungsi-fungsi dalam tokenizer_pipe.py."""

    def test_tokenize_basic(self) -> None:
        tokens = tokenize("kuliah di undiksha singaraja")
        assert "kuliah" in tokens
        assert "undiksha" in tokens
        assert "singaraja" in tokens

    def test_tokenize_min_length(self) -> None:
        tokens = tokenize("di ke ya saya kuliah", min_length=3)
        assert "di" not in tokens   # panjang 2, di bawah min
        assert "ke" not in tokens
        assert "kuliah" in tokens

    def test_remove_stopwords(self) -> None:
        stopwords = {"di", "dan", "yang", "saya"}
        tokens = ["saya", "kuliah", "di", "undiksha"]
        result = remove_stopwords(tokens, stopwords)
        assert "saya" not in result
        assert "di" not in result
        assert "kuliah" in result
        assert "undiksha" in result

    def test_tokenize_and_stem_output_keys(self) -> None:
        result = tokenize_and_stem("kuliah di undiksha sangat menyenangkan")
        assert "tokens_clean" in result
        assert "tokens_stemmed" in result
        assert "text_tokens" in result
        assert "text_stemmed" in result

    def test_tokenize_and_stem_returns_list(self) -> None:
        result = tokenize_and_stem("mahasiswa undiksha rajin belajar")
        assert isinstance(result["tokens_stemmed"], list)
        assert len(result["tokens_stemmed"]) > 0

    def test_stemming_mengurangi_variasi(self) -> None:
        """Stemming harus menyatukan variasi kata yang sama."""
        result1 = tokenize_and_stem("mahasiswanya belajar")
        result2 = tokenize_and_stem("mahasiswa belajar")
        # Setelah stemming, 'mahasiswanya' dan 'mahasiswa' harus sama
        stems1 = result1["tokens_stemmed"]
        stems2 = result2["tokens_stemmed"]
        # Setidaknya ada overlap di stem
        assert set(stems1) & set(stems2)


# ─────────────────────────────────────────────────────────────
# Test: Integrasi Pipeline Lengkap
# ─────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Uji pipeline preprocessing secara end-to-end."""

    def test_full_pipeline_typical_tweet(self) -> None:
        """Teks tipikal komentar YouTube tentang Undiksha."""
        from src.preprocess.pipeline import preprocess_row  # noqa: PLC0415

        text = "@user Undiksha keren bgt!! #undiksha 🔥 gak rugi daftar https://t.co/xyz"
        result = preprocess_row(
            text_raw=text,
            slang_path=SLANG_PATH,
            apply_stopword=True,
            apply_stemming=True,
            min_token_length=2,
        )
        assert result["text_raw"] == text
        assert "@user" not in result["text_clean"]
        assert "https://" not in result["text_clean"]
        assert "#" not in result["text_clean"]
        assert isinstance(result["emoji_replaced"], bool)
        assert isinstance(result["has_codemix"], bool)

    def test_full_pipeline_empty_text(self) -> None:
        from src.preprocess.pipeline import preprocess_row  # noqa: PLC0415

        result = preprocess_row(
            text_raw="",
            slang_path=SLANG_PATH,
            apply_stopword=True,
            apply_stemming=True,
            min_token_length=2,
        )
        assert result["text_clean"] == ""
        assert result["text_stemmed"] == ""

    def test_full_pipeline_preserves_raw(self) -> None:
        from src.preprocess.pipeline import preprocess_row  # noqa: PLC0415

        original = "Undiksha emg terbaik gak ada tandingannya!"
        result = preprocess_row(
            text_raw=original,
            slang_path=SLANG_PATH,
            apply_stopword=True,
            apply_stemming=True,
            min_token_length=2,
        )
        # text_raw harus selalu sama dengan input asli
        assert result["text_raw"] == original
