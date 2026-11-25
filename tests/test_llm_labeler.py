"""Unit tests untuk src/pipeline/llm_labeler.py."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.pipeline.llm_labeler import HFLLMLabeler, _extract_json_array, anonymize_for_llm


class _FakeClient:
    def __init__(self, responses: list[str], fail_times: int = 0) -> None:
        self.responses = responses
        self.fail_times = fail_times
        self.calls = 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("Transient API error")
        content = self.responses.pop(0) if self.responses else "[]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_anonymize_for_llm():
    raw = "Hubungi @bintang_12 atau 081234567890 dan email test@undiksha.ac.id"
    anon = anonymize_for_llm(raw)
    assert "@bintang_12" not in anon
    assert "@pengguna" in anon
    assert "081234567890" not in anon
    assert "[TELEPON]" in anon
    assert "test@undiksha.ac.id" not in anon
    assert "[EMAIL]" in anon


def test_extract_json_array_with_markdown_fence():
    raw = '```json\n[{"idx": 0, "is_spam": false, "sentiment": "negatif", "aspect": "layanan_administrasi", "confidence": 0.9}]\n```'
    parsed = _extract_json_array(raw)
    assert len(parsed) == 1
    assert parsed[0]["sentiment"] == "negatif"


def test_llm_labeler_batch_and_cache(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    payload = json.dumps([
        {
            "idx": 0,
            "is_spam": False,
            "sentiment": "negatif",
            "aspect": "layanan_administrasi",
            "confidence": 0.94,
            "reason": "SIAKAD bermasalah",
        },
        {
            "idx": 1,
            "is_spam": True,
            "sentiment": "netral",
            "aspect": "lainnya",
            "confidence": 0.98,
            "reason": "Iklan kost",
        },
    ])
    fake = _FakeClient([payload])
    labeler = HFLLMLabeler(cache_file=cache_file, client=fake, batch_size=10, token="hf_test")

    texts = ["SIAKAD error terus saat KRS", "DIJUAL KOST DEKAT UNDIKSHA WA 08123456789"]
    res1 = labeler.label_texts(texts, use_cache=True)
    assert res1[0]["sentiment"] == "negatif"
    assert res1[0]["aspect"] == "layanan_administrasi"
    assert res1[0]["is_spam"] is False
    assert res1[1]["is_spam"] is True
    assert fake.calls == 1

    # Panggilan kedua harus hit cache tanpa memanggil client lagi
    res2 = labeler.label_texts(texts, use_cache=True)
    assert res2 == res1
    assert fake.calls == 1


def test_llm_labeler_fallback_when_api_fails(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    fake = _FakeClient([], fail_times=10)
    labeler = HFLLMLabeler(cache_file=cache_file, client=fake, max_retries=1, token="hf_test")

    res = labeler.label_texts(["SIAKAD Undiksha sangat lambat dan mengecewakan"], use_cache=False)
    assert len(res) == 1
    assert res[0]["source"] == "fallback_lexicon"
    assert res[0]["aspect"] == "layanan_administrasi"

