"""
src/pipeline/llm_labeler.py
---------------------------
Modul pelabelan sentimen, aspek reputasi Undiksha, dan deteksi spam
menggunakan LLM Hugging Face (InferenceClient) dengan batching,
structured JSON output, caching idempoten, dan fallback lokal aman.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils.logger import logger

VALID_SENTIMENTS = {"positif", "netral", "negatif"}
VALID_ASPECTS = {
    "akademik",
    "penerimaan_biaya",
    "fasilitas",
    "layanan_administrasi",
    "kemahasiswaan",
    "lulusan_karier",
    "isu_citra",
    "lainnya",
}

SYSTEM_PROMPT = """Kamu adalah analis reputasi kampus untuk Universitas Pendidikan Ganesha (Undiksha), Bali.
Tugasmu adalah melabeli sekumpulan komentar/postingan media sosial secara objektif dan konsisten dalam format JSON array.

Untuk setiap item dengan indeks `idx`, tentukan:
1. `is_spam` (boolean):
   - `true` jika iklan kost/kontrakan, jual tanah/rumah/barang, pinjol, judi online, promosi jasa komersial, atau spam acak yang bukan opini/berita tentang Undiksha.
   - `false` jika opini, pertanyaan, pujian, keluhan, atau berita terkait Undiksha.
2. `sentiment` (string, wajib salah satu: "positif", "netral", "negatif"):
   - Pahami konteks sarkasme/sindiran (misal: "keren banget siakadnya tiap KRS selalu down" -> "negatif").
   - Pahami bahasa gaul/singkatan Indonesia & Bali (misal: "astungkara", "sing ngidang", "keren min", "kecewa bgt").
   - Berita informatif atau pertanyaan netral tanpa pujian/keluhan -> "netral".
3. `aspect` (string, wajib salah satu):
   - "akademik": dosen, perkuliahan, KRS, skripsi, KKN, akreditasi, kurikulum, ujian.
   - "penerimaan_biaya": UKT, SNBP, SNBT, jalur mandiri, beasiswa, KIP-K, PMB.
   - "fasilitas": gedung, wifi, lab, asrama, perpustakaan, parkir, lingkungan kampus Singaraja/Denpasar.
   - "layanan_administrasi": SIAKAD, sistem IT, birokrasi, admin, surat-menyurat, respons layanan.
   - "kemahasiswaan": BEM, UKM, ormawa, lomba, prestasi mahasiswa, PKKMB/ospek.
   - "lulusan_karier": alumni, magang, MSIB, CPNS, PPPK, lowongan kerja.
   - "isu_citra": nama baik kampus, berita viral, Dies Natalis, reputasi institusi.
   - "lainnya": komentar sangat pendek/emoji umum tanpa topik spesifik.
4. `confidence` (float 0.0 - 1.0): tingkat keyakinanmu.
5. `reason` (string singkat <= 12 kata): alasan singkat.

WAJIB mengembalikan HANYA JSON array valid tanpa teks tambahan di luar JSON."""


def anonymize_for_llm(text: str) -> str:
    """Sensor data sensitif (email, @username, nomor telepon/WA) sebelum dikirim ke API eksternal."""
    t = str(text or "").strip()
    t = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", t)
    t = re.sub(r"@[A-Za-z0-9._]+", "@pengguna", t)
    t = re.sub(r"(\+62|62|08)[0-9\-\s]{7,15}", "[TELEPON]", t)
    return t


def _text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def _extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    """Ekstrak JSON array dari output LLM meskipun dibungkus markdown code fence."""
    cleaned = str(raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            return parsed["items"]
    except Exception:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError("Respons LLM tidak mengandung JSON array yang valid.")


class HFLLMLabeler:
    """Pelabel sentimen, aspek, dan spam berbasis LLM (Gemini API & Hugging Face InferenceClient)."""

    def __init__(
        self,
        provider: str | None = None,
        gemini_model_id: str | None = None,
        gemini_fallback_model_id: str | None = None,
        gemini_api_key: str | None = None,
        model_id: str | None = None,
        fallback_model_id: str | None = None,
        batch_size: int = 20,
        max_retries: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        cache_file: str | Path = "data/interim/llm_label_cache.jsonl",
        token: str | None = None,
        client: Any = None,
    ) -> None:
        load_dotenv()
        self.gemini_api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.gemini_model_id = gemini_model_id or "gemini-flash-lite-latest"
        self.gemini_fallback_model_id = gemini_fallback_model_id or "gemini-3.8-flash"

        self.token = (token or os.getenv("HF_TOKEN", "")).strip()
        self.model_id = model_id or "Qwen/Qwen2.5-72B-Instruct"
        self.fallback_model_id = fallback_model_id or "Qwen/Qwen2.5-7B-Instruct"

        # Pilih provider otomatis: jika client disuntikkan (unit test) pakai huggingface, jika tidak utamakan gemini bila key tersedia
        if client is not None:
            self.provider = provider or "huggingface"
        elif provider == "gemini" and self.gemini_api_key:
            self.provider = "gemini"
        elif self.gemini_api_key and provider in (None, "auto", "gemini"):
            self.provider = "gemini"
        else:
            self.provider = provider or "huggingface"

        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.cache_path = Path(cache_file)
        self.cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

        self._client = client
        if self._client is None and self.provider == "huggingface" and self.token:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.token)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "HFLLMLabeler":
        cfg = (config or {}).get("llm_labeling", {})
        return cls(
            provider=cfg.get("provider"),
            gemini_model_id=cfg.get("gemini_model_id", "gemini-flash-lite-latest"),
            gemini_fallback_model_id=cfg.get("gemini_fallback_model_id", "gemini-3.8-flash"),
            model_id=cfg.get("model_id", "Qwen/Qwen2.5-72B-Instruct"),
            fallback_model_id=cfg.get("fallback_model_id", "Qwen/Qwen2.5-7B-Instruct"),
            batch_size=cfg.get("batch_size", 20),
            max_retries=cfg.get("max_retries", 3),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 4096),
            cache_file=cfg.get("cache_file", "data/interim/llm_label_cache.jsonl"),
        )

    def is_available(self) -> bool:
        if self.provider == "gemini" and self.gemini_api_key:
            return True
        return self._client is not None

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    h = rec.get("hash")
                    if h:
                        self.cache[h] = rec
        except Exception as exc:
            logger.warning("Gagal memuat cache LLM %s: %s", self.cache_path, exc)

    def _append_cache(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _fallback_single(self, text: str) -> dict[str, Any]:
        """Fallback lokal aman jika API Hugging Face tidak tersedia atau gagal."""
        t_low = str(text or "").lower()
        is_spam = bool(re.search(
            r"di\s*jual|dijual|tanah\s+kavling|terima\s+kost|kost\s+exclusive|kos\s+exclusive|"
            r"wa\.me/|hubungi\s+wa|slot\s+gacor|judi\s+online|pinjol",
            t_low,
        ))
        sentiment = "netral"
        confidence = 0.65
        try:
            from src.pipeline.labeler import LexiconLabeler
            res = LexiconLabeler().analyze_text(text)
            sentiment = str(res.get("sentiment", "netral")).lower()
            confidence = float(res.get("confidence", 0.65))
        except Exception:
            pass

        aspect = "lainnya"
        if any(k in t_low for k in ("siakad", "admin", "birokrasi", "pelayanan", "surat", "krs")):
            aspect = "layanan_administrasi"
        elif any(k in t_low for k in ("ukt", "biaya", "snbp", "snbt", "beasiswa", "kip", "pmb")):
            aspect = "penerimaan_biaya"
        elif any(k in t_low for k in ("dosen", "kuliah", "skripsi", "kkn", "akreditasi", "ujian", "nilai")):
            aspect = "akademik"
        elif any(k in t_low for k in ("gedung", "wifi", "lab", "asrama", "parkir", "perpustakaan", "kampus")):
            aspect = "fasilitas"
        elif any(k in t_low for k in ("bem", "ukm", "ormawa", "lomba", "ospek", "pkkmb")):
            aspect = "kemahasiswaan"
        elif any(k in t_low for k in ("alumni", "magang", "msib", "cpns", "pppk", "loker")):
            aspect = "lulusan_karier"
        elif any(k in t_low for k in ("undiksha", "reputasi", "dies natalis", "bangga", "viral")):
            aspect = "isu_citra"

        return {
            "is_spam": is_spam,
            "sentiment": sentiment if sentiment in VALID_SENTIMENTS else "netral",
            "aspect": aspect,
            "confidence": round(confidence, 4),
            "reason": "fallback_lokal",
            "source": "fallback_lexicon",
        }

    def _call_gemini_model(self, model_name: str, user_msg: str) -> str:
        """Panggil Gemini REST API (generateContent) dengan output JSON terstruktur."""
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, params={"key": self.gemini_api_key}, json=body, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:250]}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini tidak mengembalikan candidates: {str(data)[:200]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(p.get("text", "")) for p in parts)

    def _call_llm_batch(self, items: list[tuple[int, str]]) -> list[dict[str, Any]]:
        """Kirim 1 batch (idx, teks_anonim) ke Gemini API atau Hugging Face InferenceClient."""
        payload = [{"idx": idx, "text": anonymize_for_llm(txt)[:500]} for idx, txt in items]
        user_msg = (
            "Labeli seluruh item berikut dan kembalikan JSON array dengan kunci "
            "`idx`, `is_spam`, `sentiment`, `aspect`, `confidence`, `reason`:\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        if self.provider == "gemini":
            models_to_try = [self.gemini_model_id]
            if self.gemini_fallback_model_id and self.gemini_fallback_model_id != self.gemini_model_id:
                models_to_try.append(self.gemini_fallback_model_id)
        else:
            models_to_try = [self.model_id]
            if self.fallback_model_id and self.fallback_model_id != self.model_id:
                models_to_try.append(self.fallback_model_id)

        last_err: Exception | None = None
        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    if self.provider == "gemini":
                        content = self._call_gemini_model(model_name, user_msg)
                    else:
                        resp = self._client.chat_completion(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        content = resp.choices[0].message.content
                    parsed = _extract_json_array(content)
                    for item in parsed:
                        item["_model"] = model_name
                    return parsed
                except Exception as exc:
                    last_err = exc
                    wait_s = min(8.0, 1.5 * (attempt + 1))
                    logger.warning(
                        "LLM %s gagal (percobaan %d/%d): %s",
                        model_name, attempt + 1, self.max_retries, exc,
                    )
                    time.sleep(wait_s)

        raise RuntimeError(f"Semua percobaan LLM gagal: {last_err}")

    def label_texts(self, texts: list[str], use_cache: bool = True) -> list[dict[str, Any]]:
        """Melabeli daftar teks dengan batching, caching, dan fallback otomatis."""
        results: list[dict[str, Any] | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []

        for i, txt in enumerate(texts):
            h = _text_hash(txt)
            if use_cache and h in self.cache:
                cached = dict(self.cache[h])
                cached.pop("hash", None)
                results[i] = cached
            else:
                pending.append((i, txt, h))

        if not pending:
            return [r or self._fallback_single(texts[i]) for i, r in enumerate(results)]

        if not self.is_available():
            logger.warning("HF_TOKEN tidak tersedia; menggunakan fallback lokal untuk %d teks.", len(pending))
            for i, txt, _ in pending:
                results[i] = self._fallback_single(txt)
            return [r for r in results if r is not None]

        total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
        for batch_num, b_start in enumerate(range(0, len(pending), self.batch_size), start=1):
            batch = pending[b_start : b_start + self.batch_size]
            batch_items = [(local_idx, txt) for local_idx, (_, txt, _) in enumerate(batch)]
            new_cache_records: list[dict[str, Any]] = []

            try:
                llm_out = self._call_llm_batch(batch_items)
                by_idx = {int(row.get("idx", -1)): row for row in llm_out if isinstance(row, dict)}

                for local_idx, (orig_idx, orig_txt, h) in enumerate(batch):
                    raw = by_idx.get(local_idx)
                    if not raw:
                        results[orig_idx] = self._fallback_single(orig_txt)
                        continue

                    sent = str(raw.get("sentiment", "netral")).strip().lower()
                    if sent not in VALID_SENTIMENTS:
                        sent = "netral"

                    asp = str(raw.get("aspect", "lainnya")).strip().lower().replace(" ", "_")
                    if asp not in VALID_ASPECTS:
                        asp = "lainnya"

                    try:
                        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.85))))
                    except Exception:
                        conf = 0.85

                    record = {
                        "is_spam": bool(raw.get("is_spam", False)),
                        "sentiment": sent,
                        "aspect": asp,
                        "confidence": round(conf, 4),
                        "reason": str(raw.get("reason", ""))[:120],
                        "source": f"llm:{raw.get('_model', self.model_id)}",
                    }
                    results[orig_idx] = record
                    if use_cache:
                        self.cache[h] = {"hash": h, **record}
                        new_cache_records.append({"hash": h, **record})
                logger.info("Progress LLM: batch %d/%d selesai (%d/%d teks)", batch_num, total_batches, min(b_start + len(batch), len(pending)), len(pending))
            except Exception as exc:
                logger.warning("Batch LLM gagal (%s); fallback lokal untuk %d teks.", exc, len(batch))
                for orig_idx, orig_txt, _ in batch:
                    results[orig_idx] = self._fallback_single(orig_txt)

            if new_cache_records:
                self._append_cache(new_cache_records)
            if batch_num < total_batches:
                time.sleep(0.5)

        return [r if r is not None else self._fallback_single(texts[i]) for i, r in enumerate(results)]

