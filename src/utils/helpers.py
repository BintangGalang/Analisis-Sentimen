"""
Fungsi-fungsi pembantu umum (helper utilities) untuk pipeline data,
I/O JSONL, parsing metrik numerik, dan pemuatan konfigurasi.
"""

import os
import re
import json
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Membaca konfigurasi YAML.
    Jika config_path diberikan, memuat file tersebut.
    Jika None, mencari di configs/config.yaml atau config/config.yaml.
    """
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    candidates = [
        "configs/config.yaml",
        "config/config.yaml"
    ]
    for cp in candidates:
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"File konfigurasi default tidak ditemukan pada: {candidates}\n"
        "Pastikan Anda menjalankan skrip dari root repositori."
    )


async def random_delay(min_sec: float = 1.5, max_sec: float = 3.5) -> None:
    """Melakukan asynchronous sleep dengan durasi acak agar aksi scraper terlihat natural."""
    delay = random.uniform(min_sec, max_sec)
    await asyncio.sleep(delay)


def parse_numeric_metric(text: Any) -> int:
    """
    Mengubah teks angka medsos ke integer.
    Contoh: '1.2K' -> 1200, '3,5 rb' -> 3500, '1.5M' -> 1500000.
    """
    if not text:
        return 0
    if isinstance(text, (int, float)):
        return int(text)
        
    s = str(text).strip().lower()
    s = s.replace(",", ".").replace(" ", "")
    
    try:
        if "k" in s or "rb" in s:
            num = float(re.findall(r"[\d\.]+", s)[0])
            return int(num * 1000)
        elif "m" in s or "jt" in s:
            num = float(re.findall(r"[\d\.]+", s)[0])
            return int(num * 1000000)
        
        matches = re.findall(r"\d+", s.replace(".", ""))
        if matches:
            return int(matches[0])
    except Exception:
        pass
        
    return 0


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Membaca file JSON Lines (.jsonl) dan mengembalikan list of dict."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_to_jsonl(filepath: str, data: Dict[str, Any]) -> None:
    """Menambahkan (append) satu baris data JSON ke file JSONL."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
