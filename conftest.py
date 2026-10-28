"""
conftest.py – Root-level pytest configuration.

Menambahkan direktori root proyek ke sys.path agar semua import
`from src.xxx import yyy` dapat ditemukan selama pytest collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).parent))
