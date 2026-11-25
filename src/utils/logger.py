"""
Modul Logger Terpusat
Memberikan logging berbasis Rich dengan warna & styling informatif,
terintegrasi dengan konfigurasi logging standar proyek.
"""

import os
import sys
import logging
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    console = Console(safe_box=True)
except ImportError:
    RichHandler = None

    class SimpleConsole:
        def print(self, *args, **kwargs):
            import re
            cleaned = " ".join(re.sub(r"\[/?.*?\]", "", str(a)) for a in args)
            print(cleaned)

    console = SimpleConsole()


def setup_logger(name: str = "undiksha_sentiment", log_file: str = "pipeline.log") -> logging.Logger:
    """Menginisialisasi logger dengan RichHandler dan FileHandler."""
    logger_inst = logging.getLogger(name)
    
    if not logger_inst.handlers:
        logger_inst.setLevel(logging.INFO)
        
        # Console Handler (Rich atau StreamHandler standar)
        if RichHandler:
            rich_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                markup=True
            )
            rich_handler.setLevel(logging.INFO)
            logger_inst.addHandler(rich_handler)
        else:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
            logger_inst.addHandler(stream_handler)
        
        # File Handler
        try:
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", log_file)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
            logger_inst.addHandler(file_handler)
        except Exception:
            pass
            
    return logger_inst


# Instance logger default
logger = setup_logger()

