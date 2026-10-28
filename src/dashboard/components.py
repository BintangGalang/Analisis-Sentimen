"""
src/dashboard/components.py
===========================
Komponen UI reusable untuk Dashboard Streamlit Analisis Sentimen Undiksha.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_metric_card(title: str, value: str, subtext: str = "", delta_color: str = "normal") -> None:
    """Menampilkan metric card bergaya modern."""
    st.metric(label=title, value=value, delta=subtext, delta_color=delta_color)


def get_sentiment_badge_html(sentiment: str) -> str:
    """Mengembalikan badge HTML berwarna sesuai kelas sentimen."""
    colors = {
        "positif": ("#2ecc71", "#ffffff"),
        "netral":  ("#95a5a6", "#ffffff"),
        "negatif": ("#e74c3c", "#ffffff"),
    }
    bg, fg = colors.get(str(sentiment).lower(), ("#34495e", "#ffffff"))
    return (
        f'<span style="background-color: {bg}; color: {fg}; padding: 3px 10px; '
        f'border-radius: 12px; font-size: 0.85em; font-weight: bold;">'
        f'{str(sentiment).upper()}</span>'
    )
