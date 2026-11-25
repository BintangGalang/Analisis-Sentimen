"""
Modul Pembersih Teks (Text Cleaner & Preprocessing)
Mengoptimalkan teks media sosial bahasa Indonesia untuk analisis sentimen:
- Menghilangkan UI boilerplate medsos (Pembuat, Telah disunting, Lihat selengkapnya, [ARSIP], dll)
- Mendeteksi & memfilter spam iklan komersial non-opini (jual tanah, terima kost, joki tugas, dll)
- Menghilangkan URL / link web
- Menghilangkan mention akun (@username)
- Normalisasi / pembersihan hashtag (#)
- Menghilangkan karakter HTML entities (&amp;, dll)
- Mengurangi karakter berulang berlebihan (misal: 'kerreeennn' -> 'keren')
- Normalisasi spasi dan baris baru
"""

import re
import html
from typing import Optional, Dict, Any, List

# Pola regex deteksi iklan komersial dan promosi non-opini kampus
COMMERCIAL_SPAM_PATTERNS = [
    r'\bdi\s*jual\s+(tanah|rumah|ruko|kavling|bangunan|villa|kos|kost)\b',
    r'\btanah\s+(kavling|premium|murah|dijual|siap\s+bangun)\b',
    r'\b(shm|imb)\b',
    r'\bsertifikat\s+hak\s+milik\b',
    r'\bluas\s*:\s*\d+.*are\b',
    r'\bharga\s*:\s*.*per\s*are\b',
    r'\brumah\s+\d+\s+kamar\s+siap\s+huni\b',
    r'\bterima\s+(kost|kos|kontrakan|sewa\s+kamar)\b',
    r'\b(kost|kos)\s+exclusive\b',
    r'\[\s*kost\s+dekat\s+kampus',
    r'\bready\s+(\d+\s+)?(kamar|kos|kost)\b',
    r'\b\d+k\s*/\s*(bln|bulan)\b',
    r'\bkamar\s+mandi\s+(luar|dalam)\b',
    r'\bdi\s*sewakan\s+(kamar|kos|kost|rumah|kontrakan)\b',
    r'\bdikontrakan\b',
    r'\bover\s+sewa\b',
    r'\bbagi\s+yang\s+berminat\s+bisa\s+langsung\s+dm\b',
    r'\bminat\s+dm\s+ya\b',
    r'\bminat\s*\?\?\s*(dm|wa)\b',
    r'\bjasa\s+joki\b',
    r'\bjoki\s+(tugas|skripsi|makalah|krs|laporan|ujian|ospek)\b',
    r'\bopen\s+joki\b',
    r'\breadystock\b',
    r'\bhubungi\s+wa\s*:?\s*\d+',
    r'\bwa\s*:\s*08\d+',
    r'\bjasa\s+profesional\s+photo(box|booth)\b',
]

RE_COMMERCIAL_SPAM = re.compile('|'.join(COMMERCIAL_SPAM_PATTERNS), re.IGNORECASE)


class TextCleaner:
    """Kelas untuk membersihkan dan menormalisasi teks media sosial."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("cleaning", {}) if config else {}
        self.remove_urls = cfg.get("remove_urls", True)
        self.remove_mentions = cfg.get("remove_mentions", True)
        self.remove_hashtags = cfg.get("remove_hashtags", False)
        self.remove_emojis = cfg.get("remove_emojis", False)
        self.lowercase = cfg.get("lowercase", False)
        self.min_length = cfg.get("min_text_length", 3)

    @classmethod
    def is_commercial_spam(cls, text: Optional[str]) -> bool:
        """Mendeteksi apakah teks merupakan iklan komersial / spam non-opini kampus."""
        if not text or not isinstance(text, str):
            return False
        return bool(RE_COMMERCIAL_SPAM.search(text))

    @staticmethod
    def remove_boilerplate(text: str) -> str:
        """
        Menghapus redaksi boilerplate UI medsos:
        - Badge author komentar: 'Pembuat', 'Penulis', 'Top Fan', dll.
        - Timestamp edit & relatif: 'Telah disunting • 10 ming', '208 ming', dll.
        - Tombol UI: '... Lihat selengkapnya', 'Baca selengkapnya', 'See more', dll.
        - Header arsip scraping: '[ARSIP 2011]', '(ARSIP)', dll.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # 1. Hapus tag arsip scraping: [ARSIP 2011], [ARSIP PERISTIWA], (ARSIP), dll
        cleaned = re.sub(r'\[\s*ARSIP[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(\s*ARSIP[^\)]*\)', '', cleaned, flags=re.IGNORECASE)

        # 2. Hapus UI badges di awal baris/teks: Pembuat, Penulis, Penggemar Berat, Top Fan
        cleaned = re.sub(r'^\s*(Pembuat|Penulis|Penggemar\s+Berat|Top\s+Fan|Kontributor\s+Utama)\s*[\r\n]+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*(Pembuat|Penulis|Penggemar\s+Berat|Top\s+Fan|Kontributor\s+Utama)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*(Pembuat|Penulis)\s*$', '', cleaned, flags=re.IGNORECASE)

        # 3. Hapus status edit dan timestamp relatif medsos
        cleaned = re.sub(r'Telah\s+disunting(\s*•\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun|m|w|d|h|s))?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun)\s*[\r\n]+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'•\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun|m|w|d|h|s)', '', cleaned, flags=re.IGNORECASE)

        # 4. Hapus tombol UI selengkapnya / see more / terjemahan beserta teks lanjutannya jika ada
        cleaned = re.sub(r'(\.{2,}|…)?\s*(Lihat|Baca)\s+(selengkapnya|lebih\s+banyak)\b.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(\.{2,}|…)?\s*See\s+more\b.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(\.{2,}|…)?\s*Lihat\s+terjemahan\b.*', '', cleaned, flags=re.IGNORECASE)

        # 5. Bersihkan elipsis dan sisa spasi di ujung teks
        cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip(' .\t\r\n-–—…')
        return cleaned

    def clean_display(self, text: Optional[str]) -> str:
        """
        Membersihkan boilerplate untuk kolom tampilan (content/text_raw di Data Explorer)
        dengan mempertahankan kapitalisasi dan format alami teks asli.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = html.unescape(text)
        cleaned = self.remove_boilerplate(cleaned)
        # Rapikan newline berlebih
        cleaned = re.sub(r'\r\n|\r', '\n', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def clean(self, text: Optional[str]) -> str:
        """Membersihkan satu string teks sesuai konfigurasi untuk pemrosesan NLP & Model."""
        if not text or not isinstance(text, str):
            return ""

        # 0. Hapus boilerplate UI medsos terlebih dahulu
        cleaned = self.remove_boilerplate(text)
        if not cleaned:
            return ""

        # 1. Unescape HTML entities
        cleaned = html.unescape(cleaned)

        # 2. Hapus URL / tautan web
        if self.remove_urls:
            cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)

        # 3. Hapus mention akun (@user)
        if self.remove_mentions:
            cleaned = re.sub(r"@\w+", "", cleaned)

        # 4. Tangani hashtag
        if self.remove_hashtags:
            # Hapus total hashtag beserta katanya
            cleaned = re.sub(r"#\w+", "", cleaned)
        else:
            # Hanya hapus simbol pagar '#', biarkan katanya tetap ada
            cleaned = re.sub(r"#([A-Za-z0-9_]+)", r"\1", cleaned)

        # 5. Tangani emoji jika diaktifkan (opsional, karena emoji sering memuat sentimen)
        if self.remove_emojis:
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "\U0001f680-\U0001f6ff"  # transport & map
                "\U0001f1e0-\U0001f1ff"  # flags (iOS)
                "\U00002702-\U000027b0"
                "\U000024c2-\U0001f251"
                "]+",
                flags=re.UNICODE
            )
            cleaned = emoji_pattern.sub(r"", cleaned)

        # 6. Reduksi karakter berulang berlebih (misal: 'bangeeetttt' -> 'banget')
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        # 7. Hapus karakter non-printable dan rapikan spasi/newline
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 8. Lowercase jika diminta
        if self.lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def is_valid_text(self, text: str) -> bool:
        """Memeriksa apakah teks valid: bukan spam komersial dan memiliki panjang/kata bermakna yang cukup."""
        if not text or not isinstance(text, str):
            return False
        if self.is_commercial_spam(text):
            return False
        stripped = text.strip()
        if len(stripped) < self.min_length:
            return False
        # Harus memiliki setidaknya 3 karakter alfanumerik (menolak simbol kosong atau remah UI)
        alpha_chars = re.findall(r'[A-Za-z0-9]', stripped)
        if len(alpha_chars) < 3:
            return False
        return True
