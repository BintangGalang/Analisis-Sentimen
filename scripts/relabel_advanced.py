"""
scripts/relabel_advanced.py
============================
Perbaikan Pelabelan Lanjutan (v2):
1. Perbaiki aspek 'lainnya' → aspek yang lebih spesifik
2. Perbaiki aspek layanan_administrasi (regex diperluas)
3. Fallback cerdas: undiksha tanpa aspek → isu_citra
4. Deteksi sarkasme dasar → koreksi sentimen Positif → Negatif
5. Export baris confidence < 0.70 ke data/interim/low_confidence_review.csv

Penggunaan:
    python scripts/relabel_advanced.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled_backup_v2.csv"
LOW_CONF_PATH = PROJECT_ROOT / "data" / "interim" / "low_confidence_review.csv"

# ─────────────────────────────────────────────
# 1. Pola Aspek yang Diperluas
# ─────────────────────────────────────────────

ASPECT_RULES = {
    "layanan_administrasi": [
        r"\bsiakad\b", r"\bkrs\b", r"\bwar\s*krs\b", r"\bpendaftaran\b",
        r"\bregistrasi\b", r"\bbeasiswa\b", r"\badministrasi\b",
        r"\bsurat\s+keterangan\b", r"\blegalisir\b", r"\btranskrip\b",
        r"\bserver\s*(down|crash|error|mati|lemot|tidak\s+bisa)\b",
        r"\bportal\s*(akademik|mahasiswa|undiksha)\b",
        r"\blogin\s*(gagal|error|tidak\s+bisa)\b",
        r"\bsistem\s*(akademik|informasi)\b", r"\bsiak\b",
        r"\bpembayaran\s*(ukt|spp|kuliah)\b", r"\bukt\b", r"\bspp\b",
        r"\bipk\b", r"\bnilai\s*(semester|akhir|ujian)\b",
    ],
    "fasilitas": [
        r"\beduroam\b", r"\bwifi\s*(kampus|undiksha)?\b",
        r"\blab\s*(komputer|fisika|kimia|biologi|bahasa|terpadu)?\b",
        r"\bperpustakaan\b", r"\bperpus\b", r"\bruang\s*kuliah\b",
        r"\bkantin\b", r"\bparkir\b", r"\bac\s*(rusak|mati|panas|bocor)?\b",
        r"\bgedung\s*(kuliah|baru|lama|a|b|c|d|e|f)?\b",
        r"\btoilet\s*(kampus|kotor|rusak)?\b", r"\baula\b",
        r"\blapangan\s*(bola|basket|voli|futsal)?\b",
        r"\bkelas\s*(rusak|bocor|kotor|sempit|panas)\b",
        r"\binternet\s*(kampus|lemot|mati|putus)\b",
        r"\blistrik\s*(mati|padam|kampus)\b",
    ],
    "kemahasiswaan": [
        r"\bpkkmb\b", r"\bospek\b", r"\bdies\s*natalis\b",
        r"\bbem\s*(undiksha|kampus|pusat|fak)?\b", r"\bhima\b",
        r"\bukm\b", r"\bhimpunan\b", r"\borganisasi\s*mahasiswa\b",
        r"\bseminar\s*(nasional|internasional|kampus)?\b",
        r"\bwebinar\b", r"\bworkshop\b", r"\blomba\b",
        r"\bkompetisi\b", r"\borasi\b", r"\bdemonstrasi\b",
        r"\bmahasiswa\s*baru\b", r"\bmaba\b", r"\borienta[a-z]*\b",
        r"\bbeasiswa\s*(prestasi|miskin|bidik\s*misi|kip)\b",
    ],
    "akademik": [
        r"\bdosen\b", r"\bkuliah\b", r"\bpelajaran\b", r"\bmata\s*kuliah\b",
        r"\bskripsi\b", r"\btesis\b", r"\bdisertasi\b", r"\btugas\s*akhir\b",
        r"\bjurusan\b", r"\bprodi\b", r"\bprogram\s*studi\b",
        r"\bfakultas\b", r"\bpembimbing\b", r"\bsidang\b",
        r"\bujian\s*(akhir|semester|tengah|nasional)?\b", r"\buts\b", r"\buas\b",
        r"\bpraktikum\b", r"\bprestasi\s*(akademik|mahasiswa)?\b",
        r"\blulus\s*(tepat\s*waktu|sarjana|wisuda)?\b", r"\bwisuda\b",
        r"\bkurikulum\b", r"\bsilabus\b", r"\brps\b",
    ],
    "penerimaan_biaya": [
        r"\bsnbt\b", r"\bsnmptn\b", r"\bsbmptn\b", r"\bpenerimaan\b",
        r"\bjalur\s*(mandiri|snbt|undangan|prestasi)\b",
        r"\bseleksi\s*(masuk|mahasiswa)\b", r"\btes\s*masuk\b",
        r"\buang\s*(kuliah|spp|semester)\b", r"\biuran\b",
        r"\bbiaya\s*(kuliah|pendidikan|studi)\b",
        r"\bmahal\b", r"\bterjangkau\b", r"\bgratis\b", r"\bsubsidi\b",
        r"\bpmbp?\b",
    ],
    "lulusan_karier": [
        r"\bkerja\b", r"\bkarier\b", r"\bkarir\b", r"\blapangan\s*kerja\b",
        r"\bmagang\b", r"\bpkl\b", r"\binternship\b",
        r"\balumni\b", r"\blulusan\b", r"\bsarjana\b",
        r"\bgaji\b", r"\brekrutmen\b", r"\bperusahaan\b",
        r"\bprofesi\b", r"\bgelar\b", r"\bkompetensi\b",
        r"\bsertifikat\b", r"\btrainer\b", r"\bkonsultan\b",
    ],
    "isu_citra": [
        r"\bundiksha\b", r"\bundik\b", r"\bganesha\b",
        r"\bsingaraja\b", r"\breputasi\b", r"\bcitra\s*kampus\b",
        r"\bargumen\b", r"\bopini\b", r"\bkampus\s*(bagus|jelek|terbaik|terpuruk)\b",
        r"\bperingkat\b", r"\bwebometrics\b", r"\bakkreditasi\b",
        r"\bptn\b", r"\buniversitas\b", r"\bkolaborasi\b", r"\bkerjasama\b",
    ],
}

# ─────────────────────────────────────────────
# 2. Pola Sarkasme Dasar
# ─────────────────────────────────────────────

SARCASM_PATTERNS = [
    # "hebat ya SIAKAD selalu crash" → Negatif
    r"(hebat|bagus|keren|mantap|luar\s*biasa|top|oke|ok|baik\s*sekali).{0,40}(crash|down|error|gagal|mati|lelet|lemot|macet|rusak|minta\s*maap|nggak\s*bisa|gabisa|ga\s*bisa)",
    # "sukses tidak bisa login" → Negatif
    r"(sukses|berhasil|berjalan\s*lancar).{0,20}(tidak|tak|gak|ga|nggak|ndak).{0,10}(bisa|berfungsi|jalan|load|dibuka|diakses)",
    # "bagus tapi lambat banget" → Negatif
    r"(bagus|keren|mantap|hebat|oke).{0,25}(tapi|namun|tetapi|padahal|malah|justru|tp).{0,30}(jelek|buruk|parah|menyedihkan|kecewa|gagal|lambat|lemot|error|down|crash)",
    # "terima kasih siakad sudah selalu down" → Negatif
    r"(terima\s*kasih|thanks|makasih).{0,25}(selalu|terus|lagi|lagi-lagi).{0,20}(down|crash|error|gagal|mati|lelet|lemot)",
]

# ─────────────────────────────────────────────
# 3. Keyword Positif Instagram/Bali (False Negative)
# ─────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    r"\bjegeg\b", r"\bcemungut\b", r"\bmenyala\b", r"\bgas\s*pol\b",
    r"\bgaspol\b", r"\bkece\b", r"\bkereeen\b", r"\bkereeen+\b",
    r"\bsemangat\b", r"\bcoool+\b", r"\bkiw+i?\b", r"\bgacor\b",
    r"\bpride\b", r"\bproud\b", r"\bgood\s*job\b", r"\bkereen\b",
    r"\bkereeen\b", r"\bsuper\b", r"\bluar\s*biasa\b",
]

POSITIVE_EMOJI_PATTERN = r"[🔥❤️💪✨🎉👏😍🥳💯🤩🌟⭐👍🏆]"


def has_positive_signal(text: str) -> bool:
    t = str(text).lower()
    for pat in POSITIVE_KEYWORDS:
        if re.search(pat, t):
            return True
    if re.search(POSITIVE_EMOJI_PATTERN, str(text)):
        return True
    return False


def classify_aspect_advanced(text: str) -> str:
    """Klasifikasi aspek dengan aturan yang diperluas."""
    t = str(text).lower()
    # Prioritas: aspek lebih spesifik diuji lebih dulu
    priority = [
        "layanan_administrasi", "akademik", "fasilitas",
        "kemahasiswaan", "penerimaan_biaya", "lulusan_karier", "isu_citra"
    ]
    for asp in priority:
        for pat in ASPECT_RULES[asp]:
            if re.search(pat, t, re.IGNORECASE):
                return asp
    return "lainnya"


def detect_sarcasm(text: str) -> bool:
    """Deteksi sarkasme dasar berdasarkan pola kontras."""
    t = str(text).lower()
    for pat in SARCASM_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False


def main():
    print("=" * 65)
    print("  RELABELING LANJUTAN v2 — Aspek & Sentimen")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"[ERROR] File tidak ditemukan: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset dimuat: {len(df):,} baris, {len(df.columns)} kolom")

    # Simpan backup
    df.to_csv(BACKUP_PATH, index=False)
    print(f"[INFO] Backup disimpan ke: {BACKUP_PATH}")

    # Pastikan kolom text_raw / content ada
    text_col = "text_raw" if "text_raw" in df.columns else ("content" if "content" in df.columns else None)
    if text_col is None:
        print("[ERROR] Tidak ada kolom teks yang valid (text_raw/content).")
        sys.exit(1)

    print(f"[INFO] Kolom teks yang digunakan: '{text_col}'")

    # ─── Tahap 1-4: Perbaiki Aspek ───
    print("\n[TAHAP 1-4] Reklasifikasi aspek...")
    old_aspects = df["aspect"].copy()
    df["aspect"] = df[text_col].apply(classify_aspect_advanced)

    changed_asp = (old_aspects != df["aspect"]).sum()
    print(f"  → {changed_asp:,} baris aspek diubah")

    asp_dist = df["aspect"].value_counts()
    print("  → Distribusi aspek baru:")
    for asp, cnt in asp_dist.items():
        pct = cnt / len(df) * 100
        print(f"     {asp:<30} {cnt:>5} ({pct:.1f}%)")

    # ─── Tahap 5: Koreksi False Negative (Positif → Positif) ───
    print("\n[TAHAP 5] Koreksi false negative (sarkasme)...")
    old_sentiment = df["sentiment"].copy()

    sarcasm_mask = df[text_col].apply(detect_sarcasm)
    # Sarkasme yang dilabeli Positif → Negatif
    sarcasm_positif = sarcasm_mask & (df["sentiment"].str.lower() == "positif")
    df.loc[sarcasm_positif, "sentiment"] = "Negatif"
    df.loc[sarcasm_positif, "labeling_method"] = "sarcasm_correction"
    print(f"  → {sarcasm_positif.sum()} baris sarkasme dikoreksi (Positif → Negatif)")

    # Koreksi false negative pada Instagram: Negatif → Positif
    has_pos = df[text_col].apply(has_positive_signal)
    false_neg_mask = has_pos & (df["sentiment"].str.lower() == "negatif") & (df.get("platform", "") == "instagram")
    # Hanya koreksi jika teks pendek (< 50 karakter) dan confidence < 0.75
    if "confidence" in df.columns:
        false_neg_mask = false_neg_mask & (df["confidence"].fillna(1.0) < 0.75)
    short_mask = df[text_col].apply(lambda x: len(str(x)) < 60)
    false_neg_mask = false_neg_mask & short_mask
    df.loc[false_neg_mask, "sentiment"] = "Positif"
    df.loc[false_neg_mask, "labeling_method"] = "positive_signal_correction"
    print(f"  → {false_neg_mask.sum()} false negative Instagram dikoreksi (Negatif → Positif)")

    # ─── Tahap 6: Export low-confidence ───
    print("\n[TAHAP 6] Export low-confidence rows...")
    if "confidence" in df.columns:
        low_conf = df[df["confidence"].fillna(1.0) < 0.70].copy()
        LOW_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
        low_conf.to_csv(LOW_CONF_PATH, index=False)
        print(f"  → {len(low_conf):,} baris confidence < 0.70 disimpan ke: {LOW_CONF_PATH}")
    else:
        print("  → Kolom 'confidence' tidak ditemukan, tahap ini dilewati")

    # ─── Simpan Hasil ───
    df.to_csv(DATA_PATH, index=False)
    print(f"\n[SELESAI] Dataset disimpan: {DATA_PATH}")

    print("\n📊 Distribusi Sentimen Akhir:")
    sent_dist = df["sentiment"].value_counts()
    for s, c in sent_dist.items():
        pct = c / len(df) * 100
        print(f"   {s:<10} {c:>5} ({pct:.1f}%)")

    print("\n📋 Distribusi Aspek Akhir:")
    for asp, cnt in df["aspect"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {asp:<30} {cnt:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "=" * 65)
    print("  Relabeling lanjutan selesai!")
    print("=" * 65)


if __name__ == "__main__":
    main()

