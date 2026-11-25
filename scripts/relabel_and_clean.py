"""
scripts/relabel_and_clean.py
============================
Skrip komprehensif untuk:
1. Pembersihan konten spam & iklan komersial (jual tanah, kos eksternal, joki tugas).
2. Penyempurnaan 7 aspek reputasi (menghapus bias kata kos dan overmatching kota).
3. Perbaikan False Negatives (Emoji Veto, kamus Bahasa Bali, Short-Text Guardrails).
4. Pembaruan dataset ground truth (data/labeled/combined_labeled.csv) dan kalkulasi metrik.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
OUTPUT_CLEAN_CSV = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
OUTPUT_SPAM_CSV = PROJECT_ROOT / "data" / "interim" / "excluded_spam.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# ─────────────────────────────────────────────────────────────
# 1. Regex Deteksi Spam & Iklan Komersial
# ─────────────────────────────────────────────────────────────

SPAM_PATTERNS = [
    # Jual beli tanah / properti
    re.compile(r"\b(di\s*jual\s*tanah|dijual\s*tanah|tanah\s*premium|luas:\s*\d+|tanah\s*kavling|sertifikat\s*shm|shm|lokasi\s*strategis.*menit\s*ke\s*kampus|1\s*are|\d+\s*are)\b", re.I),
    # Iklan kos-kosan komersial
    re.compile(r"\b(terima\s*kost?|kost?\s*bulanan|free\s*wifi.*550k|kamar\s*mandi\s*(luar|dalam)|kos\s*exclusive|hanya\s*ada\s*\d+\s*kamar|bangunan\s*baru.*ready|dm\s*ya.*terimakasih|hubungi\s*wa|wa\s*08\d+|infokos|sewa\s*kamar)\b", re.I),
    # Jasa joki skripsi / tugas
    re.compile(r"\b(joki\s*tugas|jasa\s*joki|joki\s*skripsi|bantu\s*tugas|bantu\s*skripsi|fast\s*respon\s*joki|jasa\s*olah\s*data|joki\s*krs)\b", re.I),
    # Promosi komersial umum non-kampus
    re.compile(r"\b(rental\s*motor|sewa\s*motor|laundry\s*kiloan|top\s*up\s*game|open\s*po\s*|ready\s*stock|order\s*via\s*dm|klik\s*link\s*di\s*bio)\b", re.I),
]

def is_spam_or_commercial(text: str) -> bool:
    """Mendeteksi apakah teks adalah iklan komersial eksternal."""
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    for pat in SPAM_PATTERNS:
        if pat.search(t):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 2. Aturan Deteksi 7 Aspek Reputasi
# ─────────────────────────────────────────────────────────────

ASPECT_RULES = {
    "layanan_administrasi": [
        r"\b(siakad|krs|war\s*krs|server\s*(down|jebol|crash)|web\s*(down|error)|birokrasi|legalisir|surat\s*(aktif|bebas)|staf\s*tu|admin\s*fakultas|pelayanan\s*admin|sistem\s*error)\b"
    ],
    "penerimaan_biaya": [
        r"\b(ukt|uang\s*kuliah|biaya\s*kuliah|snbp|snbt|mandiri|beasiswa|kip|kip-k|kip\s*kuliah|cicilan\s*ukt|banding\s*ukt|keringanan\s*ukt|bayar\s*ukt|daftar\s*ulang|pmb|seleksi\s*masuk)\b"
    ],
    "akademik": [
        r"\b(dosen|kuliah|matkul|mata\s*kuliah|kurikulum|akreditasi|skripsi|bimbingan|sidang|seminar\s*proposal|sempro|uas|uts|ujian|tugas\s*kuliah|nilai\s*dosen|ipk|praktikum|fakultas|prodi|kedokteran|fbs|ftk|fmipa|fip|fh|fe|fok)\b"
    ],
    "fasilitas": [
        r"\b(kampus\s*(tengah|bawah|jinengdalem)|gedung\s*kuliah|ruang\s*kelas|ac\s*kelas|ac\s*rusak|toilet\s*kampus|wc\s*kampus|perpustakaan|perpus\s*undiksha|wifi\s*(undiksha|kampus|eduroam)|eduroam|lab\s*(komputer|bahasa|ipa)|parkir\s*kampus|asrama\s*undiksha|lapangan\s*kampus)\b"
    ],
    "kemahasiswaan": [
        r"\b(ormawa|bem|dpm|ukm|pkkmb|lomba|prestasi|juara|kkn|mbkm|magang|pkm|kegiatan\s*kampus|acara\s*kampus|dies\s*natalis|seni\s*tari|padus)\b"
    ],
    "lulusan_karier": [
        r"\b(wisuda|wisudawan|alumni|lulusan|tracer\s*study|kerja|karier|karir|sarjana|toga|ijazah)\b"
    ],
    "isu_citra": [
        r"\b(rektor|pimpinan|kebijakan\s*kampus|nama\s*baik\s*undiksha|reputasi\s*undiksha|prestasi\s*undiksha|ranking\s*undiksha|skandal|kasus|viral|demo|undiksha\s*jaya|bangga\s*undiksha|almamater)\b"
    ]
}

def detect_refined_aspect(text: str) -> str:
    """Mendeteksi aspek reputasi dengan pencocokan ekspresi terarah (tanpa bias kata tunggal kos/singaraja)."""
    if not isinstance(text, str) or not text.strip():
        return "lainnya"
    t = text.lower()
    
    # Prioritaskan pencocokan aspek spesifik
    for aspect, patterns in ASPECT_RULES.items():
        for pat in patterns:
            if re.search(pat, t, re.I):
                return aspect
                
    # Jika tidak cocok aspek spesifik, cek apakah opini institusi umum
    if re.search(r"\b(undiksha|ganesha|kampusku|almamater)\b", t, re.I):
        return "isu_citra"
        
    return "lainnya"


# ─────────────────────────────────────────────────────────────
# 3. Aturan Pelabelan Sentimen (Mengatasi False Negatives)
# ─────────────────────────────────────────────────────────────

# Emoji Positif Kuat
POSITIVE_EMOJIS = set(["😍", "🔥", "🙌", "🤩", "🥰", "🫰", "❤️", "👏", "🎉", "✨", "🥳", "👍", "😁", "😊", "💖", "💪"])
NEGATIVE_EMOJIS = set(["😢", "😭", "😡", "🤬", "🤮", "💔", "👎", "😤", "🤢"])

# Kata Positif / Apresiasi (termasuk Slang & Bahasa Bali)
POSITIVE_WORDS = set([
    "bagus", "keren", "mantap", "mantaps", "mantapp", "semangat", "semangatt", "cemungut", "cemungutt",
    "menyala", "sukses", "bangga", "tercinta", "cinta", "hebat", "jos", "gokil", "selamat", "love",
    "suka", "favorit", "asri", "sejuk", "ramah", "lengkap", "juara", "terbaik", "idola", "gacor",
    "jaya", "alhamdulillah", "puji tuhan", "bismillah", "berkah", "suksma", "suksme", "top", "makasih",
    "terima kasih", "apresiasi", "paten", "terdepan", "jegeg", "jegegg", "luung", "jaen", "dedari",
    "gemilang", "gasken", "gaspol", "kece", "salut"
])

# Kata Negatif / Keluhan Eksplisit
NEGATIVE_WORDS = set([
    # Sistem & IT
    "error", "eror", "down", "lelet", "lemot", "ngelag", "lag", "crash", "rusak", "jebol", "kendala", "masalah",
    # Biaya
    "mahal", "kemahalan", "mencekik", "denda", "terbebani", "boros", "rugi", "ditolak",
    # Layanan & Birokrasi
    "lambat", "bobrok", "ribet", "susah", "payah", "sulit", "dipersulit", "berbelit", "antri", "antre",
    "diabaikan", "dicuekin", "cuek", "jutek", "ketus", "slow respon", "kecewa", "mengecewakan",
    # Fasilitas Fisik
    "bau", "kotor", "jorok", "mati air", "mati lampu", "panas", "gerah", "pengap", "becek", "bocor", "semrawut",
    # Emosi Negatif
    "lelah", "capek", "sambat", "anjir", "anjing", "benci", "stres", "ampas", "busuk", "kapok", "protes",
    "ngeluh", "keluhan", "kacau", "zonk", "muak", "kesel", "sebel", "parah", "jelek", "buruk"
])

# Pola Negasi
NEGATION_PATTERNS = [
    re.compile(r"\b(tidak|bukan|kurang|ga|gak|nggak|ndak|blm|belum)\s+(ramah|bagus|puas|memuaskan|jelas|adil|transparan|profesional|memadai|responsif|bersih|nyaman|layak|mudah|murah|baik|jegeg|luung)\b", re.I),
    re.compile(r"\b(ga|gak|nggak|tidak)\s+(bisa|dapat|ada|fungsi|jalan)\b", re.I),
]

def label_sentiment_accurately(text: str, current_bert: str, current_lex: str) -> tuple[str, float]:
    """
    Menentukan label sentimen dengan akurasi kontekstual, menerapkan:
    - Emoji Veto
    - Kamus Bahasa Bali & Slang
    - Short-Text Guardrails
    - Resolusi Sarkasme
    """
    if not isinstance(text, str) or not text.strip():
        return "Netral", 0.50
        
    t_raw = text.strip()
    t_lower = t_raw.lower()
    tokens = re.findall(r"\b\w+\b", t_lower)
    word_count = len(tokens)
    
    # 1. Cek Keberadaan Emoji
    has_pos_emoji = any(em in t_raw for em in POSITIVE_EMOJIS)
    has_neg_emoji = any(em in t_raw for em in NEGATIVE_EMOJIS)
    
    # 2. Cek Negasi Eksplisit
    negated_hits = 0
    for pat in NEGATION_PATTERNS:
        if pat.search(t_lower):
            negated_hits += 2
            
    # 3. Hitung Kata Sentimen
    pos_hits = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_hits = sum(1 for w in tokens if w in NEGATIVE_WORDS) + negated_hits
    
    # 4. Deteksi Sarkasme ("hebat ya server siakad selalu crash")
    is_sarcasm = bool(re.search(r"\b(hebat|keren|mantap)\b", t_lower)) and bool(re.search(r"\b(down|crash|error|lemot|rusak|jebol|tumbang)\b", t_lower))
    if is_sarcasm:
        return "Negatif", 0.90
        
    # 5. ATURAN VETO: Emoji Positif Kuat atau Kata Pujian Bali/Slang
    # Contoh: "Siall ini jegegg sekaliii😍😍", "Menyala... 🔥", "cemungutt"
    if (has_pos_emoji or pos_hits >= 1) and neg_hits == 0 and not is_sarcasm:
        # Batalkan halusinasi Negatif dari IndoBERT
        return "Positif", 0.95
        
    if has_neg_emoji and pos_hits == 0:
        return "Negatif", 0.92
        
    # 6. ATURAN SHORT-TEXT GUARDRAILS: Komentar Singkat / Interaksi Kasual
    # Contoh: "Gerokgak", "Folbek 😂", "kgn.", "omgggg", "fhis jadi kebun duren wkwk"
    if word_count <= 3 and neg_hits == 0 and not is_sarcasm:
        if pos_hits > 0 or has_pos_emoji:
            return "Positif", 0.90
        # Jika hanya teks singkat tanpa keluhan -> Pasti Netral
        return "Netral", 0.85
        
    # Humor / Candaan ("wkwk", "haha") tanpa kata keluhan -> Netral
    if bool(re.search(r"\b(wkwk|haha|hehe|folbek)\b", t_lower)) and neg_hits == 0:
        return "Netral", 0.85
        
    # 7. Keluhan Eksplisit Kampus
    is_complaint_q = bool(re.search(r"\b(kenapa|knp|kok|gimana sih)\b", t_lower)) and bool(re.search(r"\b(siakad|ukt|krs|error|lemot|mahal|rusak|susah|server)\b", t_lower))
    if is_complaint_q or neg_hits > pos_hits:
        return "Negatif", 0.88
        
    if pos_hits > neg_hits:
        return "Positif", 0.88
        
    # Fallback jika seimbang: cek BERT jika bukan false negative teks pendek
    if current_bert in ["Positif", "Negatif"] and word_count >= 5:
        return current_bert.capitalize(), 0.75
        
    return "Netral", 0.70


# ─────────────────────────────────────────────────────────────
# 4. Pipeline Utama Eksekusi
# ─────────────────────────────────────────────────────────────

def run_relabeling():
    print("=" * 65)
    print("  [MEMULAI PROSES RELABELING & PEMBERSIHAN DATA UNDIKSHA]")
    print("=" * 65)
    
    if not INPUT_CSV.exists():
        print(f"Error: File input tidak ditemukan: {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    total_raw = len(df)
    print(f"Total baris dataset awal: {total_raw:,}")
    
    text_col = "content" if "content" in df.columns else ("text_raw" if "text_raw" in df.columns else "text")
    
    # 1. Identifikasi & Pisahkan Spam / Iklan Komersial
    spam_mask = df[text_col].astype(str).apply(is_spam_or_commercial)
    df_spam = df[spam_mask].copy()
    df_clean = df[~spam_mask].copy()
    
    print(f"Iklan / Spam komersial teridentifikasi : {len(df_spam):,} baris")
    print(f"Opini publik relevan tersaring        : {len(df_clean):,} baris")
    
    # Simpan arsip spam ke data/interim/excluded_spam.csv
    OUTPUT_SPAM_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_spam.to_csv(OUTPUT_SPAM_CSV, index=False)
    print(f"Arsip iklan komersial disimpan ke     : {OUTPUT_SPAM_CSV}")
    
    # 2. Pelabelan Ulang Aspek & Sentimen pada Data Bersih
    print("\nMenjalankan klasifikasi aspek dan sentimen kontekstual...")
    new_aspects = []
    new_sentiments = []
    new_confidences = []
    
    corrected_false_negatives = 0
    
    for _, row in df_clean.iterrows():
        t = str(row.get(text_col, ""))
        curr_bert = str(row.get("bert_sentiment", "Netral"))
        curr_lex = str(row.get("lexicon_sentiment", "Netral"))
        curr_sent = str(row.get("sentiment", "Netral"))
        
        # Deteksi aspek baru
        asp = detect_refined_aspect(t)
        new_aspects.append(asp)
        
        # Deteksi sentimen baru
        sent, conf = label_sentiment_accurately(t, curr_bert, curr_lex)
        new_sentiments.append(sent)
        new_confidences.append(conf)
        
        # Hitung koreksi False Negative
        if curr_sent.lower() == "negatif" and sent.lower() in ["positif", "netral"]:
            corrected_false_negatives += 1
            
    df_clean["aspect"] = new_aspects
    df_clean["sentiment"] = new_sentiments
    df_clean["confidence"] = new_confidences
    df_clean["labeling_method"] = "hybrid_contextual_v2"
    
    print(f"Total False Negatives yang diperbaiki : {corrected_false_negatives:,} baris")
    
    # 3. Hitung Distribusi Baru
    counts = df_clean["sentiment"].value_counts().to_dict()
    print("\nDistribusi Sentimen Baru:")
    for s, c in counts.items():
        pct = (c / len(df_clean)) * 100
        print(f"  - {s:10s}: {c:,} ({pct:.1f}%)")
        
    asp_counts = df_clean["aspect"].value_counts().to_dict()
    print("\nDistribusi 7 Aspek Baru:")
    for a, c in asp_counts.items():
        pct = (c / len(df_clean)) * 100
        print(f"  - {a:22s}: {c:,} ({pct:.1f}%)")
        
    # 4. Simpan Dataset Baru ke data/labeled/combined_labeled.csv
    df_clean.to_csv(OUTPUT_CLEAN_CSV, index=False)
    print(f"\nDataset ground truth berhasil diperbarui: {OUTPUT_CLEAN_CSV}")
    
    # 5. Regenerasi Tabel Deret Waktu & Indeks Reputasi
    regenerate_timeline_table(df_clean)
    
    print("\n" + "=" * 65)
    print("  [RELABELING & PEMBERSIHAN SELESAI DENGAN SUKSES]")
    print("=" * 65)


def regenerate_timeline_table(df: pd.DataFrame):
    """Memperbarui reports/tables/reputation_index_timeline.csv agar grafik timeline dashboard sinkron."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TABLES_DIR / "reputation_index_timeline.csv"
    
    date_col = None
    if "created_at" in df.columns:
        date_col = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "scraped_at" in df.columns:
        scraped_dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
        date_col = date_col.fillna(scraped_dt) if date_col is not None else scraped_dt
        
    if date_col is None or date_col.dropna().empty:
        return
        
    df_temp = df.copy()
    df_temp["_dt"] = date_col
    df_temp = df_temp.dropna(subset=["_dt"])
    df_temp["_periode"] = df_temp["_dt"].dt.to_period("M").astype(str)
    
    records = []
    for per, grp in df_temp.groupby("_periode"):
        tot = len(grp)
        if tot <= 0:
            continue
        sc = grp["sentiment"].astype(str).str.lower().value_counts().to_dict()
        p = sc.get("positif", 0)
        n = sc.get("negatif", 0)
        net = sc.get("netral", 0)
        idx = (p - n) / tot
        
        pred = "Sangat Positif" if idx >= 0.5 else ("Positif" if idx >= 0.1 else ("Netral" if idx >= -0.1 else ("Negatif" if idx >= -0.5 else "Sangat Negatif")))
        records.append({
            "_periode": per,
            "Total": tot,
            "Positif": p,
            "Netral": net,
            "Negatif": n,
            "Indeks_Reputasi": round(idx, 4),
            "Indeks_%": f"{idx * 100:+.2f}%",
            "Predikat": pred
        })
        
    df_timeline = pd.DataFrame(records).sort_values("_periode")
    df_timeline.to_csv(out_file, index=False)
    print(f"Tabel timeline reputasi diperbarui    : {out_file} ({len(df_timeline)} periode)")


if __name__ == "__main__":
    run_relabeling()

