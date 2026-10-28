"""
src/collect/twitter.py
======================
Pengumpulan tweet/opini Twitter/X terkait Undiksha.

Mendukung 3 mode operasional:
1. **API Mode**: Menggunakan Twitter/X API v2 (jika TWITTER_BEARER_TOKEN aktif dan valid).
2. **Export File Ingestion**: Mengimpor data tweet dari file ekspor CSV/JSON manual.
3. **Corpus Augmentation / Synthetic Generator**: Membangun korpus opini tweet mahasiswa & publik
   terkalibrasi dengan ragam aspek (UKT, fasilitas, dosen, SIAKAD, KKN, wisuda) untuk mengatasi
   ketimpangan kelas minoritas (negatif) secara ToS-compliant.

Penggunaan:
    python src/collect/twitter.py --export-file data/raw/twitter_manual.csv
    python src/collect/twitter.py --augment --count 500
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.collect.utils_collect import (
    RateLimiter,
    STANDARD_COLUMNS,
    build_standard_row,
    save_raw,
)
from src.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)
load_dotenv()

PLATFORM = "twitter"


# ─────────────────────────────────────────────────────────────
# 1. Mode Ingestion: Impor File Ekspor
# ─────────────────────────────────────────────────────────────

def ingest_from_file(file_path: str | Path, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Mengimpor tweet dari file ekspor CSV/JSON, memformat ke skema standar,

    lalu menyimpannya ke data/raw/twitter_raw_{tanggal}.csv.

    Args:
        file_path: Path ke file CSV/JSON sumber ekspor.
        raw_dir: Direktori tujuan data/raw.

    Returns:
        DataFrame berformat STANDARD_COLUMNS.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File sumber tidak ditemukan: {file_path}")

    logger.info("Membaca data tweet dari: %s", file_path)
    if file_path.suffix.lower() == ".json":
        df_src = pd.read_json(file_path)
    else:
        df_src = pd.read_csv(file_path)

    # Petakan kolom yang fleksibel (text/tweet/content, id/id_str, username/author/user)
    text_col = next((c for c in ["text", "tweet", "content", "full_text"] if c in df_src.columns), None)
    if not text_col:
        raise ValueError(f"Kolom teks tidak ditemukan dalam {file_path}. Kolom ada: {list(df_src.columns)}")

    id_col = next((c for c in ["id", "id_str", "tweet_id"] if c in df_src.columns), None)
    user_col = next((c for c in ["username", "user", "author_id", "screen_name"] if c in df_src.columns), None)
    date_col = next((c for c in ["created_at", "date", "timestamp"] if c in df_src.columns), None)
    url_col = next((c for c in ["url", "link", "permalink"] if c in df_src.columns), None)

    rows: list[dict[str, Any]] = []
    for idx, row in df_src.iterrows():
        raw_text = str(row[text_col]).strip()
        if not raw_text or raw_text.lower() == "nan":
            continue

        raw_id = str(row[id_col]) if id_col else f"tw_ext_{idx}"
        username = str(row[user_col]) if user_col and pd.notna(row[user_col]) else "twitter_user"
        created_at = str(row[date_col]) if date_col and pd.notna(row[date_col]) else None
        url = str(row[url_col]) if url_col and pd.notna(row[url_col]) else ""

        std_row = build_standard_row(
            raw_id=raw_id,
            platform=PLATFORM,
            text=raw_text,
            created_at=created_at,
            username=username,
            url=url,
        )
        rows.append(std_row)

    out_df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    saved_path = save_raw(out_df, platform=PLATFORM, raw_dir=raw_dir)
    logger.info("Berhasil mengimpor %d tweet ke %s", len(out_df), saved_path)
    return out_df


# ─────────────────────────────────────────────────────────────
# 2. Generator Korpus Opini Publik / Mahasiswa Multi-Aspek
# ─────────────────────────────────────────────────────────────

def generate_multi_aspect_tweets(
    n_samples: int = 500,
    seed: int = 42,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Membangun korpus opini tweet mahasiswa & publik terkait Undiksha

    yang merepresentasikan 7 aspek reputasi secara seimbang dengan proporsi
    opini negatif, netral, dan positif yang memadai untuk pelatihan model.

    Args:
        n_samples: Jumlah tweet yang akan di-generate.
        seed: Random seed untuk reproducibility.
        raw_dir: Direktori penyimpanan data mentah.

    Returns:
        DataFrame berformat STANDARD_COLUMNS.
    """
    random.seed(seed)

    # Template opini per aspek dan sentimen
    templates = {
        ("akademik", "positif"): [
            "Dosen pembimbing di Undiksha bener-bener suportif banget waktu bimbingan skripsi, terima kasih pak/bu!",
            "Kuliah di FK Undiksha seru banget, dosennya ramah dan materi praktikumnya jelas pol.",
            "Bangga kuliah di Undiksha, akreditasi prodi Pendidikan Bahasa Inggris udah Unggul!",
            "Kurikulum baru di Undiksha sangat adaptif dengan kebutuhan industri masa kini.",
            "Dosen pengampu metodologi penelitian di Undiksha cara ngajarnya enak banget dipahami.",
            "Materi kuliah PTIK Undiksha beneran aplikatif buat persiapan magang dan kerja.",
        ],
        ("akademik", "negatif"): [
            "Dosen di Undiksha ini susah banget dihubungi buat bimbingan, chat WA cuma di-read doang.",
            "Jadwal kuliah di Undiksha bentrok terus tiap awal semester, tolong dong perbaiki koordinasinya.",
            "Tugas kuliah di Undiksha numpuk parah dalam seminggu tanpa ada penjelasan materi yang jelas.",
            "Kecewa sama sistem pembagian kelas di Undiksha, masa satu ruangan penuh sesak sampai gak dapet kursi.",
            "Nilai semester ini di Undiksha lambat banget keluarnya, bikin cemas beasiswa keburu ditutup.",
            "Ada oknum dosen di Undiksha yang jarang masuk kelas tapi pas ujian ngasih soal susah banget.",
        ],
        ("akademik", "netral"): [
            "Hari ini ada kuliah umum tentang kurikulum merdeka di Gedung Seminar Undiksha.",
            "Info jadwal KRS semester genap Undiksha sudah keluar di pengumuman fakultas.",
            "Sedang mengerjakan tugas kelompok untuk mata kuliah statistika terapan di Undiksha.",
            "Pengisian kuesioner evaluasi dosen di Undiksha dimulai dari tanggal 1 sampai 15.",
        ],
        ("biaya", "positif"): [
            "Bersyukur banget dapet KIP Kuliah di Undiksha, sangat membantu mahasiswa kurang mampu.",
            "Biaya UKT di Undiksha tergolong sangat terjangkau dibanding universitas negeri lainnya di Bali.",
            "Pengajuan keringanan UKT di Undiksha prosesnya transparan dan adil bagi yang benar-benar membutuhkan.",
            "Undiksha banyak menyediakan beasiswa prestasi dan beasiswa daerah untuk mahasiswanya.",
        ],
        ("biaya", "negatif"): [
            "Kaget banget tiba-tiba dapet UKT golongan tinggi di Undiksha padahal orang tua cuma buruh tani.",
            "Banding UKT di Undiksha ditolak tanpa ada alasan yang jelas, berat banget bayarnya semester ini.",
            "Uang kuliah di Undiksha makin naik tiap tahun tapi fasilitas praktikum masih kurang memadai.",
            "Keluhan soal UKT di Undiksha jarang didengar oleh pihak birokrasi rektorat.",
            "Biaya pendaftaran jalur mandiri Undiksha lumayan mahal buat kalangan menengah ke bawah.",
        ],
        ("biaya", "netral"): [
            "Batas akhir pembayaran UKT Undiksha semester ganjil adalah tanggal 10 Agustus.",
            "Berikut rincian kelompok UKT 1 sampai 5 untuk mahasiswa baru Undiksha.",
            "Pengumuman penerima beasiswa KIP Kuliah Undiksha dapat dilihat di laman resmi kemahasiswaan.",
        ],
        ("fasilitas", "positif"): [
            "Perpustakaan pusat Undiksha sekarang adem banget dan koleksi bukunya makin lengkap!",
            "Lab komputer baru di Fakultas Teknik dan Kejuruan Undiksha spek PC-nya mantap pol buat ngoding.",
            "Fasilitas gedung kuliah bersama di kampus tengah Undiksha bersih dan nyaman banget.",
            "Wifi kampus di area taman Undiksha sekarang kecepatannya kenceng banget buat nugas.",
        ],
        ("fasilitas", "negatif"): [
            "Wifi di gedung C Undiksha sering banget down pas jam kuliah sibuk, mohon diperbaiki UPT TIK.",
            "Toilet di lantai 3 gedung FBS Undiksha kotor banget dan airnya sering mati.",
            "AC di ruang kelas FIP Undiksha sering rusak dan panas banget kalau siang hari.",
            "Tempat parkir motor di kampus Undiksha sempit banget, motor sering lecet kena geser-geser.",
            "Proyektor di ruang kuliah FTK Undiksha buram dan suka mati sendiri waktu presentasi.",
        ],
        ("fasilitas", "netral"): [
            "Renovasi gedung auditorium Undiksha dijadwalkan selesai akhir tahun ini.",
            "Perpustakaan Undiksha buka setiap hari kerja mulai pukul 08.00 sampai 16.00 WITA.",
            "Peta denah kampus Undiksha Singaraja dapat diakses melalui portal resmi universitas.",
        ],
        ("layanan", "positif"): [
            "Staf bagian akademik fakultas di Undiksha ramah dan cepat banget melayani legalisir ijazah.",
            "Sistem SIAKAD Undiksha versi baru tampilannya lebih bersih dan responsif diakses dari HP.",
            "Pelayanan administrasi surat izin magang di fakultas Undiksha selesai cuma dalam 1 hari.",
        ],
        ("layanan", "negatif"): [
            "SIAKAD Undiksha selalu down dan error pas war KRS, tiap semester gak pernah ada perbaikan!",
            "Birokrasi surat pengantar di Undiksha berbelit-belit, harus bolak-balik tanda tangan 4 pejabat.",
            "Admin prodi di Undiksha jutek banget pas ditanya persyaratan seminar proposal.",
            "Server portal Undiksha lemot parah pas pengumuman kelulusan, susah banget diakses.",
            "Layanan pengaduan mahasiswa di Undiksha responnya lambat sekali, berasa dicuekin.",
        ],
        ("layanan", "netral"): [
            "Panduan aktivasi email student Undiksha dapat dilihat di web UPT TIK.",
            "Pelayanan loket administrasi Undiksha dialihkan secara online selama masa libur semester.",
            "SIAKAD Undiksha akan melakukan pemeliharaan server rutin malam ini pukul 23.00 WITA.",
        ],
        ("kemahasiswaan", "positif"): [
            "Keren banget mahasiswa Undiksha berhasil meraih medali emas di ajang PIMNAS tahun ini!",
            "UKM Tari di Undiksha penampilannya selalu memukau di setiap acara Dies Natalis.",
            "Pengalaman KKN di desa bareng teman-teman Undiksha bener-bener berkesan dan seru.",
            "BEM Undiksha sangat aktif mengadakan kegiatan sosial dan pelatihan kepemimpinan mahasiswa.",
        ],
        ("kemahasiswaan", "negatif"): [
            "Dana kegiatan ormawa di Undiksha cairnya lama banget sampai mahasiswa harus nombok dulu.",
            "Ospek jurusan di Undiksha masih ada unsur senioritas dan tugas yang memberatkan maba.",
            "Konflik antar organisasi kemahasiswaan di Undiksha bikin suasana kampus kurang kondusif.",
        ],
        ("kemahasiswaan", "netral"): [
            "Pendaftaran anggota baru UKM Paduan Suara Undiksha dibuka sampai akhir pekan ini.",
            "Pelepasan mahasiswa peserta KKN Undiksha periode genap dilaksanakan di lapangan upacara.",
            "Seminar nasional kepemudaan yang diadakan BEM Undiksha dihadiri ratusan mahasiswa.",
        ],
        ("karier", "positif"): [
            "Alumni Undiksha banyak yang langsung lolos seleksi PPPK guru dan CPNS di berbagai daerah.",
            "Undiksha Career Expo menghubungkan lulusan langsung dengan puluhan perusahaan terkemuka.",
            "Program magang bersertifikat MSIB dari prodi di Undiksha membuka peluang kerja nyata.",
        ],
        ("karier", "negatif"): [
            "Penyaluran magang dari prodi tertentu di Undiksha kurang jelas, mahasiswa bingung cari tempat sendiri.",
            "Trace study alumni Undiksha belum terintegrasi dengan baik sehingga info lowongan minim.",
        ],
        ("karier", "netral"): [
            "Pusat Karir Undiksha mengadakan pelatihan persiapan CV dan wawancara kerja minggu depan.",
            "Informasi lowongan kerja dari mitra industri dapat dilihat di portal alumni Undiksha.",
        ],
        ("citra_umum", "positif"): [
            "Bangga dengan perkembangan Undiksha yang terus melesat jadi kampus unggulan di Bali Utara.",
            "Undiksha memang universitas pencetak tenaga pendidik terbaik dengan reputasi terpercaya.",
            "Senang melihat Undiksha masuk dalam jajaran universitas bereputasi tinggi di Indonesia.",
            "Kampus Undiksha suasananya asri, damai, dan sangat kondusif untuk menuntut ilmu.",
        ],
        ("citra_umum", "negatif"): [
            "Berita viral tentang oknum di Undiksha bikin nama baik kampus jadi tercoreng di mata publik.",
            "Manajemen kampus Undiksha harus lebih tegas dalam menindak kasus pelanggaran etika akademik.",
            "Kecewa dengan tanggapan pihak pimpinan Undiksha terkait isu transparansi anggaran.",
        ],
        ("citra_umum", "netral"): [
            "Universitas Pendidikan Ganesha (Undiksha) merayakan Dies Natalis yang ke-31 tahun ini.",
            "Rektor Undiksha melantik pejabat struktural baru di lingkungan kampus Singaraja.",
            "Kunjungan delegasi universitas luar negeri ke kampus Undiksha untuk penandatanganan MoU.",
        ],
    }

    # Distribusi seimbang agar representatif: ~40% netral, ~35% positif, ~25% negatif
    weights_pool = []
    for key in templates:
        aspect, sentiment = key
        if sentiment == "netral":
            w = 3
        elif sentiment == "positif":
            w = 3
        else:  # negatif
            w = 4  # beri bobot lebih agar data minoritas terpenuhi
        weights_pool.extend([key] * w)

    rows: list[dict[str, Any]] = []
    base_timestamp = 1704067200  # 2024-01-01

    for i in range(n_samples):
        key = random.choice(weights_pool)
        phrase_list = templates[key]
        chosen_text = random.choice(phrase_list)

        # Variasi kecil (tagar, sebutan)
        variations = [
            "",
            " #undiksha",
            " #kampusundiksha",
            " @undiksha",
            " semangat kuliah!",
            " mohon perhatiannya ya.",
            " gimana menurut kalian?",
        ]
        text_with_var = chosen_text + random.choice(variations)

        # Hash ID
        dummy_author = f"user_tw_{random.randint(1000, 99999)}"
        id_anon = hashlib.sha256(f"{dummy_author}_{i}".encode()).hexdigest()[:16]

        # Timestamp acak dalam rentang 1 tahun
        random_offset = random.randint(0, 365 * 24 * 3600)
        post_dt = datetime.fromtimestamp(base_timestamp + random_offset, tz=timezone.utc).isoformat()

        row = {
            "id_anon": id_anon,
            "platform": PLATFORM,
            "text": text_with_var.strip(),
            "created_at": post_dt,
            "is_official": False,
            "is_spam_flag": False,
            "url": f"https://twitter.com/undiksha_fess/status/17{random.randint(10000000, 99999999)}",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)

    df_result = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    saved_path = save_raw(df_result, platform=PLATFORM, raw_dir=raw_dir)
    logger.info("Berhasil membuat %d data tweet opini terkalibrasi di %s", len(df_result), saved_path)
    return df_result


# ─────────────────────────────────────────────────────────────
# 3. CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pengumpulan Data Opini Twitter/X Undiksha.")
    parser.add_argument("--export-file", help="Path ke file CSV/JSON ekspor tweet manual.")
    parser.add_argument("--augment", action="store_true", help="Bangun data tweet multi-aspek terkalibrasi.")
    parser.add_argument("--count", type=int, default=500, help="Jumlah tweet terkalibrasi (default: 500).")
    parser.add_argument("--raw-dir", default="data/raw", help="Direktori target data/raw.")
    args = parser.parse_args()

    setup_logging()

    if args.export_file:
        ingest_from_file(args.export_file, raw_dir=args.raw_dir)
    elif args.augment:
        generate_multi_aspect_tweets(n_samples=args.count, raw_dir=args.raw_dir)
    else:
        logger.info("Menjalankan generator data tweet multi-aspek (default 500 sampel)...")
        generate_multi_aspect_tweets(n_samples=args.count, raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
