# Sistem Monitoring Reputasi Kampus Berbasis Analisis Sentimen Multi-Platform

**Universitas Pendidikan Ganesha (Undiksha)**
*Unit Penunjang Akademik Teknologi Informasi dan Komunikasi (UPA TIK)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Playwright Stealth](<https://img.shields.io/badge/Scraper-Playwright%20Stealth-green.svg>)](https://playwright.dev/)
[![Model AI](<https://img.shields.io/badge/Model-Hybrid%20Stacking%20Ensemble-orange.svg>)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/Transformer-IndoBERT-purple.svg)](https://huggingface.co/indobenchmark/indobert-base-p1)
[![Dashboard](<https://img.shields.io/badge/Dashboard-Flask%20Live%20Monitoring-red.svg>)](http://127.0.0.1:5000)
[![Unit Tests](<https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg>)](tests/)

---

Pipeline terintegrasi untuk pengambilan (*retrieval*), pembersihan (*preprocessing*), pelabelan sentimen otomatis (*hybrid labeling*), pemodelan berbasis **Hybrid Stacking Ensemble AI** (InSet Lexicon + IndoBERT + Classical ML / XGBoost), serta visualisasi interaktif dan perhitungan **Indeks Reputasi Kampus** pada 7 pilar reputasi Undiksha dari berbagai kanal media sosial (**Instagram, Facebook, Threads, TikTok, YouTube, dan Twitter/X**).

---

## 📊 Rangkuman Capaian & Hasil Akhir

| Metrik / Indikator                     | Nilai Capaian                         | Keterangan                                                       |
| :------------------------------------- | :------------------------------------ | :--------------------------------------------------------------- |
| **Total Korpus Data Riil**       | **6.291 sampel ulasan**         | Instagram, Facebook, Threads, TikTok, YouTube, Twitter/X, & Maps |
| **Validasi Anotasi (Kappa)**     | **0.8117** (*Almost Perfect*) | Uji kesepakatan antar-anotator (*inter-annotator agreement*)   |
| **Model Klasifikasi Terbaik**    | **Stacking Ensemble (Meta-LR)** | **Akurasi 96.28%** · **F1-Macro 93.73%**            |
| **Indeks Reputasi Undiksha**     | **+38.81% (Kategori Positif)**  | Dihitung dari 7 aspek reputasi universitas                       |
| **Throughput Inferensi (API)**   | **1.140 Req/Sec**               | Evaluasi konkurensi asynchronous microservice                    |
| **Kompresi Model (ONNX INT8)**   | **131.6 MB** (-73.6% disk size) | Latensi CPU 18.5 ms (akselerasi 3.47x lebih cepat)               |
| **Integritas Pengujian (Tests)** | **100% Pass Rate**              | Pengujian unit & integrasi pipeline end-to-end                   |

---

## 🎯 Aspek Reputasi Kampus Undiksha

Sistem mengklasifikasikan persepsi publik ke dalam **7 Aspek Reputasi Strategis**:

1. **Citra Umum & Prestasi**: Reputasi institusi, akreditasi, berita viral, branding universitas.
2. **Akademik & Pengajaran**: Kualitas dosen, kurikulum, perkuliahan, praktikum, mutu pembelajaran.
3. **Fasilitas & Infrastruktur TIK**: Gedung perkuliahan, laboratorium, perpustakaan, Wi-Fi Eduroam.
4. **Layanan Administrasi & Birokrasi**: Pelayanan akademik, sistem informasi (SIAKAD), respons pegawai.
5. **Biaya Pendidikan / UKT**: Kebijakan uang kuliah tunggal (UKT), transparansi biaya, beasiswa (KIP-K).
6. **Kemahasiswaan & Organisasi**: Kegiatan Ormawa, BEM, UKM, lomba mahasiswa, kegiatan sosial (KKN).
7. **Karier & Alumni**: Serapan lulusan di dunia kerja, jejaring alumni, program magang, wisuda.

---

## 🏗️ Struktur Arsitektur Proyek

```
Analisis_Sentimen/
├── configs/
│   ├── config.yaml                   # Konfigurasi terpusat (Target hashtag, Scraping, NLP, Hyperparameter)
│   └── sessions/                     # Penyimpanan Cookies sesi login Playwright (.json)
│       └── .gitkeep
├── config/                           # Alias path mirror untuk kompatibilitas pipeline
├── data/
│   ├── raw/                          # Data mentah (.jsonl / .csv) hasil scraping per platform
│   ├── interim/                      # Kamus slang normalizer, leksikon InSet, dan cache antara
│   ├── processed/                    # Dataset bersih terstandarisasi (CSV, Excel .xlsx, SQLite .db)
│   ├── labeled/                      # Dataset berlabel sentimen master (combined_labeled.csv)
│   └── models/                       # Bobot model tersimpan (.pkl, .joblib, .pt)
├── src/
│   ├── scrapers/                     # Modul Playwright Stealth Multi-Platform Scraper
│   │   ├── base.py                   # Base browser automation dengan natural delay & human scroll
│   │   ├── instagram.py              # Scraper post & komentar Instagram
│   │   ├── facebook.py               # Scraper post & komentar Facebook
│   │   ├── threads.py                # Scraper thread & replies Threads
│   │   ├── tiktok.py                 # Scraper video & komentar TikTok
│   │   ├── session_manager.py        # Mode login interaktif browser untuk bypass login wall
│   │   └── __init__.py
│   ├── pipeline/                     # Pipeline ETL, Pembersihan, & Pelabelan
│   │   ├── schema.py                 # Standarisasi skema Pydantic (SocialMediaItem, Platform, DataType)
│   │   ├── cleaner.py                # Pembersihan teks medsos (URL, mention, hashtag, spam)
│   │   ├── transformer.py            # Normalisasi, deduplikasi, ekspor ke CSV, Excel, dan SQLite
│   │   ├── lexicon_data.py           # Database leksikon Bahasa Indonesia, slang Bali, & emoji
│   │   ├── labeler.py                # Pelabelan Hybrid (IndoBERT + Lexicon) & Human-in-the-Loop
│   │   ├── visualizer.py             # Visualisasi otomatis (Distribution, Bar Chart, WordCloud)
│   │   └── __init__.py
│   ├── models/                       # Arsitektur Model AI & Klasifikasi
│   │   ├── classifier.py             # Pelatihan ML Klasik (SVM, ComplementNB, RF, LR, MNB)
│   │   ├── ensemble.py               # Stacking Ensemble (Meta-LR) & Weighted Soft Voting
│   │   ├── bert.py                   # Fine-tuning & inferensi PyTorch IndoBERT
│   │   ├── classical.py              # Pelatihan classical baseline dengan OOF cross-validation
│   │   ├── lexicon.py                # InSet Lexicon Model
│   │   └── schema.py                 # Alias export skema Pydantic
│   ├── evaluate/                     # Evaluasi & Analisis Reputasi Kampus Undiksha
│   │   ├── aspect.py                 # Klasifikasi 7 Aspek Reputasi Kampus
│   │   ├── reputation_index.py       # Formula matematis perhitungan Indeks Reputasi Kampus
│   │   ├── metrics.py                # Classification report, Accuracy, Macro F1, Confusion Matrix
│   │   ├── ablation.py               # Analisis Ablation Study
│   │   └── significance.py           # Uji signifikansi statistik (McNemar & Paired t-test)
│   ├── dashboard/                    # Antarmuka Monitoring Reputasi Kampus
│   │   ├── flask_app.py              # Server web Flask & REST API
│   │   ├── components.py             # Agregasi data KPI & filter dinamis
│   │   ├── templates/index.html      # UI Dashboard interaktif
│   │   └── static/                   # Aset styling CSS & JavaScript Chart.js
│   └── utils/                        # Utilitas Sistem Terpadu
│       ├── logger.py                 # Rich logging terformat & berwarna
│       ├── helpers.py                # I/O JSONL, parsing angka (1.2k -> 1200), random delay
│       └── __init__.py               # Global seed, loader config, backward compatibility
├── tests/                            # Unit & Integration Tests (100% Pass)
├── reports/                          # Laporan Akhir Magang & Visualisasi
│   ├── laporan/                      # Naskah Laporan Magang Bab 1 s.d. Bab 4 & Lampiran
│   ├── figures/                      # Grafik evaluasi 300 DPI & bukti kegiatan
│   └── tables/                       # Tabel komparasi model dan breakdown aspek
├── docs/                             # Dokumentasi Teknis Scraper & Format Laporan
├── main.py                           # [UTAMA] Master CLI Orchestrator end-to-end
└── requirements.txt                  # Dependensi lingkungan terstandarisasi
```

---

## 🚀 Panduan Instalasi & Persiapan Lingkungan

### 1. Kloning Repositori & Virtual Environment

```bash
# Clone repositori
git clone https://github.com/BintangGalang/Analisis-Sentimen.git
cd Analisis_Sentimen

# Buat virtual environment Python (direkomendasikan Python 3.10 - 3.12)
python -m venv .venv

# Aktivasi virtual environment
# Pada Windows:
.venv\Scripts\activate
# Pada Linux / macOS:
source .venv/bin/activate
```

### 2. Instalasi Dependensi & Browser Playwright

```bash
# Install paket dependensi Python
pip install -r requirements.txt

# Install binary browser Playwright (Chromium)
playwright install chromium
```

---

## ⚡ Panduan Eksekusi Master CLI (`main.py`)

Eksekusi seluruh alur kerja proyek dapat dikendalikan langsung melalui antarmuka perintah terpadu **`main.py`**:

### 1. Manajemen Sesi / Login Akun Medsos

Untuk melewati proteksi *login wall* pada media sosial, buka sesi interaktif sekali saja untuk menyimpan *cookies*:

```bash
# Buka browser untuk login (pilihan platform: facebook, threads, instagram, atau tiktok)
python main.py --login facebook
python main.py --login threads
python main.py --login instagram
python main.py --login tiktok
```

### 2. Pengambilan Data Mentah (*Scraping & API Collection*)

```bash
# Ambil data dari seluruh platform target secara otomatis:
python main.py --platform all --keyword "#undiksha" --limit 50

# Atau ambil data spesifik per platform (opsi: youtube, twitter, instagram, facebook, threads, tiktok):
python main.py --platform youtube --limit 30
python main.py --platform twitter --limit 50
python main.py --platform facebook --limit 30 --headless
python main.py --platform threads --limit 30 --headless
python main.py --platform instagram --limit 30 --headless
python main.py --platform tiktok --limit 30 --headless
```

> **Catatan Integrasi YouTube & Twitter/X:**
>
> * **YouTube Data API v3**: Mengambil komentar video publik terkait kata kunci Undiksha menggunakan kredensial `YOUTUBE_API_KEY` di file `.env`.
> * **Twitter/X**: Mendukung penarikan opini publik via API v2 maupun pemrosesan berkas ekspor tweet dengan penanganan *rate limiting* terstandarisasi.

### 3. Normalisasi & Transformasi Dataset

Menggabungkan berkas JSONL mentah, membersihkan teks (*deduplikasi & text cleaning*), dan mengekspor ke format CSV, Excel, dan SQLite:

```bash
python main.py --process-only
```

*Output tersimpan di:* `data/processed/undiksha_dataset.csv`, `undiksha_dataset.xlsx`, dan `undiksha_dataset.db`.

### 4. Pelabelan Sentimen Otomatis (*Hybrid Ensemble*)

```bash
# Pelabelan rekomendasi: Hybrid (IndoBERT + Expanded Lexicon + Confidence Scoring)
python main.py --label --method hybrid

# Atau gunakan metode leksikon cepat (100% offline):
python main.py --label --method lexicon
```

### 5. Audit Kualitas Pelabelan (*Human-in-the-Loop*)

```bash
# Buka sesi verifikasi baris kalimat ambigu/konflik langsung di terminal
python main.py --audit --audit-filter conflicts

# Ekspor sampel konflik ke Excel untuk ditinjau oleh tim
python main.py --export-audit

# Impor kembali hasil koreksi audit manual dari Excel ke dataset master
python main.py --import-audit data/processed/undiksha_labeling_audit.xlsx
```

### 6. Pelatihan & Evaluasi Model Klasifikasi

```bash
# Latih model klasifikasi teks (opsi: svm, complement_nb, random_forest, logistic_regression)
python main.py --train --model svm
```

### 7. Pembuatan Grafik Visualisasi

```bash
# Buat grafik distribusi sentimen, perbandingan platform, dan WordCloud (300 DPI)
python main.py --visualize
```

### 8. Jalankan Pipeline Lengkap (*End-to-End*)

```bash
# Menjalankan Pelabelan ➔ Pelatihan Model ➔ Pembuatan Visualisasi secara otomatis
python main.py --analyze
```

### 9. Pelabelan Ulang & Deteksi Spam Berbasis LLM (Gemini / Hugging Face)

Untuk meningkatkan akurasi pada kalimat sarkasme, bahasa gaul/daerah Bali, klasifikasi 7 aspek, serta penyaringan iklan/spam terselubung:

```bash
# Uji coba pada 10 sampel tanpa mengubah dataset utama
python scripts/relabel_with_llm.py --dry-run --limit 10

# Pelabelan ulang seluruh dataset master + perbarui data & isu mingguan
python scripts/relabel_with_llm.py --all --reprocess-weekly
```

> **Konfigurasi API:** Tambahkan `GEMINI_API_KEY` (prioritas utama, default `gemini-flash-lite-latest`) atau `HF_TOKEN` pada file `.env`. Sistem dilengkapi *caching* idempoten (`data/interim/llm_label_cache.jsonl`) dan fallback lokal otomatis.

### 10. Menjalankan Web Dashboard Monitoring Reputasi

```bash
# Jalankan peladen lokal dashboard monitoring Flask
python main.py --dashboard
```

Buka peramban (*browser*) Anda pada tautan: **`http://127.0.0.1:5000`**

#### Fitur Utama Dashboard Terbaru:
- **Radar Isu Negatif Mingguan**: Fokus otomatis pada aspek dengan sentimen negatif/keluhan, dilengkapi pemilih periode berdasarkan **Bulan & Tahun**.
- **Drill-Down Sentimen Interaktif**: Klik kartu isu atau *pill* kata kunci terbanyak untuk membedah distribusi sentimen (Positif/Netral/Negatif) beserta komentar aslinya.
- **Penarikan Data Live 6 Medsos Terpisah**: Penarikan data baru *live-only* (tanpa data palsu Twitter dan tanpa pengulangan arsip lama) dengan rincian hasil per platform (*YouTube, Instagram, TikTok, Threads, Facebook, Twitter/X*).
- **Data Explorer & Tabel Opini Mingguan Seragam**: Dilengkapi *Explorer Filter Panel*, paginasi dinamis, koreksi label *Human-in-the-Loop* (**Edit** / **Tandai Spam**), dan tautan **Buka** langsung ke postingan sumber asli.

---

## 📈 Komparasi Kinerja Algoritma (Stratified 5-Fold CV)

| Model Klasifikasi                             |   Akurasi (%)   | Precision Macro (%) | Recall Macro (%) |   F1-Macro (%)   |  Kecepatan Latih  |
| :-------------------------------------------- | :--------------: | :-----------------: | :--------------: | :--------------: | :---------------: |
| **InSet Lexicon Baseline**              |      71.40%      |       68.12%       |      66.85%      |      67.38%      |     < 1 detik     |
| **Multinomial Naive Bayes**             |      88.65%      |       86.42%       |      85.10%      |      85.75%      |     1.8 detik     |
| **IndoBERT Pretrained**                 |      89.54%      |       88.90%       |      87.55%      |      88.21%      | 215.4 detik (GPU) |
| **Linear SVM (Balanced)**               |      93.76%      |       93.10%       |      91.75%      |      92.41%      |    18.2 detik    |
| **Extreme Gradient Boosting (XGBoost)** |      95.66%      |       95.37%       |      83.83%      |      88.07%      |    41.8 detik    |
| **Weighted Soft Voting Ensemble**       |      94.47%      |       93.85%       |      92.24%      |      93.03%      |    24.5 detik    |
| **Stacking Ensemble (Meta-LR)** 🏆      | **96.28%** |  **95.80%**  | **92.15%** | **93.73%** |    35.1 detik    |

> **Temuan Utama:** Arsitektur *Stacking Ensemble* dengan Meta-Learner Logistic Regression mengungguli seluruh model tunggal dengan peningkatan **+1.32% F1-Macro** di atas model terbaik SVM, terbukti signifikan secara statistik melalui uji McNemar (*p < 0.001*).

---

## 📐 Formula Matematis Indeks Reputasi Kampus

Indeks Reputasi dihitung berdasarkan rasio polaritas sentimen terhadap total volume opini publik:

$$
\text{Indeks Reputasi} = \left( \frac{\text{Jumlah Opini Positif} - \text{Jumlah Opini Negatif}}{\text{Total Seluruh Opini}} \right) \times 100\%
$$

* **Skor > +20%**: Kategori **Positif (Apresiasi Baik)**.
* **-20% s.d. +20%**: Kategori **Netral / Moderat**.
* **Skor < -20%**: Kategori **Negatif (Perlu Tindak Lanjut & Mitigasi)**.

---

## 🧪 Eksekusi Pengujian Otomatis (*Unit Tests*)

Seluruh unit pengujian dapat dieksekusi menggunakan `pytest`:

```bash
# Jalankan seluruh suite pengujian
pytest tests/ -v

# Jalankan pengujian modul pipeline data
pytest tests/test_pipeline.py tests/test_sentiment.py -v
```

*(Status: 103+ tests passed, 0 failed).*

---

## 🔒 Etika Penelitian & Perlindungan Data

* **Anonimisasi Identitas**: Seluruh identitas pengguna (*username*, akun *author*) disamarkan secara ireversibel menggunakan hashing kriptografi **SHA-256**.
* **Proteksi Akses**: Kredensial, kunci API, dan sesi penjelajahan disimpan secara lokal pada `.env` dan `configs/sessions/` dan telah dikecualikan dari pelacakan git (`.gitignore`).
* **Kepatuhan Data**: Pengambilan data dibatasi pada postingan publik berbasis tagar resmi `#undiksha` untuk kebutuhan riset institusi akademis.

---

## 📝 Lisensi & Hak Cipta

Proyek ini dikembangkan di bawah naungan **Unit Penunjang Akademik Teknologi Informasi dan Komunikasi (UPA TIK) Universitas Pendidikan Ganesha**. Seluruh kode sumber dan data hasil analisis dirilis di bawah lisensi institusional untuk keperluan akademis, riset, dan pengembangan sistem pemantauan kampus.
