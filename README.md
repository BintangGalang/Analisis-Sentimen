# Analisis Sentimen Reputasi Undiksha

> **Hybrid Ensemble Sentiment Analysis** — YouTube · Twitter/X · Instagram/Threads

Pipeline analisis sentimen berbahasa Indonesia yang menggabungkan **lexicon (InSet)**, **machine learning klasik (SVM + TF-IDF)**, dan **transformer (IndoBERT)** melalui teknik **stacking ensemble** untuk memantau reputasi Universitas Pendidikan Ganesha (Undiksha) di media sosial.

---

## Hasil Akhir

| Metrik | Nilai |
|--------|-------|
| **Dataset Total** | 2.983 sampel berlabel (YouTube + Twitter/X + Instagram) |
| **Cohen's Kappa** | 0.8117 (*Almost Perfect Agreement*) — melampaui target ≥0.60 |
| **Model Terbaik** | Stacking Ensemble: **Akurasi 94.87%** · **F1-Macro 93.73%** |
| **Indeks Reputasi Undiksha** | **+43.60% (Kategori Positif)** |
| **Unit Tests** | 103 tests · 100% pass rate |

---

## Konteks

- **Objek analisis**: Reputasi Universitas Pendidikan Ganesha (Undiksha)
- **Konteks**: Proyek magang 20 SKS di UPT TIK Undiksha
- **Label**: positif · netral · negatif
- **Aspek Reputasi**: akademik · fasilitas · layanan · biaya · kemahasiswaan · karier · citra_umum
- **Bahasa data**: Indonesia, slang, campuran Inggris, sesekali istilah Bali

---

## Prasyarat

- Python 3.10+
- (Opsional) GPU dengan CUDA untuk fine-tuning IndoBERT

---

## Setup Cepat

```bash
# 1. Clone repositori
git clone <repo-url>
cd Analisis_Sentimen

# 2. Buat virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Salin dan isi variabel lingkungan
cp .env.example .env
# Edit .env dan isi token API yang diperlukan
```

---

## Struktur Proyek

```
Analisis_Sentimen/
├── configs/config.yaml              # Semua parameter terpusat (seed=42, path, hyperparameter)
├── data/
│   ├── raw/                         # Data mentah per platform (di-ignore git)
│   ├── interim/                     # Data perantara (kamus slang 180+ entri, InSet lexicon)
│   ├── processed/                   # Data bersih siap fitur
│   └── labeled/
│       ├── annotation_guide.md      # Panduan anotasi 12 halaman
│       ├── youtube_labeled.csv      # 1.425 sampel berlabel (100% Real YouTube)
│       ├── twitter_labeled.csv      # 866 sampel berlabel (100% Real Twitter/X)
│       ├── instagram_labeled.csv    # 1.499 sampel berlabel (100% Real Instagram)
│       └── combined_labeled.csv    # 3.790 sampel FINAL (3 Platform Murni Real Lapangan)
├── notebooks/                       # Jupyter notebooks eksplorasi & evaluasi
├── src/
│   ├── collect/                     # Pengumpulan data per platform
│   ├── preprocess/                  # Cleaning, normalisasi slang, stemming Sastrawi
│   ├── features/                    # TF-IDF anti-leakage, BERT dataset
│   ├── models/                      # lexicon.py, classical.py, bert.py, ensemble.py
│   ├── evaluate/                    # Metrics, ablation, aspek, indeks reputasi, McNemar
│   ├── visualize/                   # Plot 300 DPI & Word Cloud
│   ├── label/                       # Tool anotasi & Cohen's Kappa
│   └── dashboard/                   # Flask app + Streamlit app
│       ├── flask_app.py             # Flask REST API backend
│       ├── templates/index.html     # HTML5 responsive dashboard
│       ├── static/css/style.css     # Glassmorphism modern styling
│       └── static/js/main.js        # Chart.js + AJAX + Live Predictor
├── tests/                           # 103 unit tests (100% pass rate)
├── reports/
│   ├── figures/                     # 8 grafik PNG 300 DPI
│   ├── tables/                      # Tabel CSV siap laporan
│   └── logbook.md                   # Logbook mingguan (Minggu 1-16)
├── requirements.txt
├── PRD.md
└── README.md
```

---

## Cara Menjalankan Pipeline

> Jalankan setiap tahap secara berurutan.

### Tahap 2 – Pengumpulan Data

```bash
python src/collect/youtube.py
python src/collect/twitter.py
python src/collect/instagram.py
```

### Tahap 3 – Preprocessing

```bash
python src/preprocess/pipeline.py
```

### Tahap 4 – Pelabelan

```bash
python src/label/annotation_tool.py
python src/label/kappa.py
```

### Tahap 5 – Pelatihan Model

```bash
python src/models/lexicon.py
python src/models/classical.py
python src/models/bert.py
```

### Tahap 6 – Ensemble & Evaluasi

```bash
python src/models/ensemble.py
python src/evaluate/ablation.py
python src/evaluate/significance.py
```

### Tahap 7 – Aspek & Reputasi & Visualisasi

```bash
python src/evaluate/aspect.py
python src/evaluate/reputation_index.py
python src/visualize/plots.py
python src/visualize/wordcloud_gen.py
```

### Tahap 8 – Dashboard Flask

```bash
python src/dashboard/flask_app.py
# Akses di: http://127.0.0.1:5000
```

### Jalankan Semua Unit Tests

```bash
pytest tests/ -v
# Expected: 103 passed, 0 failed
```

---

## Perbandingan Model (Stratified 5-Fold CV, n=2.983)

| Model | Akurasi | F1-Macro |
|-------|---------|----------|
| Komponen A: Lexicon InSet | 71.40% (±1.59) | 67.38% (±1.51) |
| Komponen B: ML Klasik (SVM) | 93.76% (±1.01) | 92.41% (±1.18) |
| Komponen C: IndoBERT | 89.54% (±1.29) | 88.21% (±1.30) |
| Ensemble: Weighted Soft Voting | 94.47% (±0.99) | 93.03% (±1.08) |
| **Ensemble: Stacking (Meta-LR)** | **94.87% (±1.13)** | **93.73% (±1.64)** |

Stacking Ensemble unggul **+1.32% F1-Macro** vs model tunggal terbaik (SVM), signifikan secara statistik (McNemar's test, p < 0.001).

---

## Indeks Reputasi Undiksha per Aspek

| Aspek | Indeks | Predikat |
|-------|--------|----------|
| lainnya | +45.99% | Positif |
| citra_umum | +38.03% | Positif |
| akademik | +29.82% | Positif |
| kemahasiswaan | +19.44% | Netral/Moderat |
| karier | +10.45% | Netral/Moderat |
| fasilitas | +9.43% | Netral/Moderat |
| biaya | +1.71% | Netral/Moderat |
| layanan | +1.28% | Netral/Moderat |

**Indeks Keseluruhan: +43.60% (Positif)**

---

## Reproducibility

Semua percobaan menggunakan **seed = 42** (dikonfigurasi di `configs/config.yaml`).

```python
from src.utils import set_seed, load_config
config = load_config("configs/config.yaml")
set_seed(config["seed"])
```

---

## Etika & Privasi

- Username dan ID pengguna di-hash SHA-256 sebelum disimpan
- Data mentah tidak di-commit ke repositori
- API key disimpan di `.env` (tidak di-commit)
- Hanya menggunakan API resmi dan dataset publik sesuai ToS

---

## Lisensi & Penggunaan Data

Data hasil analisis hanya untuk keperluan akademis magang. Konsultasikan dengan pembimbing sebelum publikasi.
