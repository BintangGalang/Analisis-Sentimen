# Folder Input Data Mingguan (data/weekly_input/)

Folder ini digunakan sebagai jalur alternatif (*fallback loader*) untuk memasukkan data media sosial mingguan yang diambil secara manual atau diekspor dari platform yang memiliki akses terbatas/ToS ketat (misalnya file export CSV/JSON dari Twitter/X Analytics, Facebook Creator Studio, TikTok, atau Instagram).

## Format File yang Didukung:
1. **CSV (`.csv`)**
2. **JSON (`.json`)**
3. **JSON Lines (`.jsonl`)**

## Skema Kolom yang Diharapkan:
- `text` / `teks_asli` / `caption` / `comment`: Konten teks opini / postingan.
- `platform`: Nama platform (`youtube`, `twitter`, `instagram`, `tiktok`, `threads`, `facebook`).
- `created_at` / `date` / `timestamp`: Tanggal postingan (format ISO `YYYY-MM-DD` atau datetime).
- `username` / `author` (opsional): Username pembuat (akan otomatis dianonimkan oleh sistem).

## Cara Kerja:
Saat pipeline `python -m src.pipeline.weekly_run` dijalankan, modul `src/collect/weekly.py` akan otomatis membaca semua file di folder ini, memvalidasi skema, menganonimkan identitas pengguna, menghapus duplikasi, dan mengarsipkan ke `data/raw/weekly/{tahun}-W{minggu}/{platform}.csv`.
File yang sudah berhasil diproses akan dipindahkan ke subfolder `data/weekly_input/archive/`.

