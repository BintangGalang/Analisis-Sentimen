"""
================================================================================
Sistem Monitoring Reputasi Kampus Berbasis Analisis Sentimen Multi-Platform
Universitas Pendidikan Ganesha (Undiksha)
================================================================================
Pipeline Terpadu:
1. Retrieval / Scraper Multi-Platform (Instagram, Facebook, Threads, TikTok, YT, Twitter)
2. Normalisasi, Pembersihan NLP, & Transformasi Dataset (CSV, Excel, SQLite)
3. Pelabelan Sentimen Hybrid (IndoBERT + Lexicon) & Human-in-the-Loop Audit
4. Pelatihan Model Klasifikasi ML / Stacking Ensemble & Evaluasi Metrik
5. Visualisasi Hasil Analisis Sentimen (Grafik Distribusi, WordCloud, dll.)
6. Antarmuka Web Dashboard Monitoring Reputasi Kampus (Flask)
"""

import os
import sys
import asyncio
import argparse
import pandas as pd

# Pastikan direktori root selalu ada di sys.path agar impor 'src' selalu valid
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rich.panel import Panel
except ImportError:
    Panel = None

from src.utils.logger import logger, console
from src.utils.helpers import load_config
from src.scrapers import (
    InstagramScraper,
    FacebookScraper,
    ThreadsScraper,
    TikTokScraper,
    interactive_login
)
from src.pipeline.transformer import DataTransformer
from src.pipeline.labeler import (
    SentimentLabelingOrchestrator,
    InteractiveLabeler,
    AuditManager,
    LexiconLabeler
)
from src.pipeline.visualizer import SentimentVisualizer
from src.models.classifier import SentimentClassifier


def show_banner():
    banner_text = (
        "[bold cyan]SISTEM ANALISIS SENTIMEN & MONITORING REPUTASI KAMPUS (UNDIKSHA)[/bold cyan]\n"
        "[white]Pipeline Multi-Platform: Instagram | Facebook | Threads | TikTok | YouTube | Twitter[/white]\n"
        "[dim]UPT TIK Universitas Pendidikan Ganesha[/dim]"
    )
    if Panel:
        console.print(Panel.fit(banner_text, border_style="cyan"))
    else:
        console.print(banner_text)


async def run_scraper_platform(platform_name: str, config: dict, keyword: str, limit: int, headless: bool, days: int | None = None):
    """Menjalankan scraper untuk platform tertentu."""
    logger.info(f"==> Menjalankan scraper: {platform_name.upper()} (rentang waktu: {days or 'semua'} hari) <==")
    
    scraper = None
    if platform_name == "instagram":
        scraper = InstagramScraper(config=config, headless=headless)
    elif platform_name == "facebook":
        scraper = FacebookScraper(config=config, headless=headless)
    elif platform_name == "threads":
        scraper = ThreadsScraper(config=config, headless=headless)
    elif platform_name == "tiktok":
        scraper = TikTokScraper(config=config, headless=headless)
    elif platform_name == "youtube":
        try:
            from src.collect.youtube import collect_youtube
            keywords = config.get("search_keywords", [keyword])
            raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
            df_yt = collect_youtube(keywords=keywords, raw_dir=raw_dir, max_videos=limit or 15, max_comments=limit or 40, days=days)
            logger.info(f"[SUCCESS] [YOUTUBE] Selesai! Mengumpulkan {len(df_yt)} baris komentar.")
            return df_yt.to_dict(orient="records")
        except Exception as e:
            logger.error(f"[ERROR] [YOUTUBE] Terjadi kendala: {e}")
            return []
    elif platform_name == "twitter":
        try:
            from src.collect.twitter import generate_multi_aspect_tweets
            raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
            df_tw = generate_multi_aspect_tweets(n_samples=limit or 100, raw_dir=raw_dir)
            logger.info(f"[SUCCESS] [TWITTER] Selesai! Mengumpulkan/mengimpor {len(df_tw)} tweet.")
            return df_tw.to_dict(orient="records")
        except Exception as e:
            logger.error(f"[ERROR] [TWITTER] Terjadi kendala: {e}")
            return []
    else:
        logger.error(f"Platform '{platform_name}' tidak didukung.")
        return []

    try:
        items = await scraper.scrape(keyword=keyword, limit=limit)
        logger.info(f"[SUCCESS] [{platform_name.upper()}] Selesai! Mengumpulkan {len(items)} item (post & komentar).")
        return items
    except Exception as e:
        logger.error(f"[ERROR] [{platform_name.upper()}] Terjadi kendala: {e}", exc_info=True)
        return []


async def main_async():
    parser = argparse.ArgumentParser(
        description="Master Pipeline Pengambilan, Pengolahan, dan Analisis Sentimen Reputasi Undiksha."
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["all", "instagram", "facebook", "threads", "tiktok", "twitter", "youtube"],
        default="all",
        help="Platform media sosial yang ingin di-scrape (default: all)"
    )
    parser.add_argument(
        "--keyword", "-k",
        default="#undiksha",
        help="Kata kunci atau hashtag target (default: #undiksha)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Batas maksimal post yang diambil per platform"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Rentang hari pengambilan data (misal: 7 untuk seminggu terakhir)"
    )
    parser.add_argument(
        "--fetch-weekly",
        action="store_true",
        help="Mode Cepat: Tarik dan proses data opini 7 hari terakhir dari multi-platform"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Jalankan browser tanpa antarmuka grafis (headless)"
    )
    parser.add_argument(
        "--login",
        choices=["instagram", "facebook", "threads", "tiktok"],
        help="Buka browser interaktif untuk login akun medsos dan simpan cookies sesi"
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Hanya jalankan pembersihan dan transformasi data mentah tanpa scraping"
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Hanya scrape data mentah tanpa menjalankan pembersihan akhir"
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Jalankan pelabelan sentimen otomatis pada dataset (Hybrid, IndoBERT, atau Lexicon)"
    )
    parser.add_argument(
        "--method",
        choices=["hybrid", "indobert", "lexicon"],
        default=None,
        help="Pilihan metode pelabelan: hybrid (IndoBERT+Leksikon, rekomendasi), indobert (Deep Learning), lexicon (Aturan leksikon)"
    )
    parser.add_argument(
        "--audit", "--interactive",
        dest="audit",
        action="store_true",
        help="Buka sesi audit pelabelan interaktif di terminal (Human-in-the-Loop)"
    )
    parser.add_argument(
        "--audit-filter",
        choices=["conflicts", "uncertain", "all"],
        default="conflicts",
        help="Filter data saat audit interaktif: conflicts (konflik metode), uncertain (confidence < 0.75), all (semua)"
    )
    parser.add_argument(
        "--export-audit",
        action="store_true",
        help="Ekspor baris ambigu/konflik ke file Excel (undiksha_labeling_audit.xlsx) untuk ditinjau"
    )
    parser.add_argument(
        "--import-audit",
        type=str,
        default=None,
        help="Impor kembali hasil koreksi manual dari file Excel ke dataset master"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Latih dan evaluasi model Machine Learning (TF-IDF + Klasifikasi)"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["svm", "complement_nb", "random_forest", "logistic_regression", "naive_bayes"],
        default=None,
        help="Pilihan algoritma Machine Learning: svm, complement_nb, random_forest, logistic_regression, naive_bayes (default: svm)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Buat visualisasi grafik analisis sentimen (Distribution, WordCloud, dll.)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Jalankan pipeline analisis lengkap (Labeling ➔ Modeling ➔ Visualizing)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Jalankan server antarmuka web dashboard monitoring Flask (http://127.0.0.1:5000)"
    )

    args = parser.parse_args()
    show_banner()

    # Muat konfigurasi terpusat
    config = load_config()

    # Mode 1: Sesi Login Interaktif
    if args.login:
        await interactive_login(args.login)
        return

    # Mode 2: Hanya Pembersihan & Transformasi
    if args.process_only:
        console.print("[bold cyan]Menjalankan proses normalisasi & pembersihan data...[/bold cyan]")
        transformer = DataTransformer()
        transformer.process()
        return

    # Mode 3: Ekspor Audit Excel
    if args.export_audit:
        console.print("[bold cyan]Mengekspor data ambigu/konflik ke format Excel untuk audit...[/bold cyan]")
        audit_file = config.get("sentiment_analysis", {}).get("labeling", {}).get("audit_file", "data/processed/undiksha_labeling_audit.xlsx")
        AuditManager.export_for_audit(output_excel=audit_file)
        console.print(f"[bold green]File audit tersimpan di: {audit_file}[/bold green]")
        return

    # Mode 4: Impor Audit Excel
    if args.import_audit:
        console.print(f"[bold cyan]Mengimpor hasil audit dari file Excel: {args.import_audit}...[/bold cyan]")
        updated = AuditManager.import_from_audit(audit_excel=args.import_audit)
        console.print(f"[bold green]Berhasil memperbarui {updated} baris dari hasil audit manual.[/bold green]")
        return

    # Mode 5: Sesi Audit Interaktif CLI (Human-in-the-Loop)
    if args.audit:
        interactive_labeler = InteractiveLabeler()
        interactive_labeler.run_interactive_session(filter_mode=args.audit_filter)
        return

    # Mode 6: Pelabelan Sentimen Otomatis
    if args.label:
        label_method = args.method or config.get("sentiment_analysis", {}).get("labeling", {}).get("default_method", "hybrid")
        console.print(f"[bold cyan]Menjalankan modul Pelabelan Sentimen (Metode: {label_method.upper()})...[/bold cyan]")
        orchestrator = SentimentLabelingOrchestrator(config=config)
        orchestrator.label_dataset(method=label_method)
        return

    # Mode 7: Pelatihan Model Machine Learning
    if args.train:
        selected_model = args.model or config.get("sentiment_analysis", {}).get("modeling", {}).get("default_model", "svm")
        console.print(f"[bold cyan]Menjalankan pelatihan dan evaluasi model sentimen ({selected_model})...[/bold cyan]")
        classifier = SentimentClassifier(config=config)
        classifier.train_and_evaluate(model_type=selected_model)
        return

    # Mode 8: Visualisasi Grafik
    if args.visualize:
        console.print("[bold cyan]Membuat visualisasi grafik analisis sentimen...[/bold cyan]")
        labeled_csv = "data/processed/undiksha_dataset_labeled.csv"
        if not os.path.exists(labeled_csv):
            labeled_csv = "data/labeled/combined_labeled.csv"
            
        if not os.path.exists(labeled_csv):
            console.print("[yellow]Dataset berlabel belum ditemukan, menjalankan pelabelan otomatis terlebih dahulu...[/yellow]")
            orchestrator = SentimentLabelingOrchestrator(config=config)
            df = orchestrator.label_dataset()
        else:
            df = pd.read_csv(labeled_csv)
            # Normalisasi nama kolom jika text_clean
            if "clean_content" not in df.columns and "text_clean" in df.columns:
                df["clean_content"] = df["text_clean"]

        visualizer = SentimentVisualizer(config=config)
        visualizer.generate_all(df)
        return

    # Mode 9: Pipeline Analisis Lengkap End-to-End
    if args.analyze:
        selected_model = args.model or config.get("sentiment_analysis", {}).get("modeling", {}).get("default_model", "svm")
        label_method = args.method or config.get("sentiment_analysis", {}).get("labeling", {}).get("default_method", "hybrid")
        console.print(f"[bold cyan]Menjalankan Analisis Sentimen End-to-End (Labeling: {label_method.upper()} | Model: {selected_model})...[/bold cyan]")
        
        orchestrator = SentimentLabelingOrchestrator(config=config)
        df = orchestrator.label_dataset(method=label_method)

        classifier = SentimentClassifier(config=config)
        eval_res = classifier.train_and_evaluate(model_type=selected_model)

        visualizer = SentimentVisualizer(config=config)
        visualizer.generate_all(df, cm=eval_res.get("confusion_matrix"), labels=eval_res.get("labels"))
        return

    # Mode 10: Jalankan Web Dashboard Monitoring Flask
    if args.dashboard:
        console.print("[bold green]Menjalankan Web Dashboard Monitoring Reputasi Kampus...[/bold green]")
        console.print("Akses antarmuka visual di browser: [bold cyan]http://127.0.0.1:5000[/bold cyan]")
        from src.dashboard.flask_app import app
        app.run(debug=False, host="127.0.0.1", port=5000)
        return

    # Mode Default / Fetch-Weekly: Menjalankan Scraper
    if args.fetch_weekly and not args.days:
        args.days = 7

    post_limit = args.limit or config.get("scraping", {}).get("default_post_limit", 30)
    targets = ["instagram", "facebook", "threads", "tiktok", "twitter", "youtube"] if args.platform == "all" else [args.platform]

    if args.days:
        console.print(f"[bold cyan]Mode Rentang Waktu Aktif: Mengambil opini {args.days} hari terakhir...[/bold cyan]")

    console.print("[bold green]Memulai Scraping Multi-Platform...[/bold green]")
    console.print(f"• Target Keyword : [yellow]{args.keyword}[/yellow]")
    console.print(f"• Target Platform: [yellow]{', '.join(targets)}[/yellow]")
    console.print(f"• Limit Post     : [yellow]{post_limit} per platform[/yellow]")
    console.print(f"• Mode Headless  : [yellow]{args.headless}[/yellow]\n")

    for plat in targets:
        await run_scraper_platform(
            platform_name=plat,
            config=config,
            keyword=args.keyword,
            limit=post_limit,
            headless=args.headless,
            days=args.days,
        )

    if not args.skip_process:
        console.print("\n[bold cyan]Menjalankan modul normalisasi dan pembersihan dataset...[/bold cyan]")
        transformer = DataTransformer()
        transformer.process()


def main():
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold red]Proses dihentikan oleh pengguna (Ctrl+C).[/bold red]")
        sys.exit(0)


if __name__ == "__main__":
    main()

