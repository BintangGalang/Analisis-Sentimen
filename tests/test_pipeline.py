"""
Unit test dan verifikasi fungsionalitas modul pipeline.
"""

import os
import sys
import unittest

# Tambahkan project root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.schema import Platform, DataType, EngagementMetrics, SocialMediaItem
from src.pipeline.cleaner import TextCleaner
from src.pipeline.transformer import DataTransformer
from src.utils.helpers import parse_numeric_metric


class TestSentimentPipeline(unittest.TestCase):

    def setUp(self):
        self.cleaner = TextCleaner({
            "cleaning": {
                "remove_urls": True,
                "remove_mentions": True,
                "remove_hashtags": False,
                "remove_emojis": False,
                "min_text_length": 3
            }
        })

    def test_schema_serialization(self):
        """Memastikan model data Pydantic dapat diserialisasi dengan benar."""
        item = SocialMediaItem(
            id="test_post_1",
            platform=Platform.THREADS,
            data_type=DataType.POST,
            parent_id=None,
            author="mahasiswa_undiksha",
            content="Bangga kuliah di Kampus Undiksha!",
            engagement=EngagementMetrics(likes=50, comments_count=10),
            hashtag_query="#undiksha"
        )
        data = item.to_flat_dict()
        self.assertEqual(data["id"], "test_post_1")
        self.assertEqual(data["platform"], "threads")
        self.assertEqual(data["data_type"], "post")
        self.assertEqual(data["likes"], 50)

    def test_text_cleaner(self):
        """Menguji efektivitas pembersihan teks medsos."""
        raw_text = "Halo @undiksha_fbs! Cek web https://undiksha.ac.id/ yaa #undiksha keren bangeeetttt 😍"
        cleaned = self.cleaner.clean(raw_text)
        
        # URL dan mention harus hilang
        self.assertNotIn("https://undiksha.ac.id/", cleaned)
        self.assertNotIn("@undiksha_fbs", cleaned)
        
        # Tanda '#' hilang tapi kata undiksha tetap ada
        self.assertIn("undiksha", cleaned)
        self.assertNotIn("#undiksha", cleaned)
        
        # Karakter berulang 'bangeeetttt' direduksi
        self.assertNotIn("bangeeetttt", cleaned)
        self.assertIn("bangeett", cleaned)
        
        # Emoji dipertahankan (karena remove_emojis: False)
        self.assertIn("😍", cleaned)

    def test_numeric_metric_parser(self):
        """Menguji parsing metrik engagement (K, rb, M)."""
        self.assertEqual(parse_numeric_metric("1.5K"), 1500)
        self.assertEqual(parse_numeric_metric("2,3 rb"), 2300)
        self.assertEqual(parse_numeric_metric("1M"), 1000000)
        self.assertEqual(parse_numeric_metric("45"), 45)
        self.assertEqual(parse_numeric_metric(None), 0)

    def test_transformer_integration(self):
        """Menguji end-to-end pemrosesan data mentah ke CSV, Excel, dan SQLite."""
        test_raw_dir = "data/raw_test"
        test_raw_file = os.path.join(test_raw_dir, "test_sample_raw.jsonl")
        os.makedirs(test_raw_dir, exist_ok=True)
        
        sample_data = [
            {
                "id": "ig_1",
                "platform": "instagram",
                "data_type": "post",
                "parent_id": None,
                "author": "user1",
                "content": "Info pendaftaran mahasiswa baru #undiksha http://bit.ly/pmb",
                "engagement": {"likes": 120, "comments_count": 5}
            },
            {
                "id": "ig_comment_1",
                "platform": "instagram",
                "data_type": "comment",
                "parent_id": "ig_1",
                "author": "camaba",
                "content": "Jurusan ilkom akreditasinya apa ya kak?",
                "engagement": {"likes": 2, "comments_count": 0}
            },
            # Duplikat yang harus dibersihkan
            {
                "id": "ig_1",
                "platform": "instagram",
                "data_type": "post",
                "parent_id": None,
                "author": "user1",
                "content": "Info pendaftaran mahasiswa baru #undiksha http://bit.ly/pmb",
                "engagement": {"likes": 120, "comments_count": 5}
            }
        ]
        
        # Tulis dummy JSONL
        with open(test_raw_file, "w", encoding="utf-8") as f:
            import json
            for s in sample_data:
                f.write(json.dumps(s) + "\n")

        # Jalankan transformer
        transformer = DataTransformer(raw_dir=test_raw_dir)
        df = transformer.process(output_prefix="test_output")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)  # Duplikat dibuang
        self.assertTrue(os.path.exists("data/processed/test_output.csv"))
        self.assertTrue(os.path.exists("data/processed/test_output.xlsx"))
        self.assertTrue(os.path.exists("data/processed/test_output.db"))

        # Bersihkan file uji coba
        if os.path.exists(test_raw_file):
            os.remove(test_raw_file)
        if os.path.exists(test_raw_dir):
            import shutil
            shutil.rmtree(test_raw_dir, ignore_errors=True)
        for ext in [".csv", ".xlsx", ".db"]:
            fpath = f"data/processed/test_output{ext}"
            if os.path.exists(fpath):
                os.remove(fpath)


if __name__ == "__main__":
    unittest.main()
