"""
Unit Test untuk Pipeline Analisis Sentimen (Labeler, Classifier, Visualizer)
"""

import os
import shutil
import unittest
import pandas as pd
import numpy as np

from src.pipeline.labeler import LexiconLabeler
from src.models.classifier import SentimentClassifier
from src.pipeline.visualizer import SentimentVisualizer


class TestSentimentPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = "data/test_sentiment_pipeline"
        self.figures_dir = os.path.join(self.test_dir, "figures")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.config = {
            "sentiment_analysis": {
                "lexicon": {
                    "neutral_threshold": 0.0,
                    "negation_window": 2,
                    "booster_multiplier": 1.5
                },
                "modeling": {
                    "test_size": 0.25,
                    "random_state": 42,
                    "models_dir": self.models_dir
                },
                "visualization": {
                    "figures_dir": self.figures_dir,
                    "dpi": 100
                }
            }
        }

        # Buat dataset sampel untuk pengujian
        self.sample_df = pd.DataFrame([
            {"id": "1", "platform": "instagram", "clean_content": "Undiksha kampus terbaik dan sangat keren prestasi juara", "content": "..."},
            {"id": "2", "platform": "instagram", "clean_content": "Bangga sekali bisa kuliah di sini fasilitas lengkap", "content": "..."},
            {"id": "3", "platform": "tiktok", "clean_content": "Pelayanan sangat lambat dan sistem sering error kecewa", "content": "..."},
            {"id": "4", "platform": "tiktok", "clean_content": "Susah sekali urus berkas ribet parah", "content": "..."},
            {"id": "5", "platform": "threads", "clean_content": "Pengumuman pendaftaran tanggal 20 september", "content": "..."},
            {"id": "6", "platform": "threads", "clean_content": "Jadwal kuliah semester ganjil sudah keluar", "content": "..."},
            {"id": "7", "platform": "facebook", "clean_content": "Hebat mahasiswa undiksha berhasil juara lomba nasional", "content": "..."},
            {"id": "8", "platform": "facebook", "clean_content": "Gagal lolos seleksi sedih banget", "content": "..."}
        ])

        self.sample_csv = os.path.join(self.test_dir, "test_input.csv")
        self.sample_df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_lexicon_labeling_logic(self):
        """Memastikan logika scoring positif, negatif, negasi, dan booster bekerja."""
        labeler = LexiconLabeler(config=self.config)

        # 1. Teks Positif dengan Booster
        res_pos = labeler.analyze_text("kampus ini sangat keren dan berprestasi")
        self.assertEqual(res_pos["sentiment"], "Positif")
        self.assertGreater(res_pos["sentiment_score"], 0)

        # 2. Teks Negatif
        res_neg = labeler.analyze_text("pelayanan lambat dan sangat mengecewakan kecewa")
        self.assertEqual(res_neg["sentiment"], "Negatif")
        self.assertLess(res_neg["sentiment_score"], 0)

        # 3. Negasi Membalik Kata Positif ("tidak bagus" -> Negatif)
        res_negation = labeler.analyze_text("fasilitas di sini tidak bagus")
        self.assertEqual(res_negation["sentiment"], "Negatif")

        # 4. Teks Netral (tanpa kata sentimen)
        res_neutral = labeler.analyze_text("jadwal pendaftaran dibuka besok pagi")
        self.assertEqual(res_neutral["sentiment"], "Netral")
        self.assertEqual(res_neutral["sentiment_score"], 0.0)

    def test_dataset_labeling_export(self):
        """Menguji pelabelan file dataset dan penyimpanan output."""
        labeler = LexiconLabeler(config=self.config)
        labeled_df = labeler.label_dataset(
            input_file=self.sample_csv,
            output_prefix="test_labeled",
            processed_dir=self.test_dir
        )

        self.assertIn("sentiment", labeled_df.columns)
        self.assertIn("sentiment_score", labeled_df.columns)
        self.assertEqual(len(labeled_df), 8)

        # Periksa file output terbentuk
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_labeled.csv")))

    def test_classifier_training(self):
        """Menguji pelatihan model sentimen dan inferensi teks."""
        labeler = LexiconLabeler(config=self.config)
        labeled_df = labeler.label_dataset(
            input_file=self.sample_csv,
            output_prefix="test_labeled_cls",
            processed_dir=self.test_dir
        )

        classifier = SentimentClassifier(config=self.config)
        eval_result = classifier.train_and_evaluate(
            dataset_path=os.path.join(self.test_dir, "test_labeled_cls.csv")
        )

        self.assertIn("accuracy", eval_result)
        self.assertIn("confusion_matrix", eval_result)

        # Uji prediksi teks baru
        pred = classifier.predict("Undiksha sangat luar biasa dan berprestasi")
        self.assertIn("predicted_sentiment", pred)
        self.assertIn("confidence", pred)

    def test_visualizer_generation(self):
        """Menguji pembuatan file gambar visualisasi."""
        labeler = LexiconLabeler(config=self.config)
        labeled_df = labeler.label_dataset(
            input_file=self.sample_csv,
            output_prefix="test_labeled_vis",
            processed_dir=self.test_dir
        )

        visualizer = SentimentVisualizer(config=self.config)
        cm_dummy = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        labels_dummy = ["Negatif", "Netral", "Positif"]

        generated = visualizer.generate_all(labeled_df, cm=cm_dummy, labels=labels_dummy)

        self.assertTrue(os.path.exists(generated["distribution"]))
        self.assertTrue(os.path.exists(generated["platform"]))
        self.assertTrue(os.path.exists(generated["wordclouds"]))
        self.assertTrue(os.path.exists(generated["confusion_matrix"]))


if __name__ == "__main__":
    unittest.main()
