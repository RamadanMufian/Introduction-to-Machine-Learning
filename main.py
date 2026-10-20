"""Menjalankan ketiga proyek berurutan: python main.py"""
import subprocess
import sys
from pathlib import Path

folder = Path(__file__).parent
for skrip in ["proyek1_regresi_harga_rumah.py",
              "proyek2_klasifikasi_spam.py",
              "proyek3_clustering_pelanggan.py"]:
    subprocess.run([sys.executable, str(folder / skrip)], check=True)
    print("\n")
