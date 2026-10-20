"""
PROYEK 2 - DETEKSI SPAM PESAN
Jenis ML : Supervised Learning (Klasifikasi)   -> Materi Bab 4.1 A
Metrik   : Accuracy, Precision, Recall, F1, Confusion Matrix  -> Materi Bab 7.1

Catatan: data dibuat sintetis dari template kalimat berbahasa Indonesia.
Untuk data asli, ganti `kumpulkan_data()` dengan dataset "SMS Spam Collection"
(UCI) atau dataset spam lain yang punya kolom teks dan label.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TEMPLATE_SPAM = [
    "SELAMAT! Anda memenangkan {hadiah}. Klik {link} sekarang untuk klaim",
    "Promo gila! Pinjaman cair {durasi} tanpa jaminan, hubungi {nomor}",
    "Dapatkan {hadiah} gratis, daftar sekarang juga di {link}",
    "Anda terpilih menerima bonus {hadiah}. Segera transfer biaya admin",
    "Pinjaman online langsung cair, bunga rendah, klik {link}",
    "Hadiah {hadiah} menunggu Anda! Klaim hari ini, gratis tanpa syarat",
]
TEMPLATE_HAM = [
    "Besok rapat jam {jam} di ruang {ruang}, jangan lupa bawa laptop",
    "Tugas {matkul} dikumpulkan hari {hari} ya",
    "Nanti malam makan di {tempat} bareng yuk",
    "Aku sudah sampai di {tempat}, kamu di mana?",
    "Jadwal kuliah {matkul} dipindah ke jam {jam}",
    # Pesan normal yang "mirip spam" supaya masalahnya tidak terlalu mudah:
    "Klik {link} untuk masuk ke rapat zoom jam {jam}",
    "Makan siang gratis di {tempat} hari {hari}, datang ya",
    "Hubungi {nomor} kalau tugas {matkul} belum jelas",
]
ISIAN = {
    "hadiah": ["mobil", "uang Rp50 juta", "iPhone terbaru", "voucher belanja", "umroh gratis"],
    "link": ["bit.ly/abc12", "promo-cepat.com", "klaim-hadiah.net", "meet.kampus.ac.id"],
    "durasi": ["5 menit", "10 menit", "1 jam"],
    "nomor": ["0812xxxx", "0857xxxx", "0821xxxx"],
    "jam": ["08.00", "10.30", "13.00", "15.30"],
    "ruang": ["A101", "B204", "Lab 3"],
    "matkul": ["Statistika", "Basis Data", "Machine Learning", "Algoritma"],
    "hari": ["Senin", "Selasa", "Rabu", "Jumat"],
    "tempat": ["kantin", "kampus", "kafe depan", "perpustakaan"],
}


class _AmanDict(dict):
    def __missing__(self, key):
        return ""


def _isi(template, rng):
    nilai = _AmanDict({k: rng.choice(v) for k, v in ISIAN.items()})
    return template.format_map(nilai)


def kumpulkan_data(n_per_kelas=300, noise_label=0.04):
    """Tahap 1 - Data collection. noise_label = proporsi label yang sengaja salah,
    meniru kesalahan pelabelan di dunia nyata."""
    rng = np.random.default_rng(SEED)
    teks, label = [], []
    for _ in range(n_per_kelas):
        teks.append(_isi(rng.choice(TEMPLATE_SPAM), rng)); label.append(1)  # 1 = spam
        teks.append(_isi(rng.choice(TEMPLATE_HAM), rng));  label.append(0)  # 0 = bukan spam
    label = np.array(label)
    balik = rng.random(len(label)) < noise_label
    label[balik] = 1 - label[balik]
    return pd.DataFrame({"pesan": teks, "spam": label})


def main():
    print("=" * 60)
    print("PROYEK 2 - DETEKSI SPAM (KLASIFIKASI)")
    print("=" * 60)

    print("\n[Tahap 1] Data collection")
    df = kumpulkan_data()
    print(df.sample(5, random_state=1).to_string())
    print(f"  Proporsi spam: {df['spam'].mean():.1%}")

    print("\n[Tahap 2] Preprocessing teks: TF-IDF (teks -> angka), huruf kecil otomatis")
    # TF-IDF berada di dalam Pipeline, jadi hanya 'belajar' dari data training.

    print("\n[Tahap 3] Data splitting (80/20, stratified agar proporsi kelas sama)")
    X_train, X_test, y_train, y_test = train_test_split(
        df["pesan"], df["spam"], test_size=0.2, random_state=SEED, stratify=df["spam"]
    )
    print(f"  Training: {len(X_train)} | Testing: {len(X_test)}")

    print("\n[Tahap 4-6] Melatih dan membandingkan dua model")
    kandidat = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }
    hasil, model_terlatih = {}, {}
    for nama, clf in kandidat.items():
        model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", clf)])
        model.fit(X_train, y_train)
        acc_tr = accuracy_score(y_train, model.predict(X_train))
        pred = model.predict(X_test)
        hasil[nama] = {
            "acc_train": acc_tr,
            "acc_test": accuracy_score(y_test, pred),
            "f1_test": f1_score(y_test, pred),
        }
        model_terlatih[nama] = model

    tabel = pd.DataFrame(hasil).T.round(3)
    print(tabel.to_string())
    print("  -> Bandingkan acc_train dan acc_test. Selisih besar = tanda overfitting.")

    terbaik = tabel["f1_test"].idxmax()
    print(f"\n  Model terbaik (F1 data uji): {terbaik}")
    model = model_terlatih[terbaik]
    pred = model.predict(X_test)
    print("\n  Laporan klasifikasi (precision, recall, F1):")
    print(classification_report(y_test, pred, target_names=["Bukan spam", "Spam"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Bukan spam", "Spam"], cmap="Blues"
    )
    plt.title(f"Confusion matrix - {terbaik}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proyek2_confusion_matrix.png", dpi=120)
    plt.close()

    print("[Tahap 7] Inference: pesan baru")
    pesan_baru = [
        "Selamat! Anda menang hadiah uang tunai, klik link ini sekarang",
        "Besok kita rapat proyek jam 9 ya, jangan telat",
        "Pinjaman cair 5 menit tanpa jaminan, hubungi kami",
    ]
    for p, y in zip(pesan_baru, model.predict(pesan_baru)):
        print(f"  [{'SPAM' if y == 1 else 'AMAN'}] {p}")

    print("\nGrafik tersimpan di folder output/")


if __name__ == "__main__":
    main()
