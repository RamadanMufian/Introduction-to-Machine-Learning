"""
PROYEK 3 - SEGMENTASI PELANGGAN
Jenis ML : Unsupervised Learning (Clustering - K-Means)  -> Materi Bab 4.2 A
Alur     : Data tanpa label -> Preprocessing -> Pilih algoritma
           -> Hitung jarak -> Bentuk cluster -> Evaluasi & interpretasi

Catatan: data sintetis. Untuk data asli, ganti `kumpulkan_data()` dengan
dataset "Mall Customers" (Kaggle) atau data pelanggan Anda sendiri.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
FITUR = ["pendapatan_juta", "skor_belanja"]


def kumpulkan_data():
    """Data pelanggan sintetis: 4 kelompok tersembunyi (model TIDAK diberi tahu)."""
    rng = np.random.default_rng(SEED)
    kelompok = [  # (pendapatan rata-rata, skor belanja rata-rata, jumlah)
        (85, 80, 60),
        (85, 25, 60),
        (30, 78, 60),
        (30, 22, 60),
    ]
    baris = []
    for pend, skor, n in kelompok:
        p = rng.normal(pend, 8, n).clip(10, 150)
        s = rng.normal(skor, 9, n).clip(1, 100)
        baris.append(pd.DataFrame({"pendapatan_juta": p.round(1), "skor_belanja": s.round(1)}))
    df = pd.concat(baris, ignore_index=True).sample(frac=1, random_state=SEED)
    return df.reset_index(drop=True)


def beri_nama(pendapatan, skor, rata_pend, rata_skor):
    """Interpretasi cluster secara sederhana berdasarkan posisi terhadap rata-rata."""
    tinggi_p, tinggi_s = pendapatan >= rata_pend, skor >= rata_skor
    if tinggi_p and tinggi_s:
        return "Pembeli premium"
    if tinggi_p and not tinggi_s:
        return "Mampu tapi selektif"
    if not tinggi_p and tinggi_s:
        return "Pencari diskon / impulsif"
    return "Hemat / pasif"


def main():
    print("=" * 60)
    print("PROYEK 3 - SEGMENTASI PELANGGAN (CLUSTERING K-MEANS)")
    print("=" * 60)

    print("\n[1] Data tanpa label")
    df = kumpulkan_data()
    print(df.describe().round(1).to_string())

    print("\n[2] Preprocessing: standardisasi (K-Means berbasis jarak, jadi skala fitur harus setara)")
    X = StandardScaler().fit_transform(df[FITUR])

    print("\n[3] Memilih jumlah cluster K (inertia/elbow dan silhouette score)")
    ks = range(2, 9)
    inertia, silhouette = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(X, km.labels_))
        print(f"  K={k}: inertia={km.inertia_:7.1f} | silhouette={silhouette[-1]:.3f}")
    k_terbaik = list(ks)[int(np.argmax(silhouette))]
    print(f"  -> K dengan silhouette tertinggi: {k_terbaik}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(list(ks), inertia, "o-")
    ax[0].set_title("Elbow method (inertia)")
    ax[0].set_xlabel("K")
    ax[1].plot(list(ks), silhouette, "o-", color="green")
    ax[1].set_title("Silhouette score (makin tinggi makin baik)")
    ax[1].set_xlabel("K")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proyek3_memilih_k.png", dpi=120)
    plt.close()

    print(f"\n[4] Membentuk cluster dengan K={k_terbaik}")
    km = KMeans(n_clusters=k_terbaik, n_init=10, random_state=SEED).fit(X)
    df["cluster"] = km.labels_

    print("\n[5] Evaluasi dan interpretasi kelompok")
    profil = df.groupby("cluster")[FITUR].mean().round(1)
    profil["jumlah"] = df.groupby("cluster").size()
    rata = df[FITUR].mean()
    profil["segmen"] = [
        f"Cluster {c}: " + beri_nama(r.pendapatan_juta, r.skor_belanja,
                                     rata.pendapatan_juta, rata.skor_belanja)
        for c, r in profil.iterrows()
    ]
    print(profil.to_string())
    print(f"\n  Silhouette score akhir: {silhouette_score(X, km.labels_):.3f}")

    plt.figure(figsize=(7, 5))
    for c, r in profil.iterrows():
        sub = df[df["cluster"] == c]
        plt.scatter(sub["pendapatan_juta"], sub["skor_belanja"], alpha=0.7, label=r["segmen"])
    plt.scatter(profil["pendapatan_juta"], profil["skor_belanja"],
                c="black", marker="X", s=150, label="Pusat cluster")
    plt.xlabel("Pendapatan (juta Rp / bulan)")
    plt.ylabel("Skor belanja")
    plt.title("Segmentasi pelanggan dengan K-Means")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proyek3_segmentasi.png", dpi=120)
    plt.close()

    print("\n[6] Menempatkan pelanggan baru ke cluster (inference)")
    scaler = StandardScaler().fit(df[FITUR])
    baru = pd.DataFrame({"pendapatan_juta": [90, 28], "skor_belanja": [85, 20]})
    for (_, r), c in zip(baru.iterrows(), km.predict(scaler.transform(baru))):
        print(f"  Pendapatan {r.pendapatan_juta:.0f} jt, skor {r.skor_belanja:.0f} -> {profil.loc[c, 'segmen']}")

    print("\nGrafik tersimpan di folder output/")


if __name__ == "__main__":
    main()
