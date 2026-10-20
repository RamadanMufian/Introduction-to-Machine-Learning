"""
PROYEK 1 - PREDIKSI HARGA RUMAH
Jenis ML : Supervised Learning (Regresi)        -> Materi Bab 4.1 B dan Bab 8
Workflow : Collection -> Preprocessing -> Splitting -> Model -> Training
           -> Evaluasi -> Inference                -> Materi Bab 6

Catatan: data dibuat sintetis agar proyek bisa langsung jalan tanpa unduhan.
Untuk data asli, ganti fungsi `kumpulkan_data()` dengan pd.read_csv("file_anda.csv").
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- TAHAP 1
def kumpulkan_data(n=500):
    """Tahap 1 - Data Collection: membuat data rumah sintetis (harga dalam juta rupiah)."""
    rng = np.random.default_rng(SEED)
    luas = rng.uniform(40, 250, n).round(0)
    kamar = np.clip((luas / 50 + rng.normal(0, 0.6, n)).round(), 1, 6).astype(int)
    lokasi = rng.choice(["Kota A", "Kota B", "Kota C"], n, p=[0.40, 0.35, 0.25])

    harga_per_m2 = {"Kota A": 12, "Kota B": 8, "Kota C": 5}  # juta rupiah / m2
    harga = (
        luas * pd.Series(lokasi).map(harga_per_m2).values
        + kamar * 25
        + rng.normal(0, 80, n)  # noise: faktor lain yang tidak tercatat
    ).round(0)

    df = pd.DataFrame(
        {"luas_m2": luas, "kamar_tidur": kamar, "lokasi": lokasi, "harga_juta": harga}
    )

    # Sengaja dikotori supaya tahap preprocessing bermakna (seperti data dunia nyata)
    idx_kosong = rng.choice(n, 20, replace=False)
    df.loc[idx_kosong, "luas_m2"] = np.nan                       # missing values
    df = pd.concat([df, df.sample(5, random_state=SEED)])        # duplikat
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- TAHAP 2
def preprocessing(df):
    """Tahap 2 - Data Preprocessing."""
    print(f"  Jumlah baris awal          : {len(df)}")
    print(f"  Missing values per kolom   :\n{df.isna().sum().to_string()}")
    print(f"  Baris duplikat             : {df.duplicated().sum()}")

    df = df.drop_duplicates().copy()                              # hapus duplikat
    X = df.drop(columns="harga_juta")                             # fitur
    y = df["harga_juta"]                                          # target / label

    # Missing value diisi median, kategori di-encode, fitur numerik distandardisasi.
    # Semua dibungkus Pipeline agar transformasi dipelajari HANYA dari data training
    # (mencegah kebocoran data / data leakage).
    kolom_num = ["luas_m2", "kamar_tidur"]
    kolom_kat = ["lokasi"]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scl", StandardScaler())]), kolom_num),
            ("kat", OneHotEncoder(handle_unknown="ignore"), kolom_kat),
        ]
    )
    return X, y, pre


def main():
    print("=" * 60)
    print("PROYEK 1 - PREDIKSI HARGA RUMAH (REGRESI)")
    print("=" * 60)

    print("\n[Tahap 1] Data collection")
    df = kumpulkan_data()
    print(df.head().to_string())

    print("\n[Tahap 2] Preprocessing")
    X, y, pre = preprocessing(df)

    print("\n[Tahap 3] Data splitting (80% training, 20% testing)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"  Training: {len(X_train)} baris | Testing: {len(X_test)} baris")

    print("\n[Tahap 4-5] Model selection dan training: Linear Regression")
    model = Pipeline([("pre", pre), ("reg", LinearRegression())])
    model.fit(X_train, y_train)

    print("\n[Tahap 6] Evaluasi (MAE dan R2)")
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    mae_tr, r2_tr = mean_absolute_error(y_train, pred_train), r2_score(y_train, pred_train)
    mae_te, r2_te = mean_absolute_error(y_test, pred_test), r2_score(y_test, pred_test)
    print(f"  Training : MAE = {mae_tr:8.1f} juta | R2 = {r2_tr:.3f}")
    print(f"  Testing  : MAE = {mae_te:8.1f} juta | R2 = {r2_te:.3f}")
    selisih = r2_tr - r2_te
    if selisih > 0.10:
        print("  PERINGATAN: skor training jauh di atas testing -> indikasi overfitting.")
    else:
        print(f"  Selisih R2 training-testing = {selisih:.3f} -> tidak ada indikasi overfitting.")

    # Plot aktual vs prediksi
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred_test, alpha=0.6)
    batas = [y_test.min(), y_test.max()]
    plt.plot(batas, batas, "r--", label="Prediksi sempurna")
    plt.xlabel("Harga aktual (juta Rp)")
    plt.ylabel("Harga prediksi (juta Rp)")
    plt.title("Regresi harga rumah: aktual vs prediksi (data uji)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proyek1_aktual_vs_prediksi.png", dpi=120)
    plt.close()

    print("\n[Tahap 7] Inference: memprediksi rumah baru")
    rumah_baru = pd.DataFrame(
        {"luas_m2": [120], "kamar_tidur": [3], "lokasi": ["Kota B"]}
    )
    harga = model.predict(rumah_baru)[0]
    print(f"  Rumah 120 m2, 3 kamar, Kota B -> prediksi harga ~ Rp {harga:,.0f} juta")

    print("\nGrafik tersimpan di folder output/")


if __name__ == "__main__":
    main()
