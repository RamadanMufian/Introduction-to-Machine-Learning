# Praktikum Introduction to Machine Learning

Tiga mini-proyek Python yang menerapkan materi *Introduction to Machine Learning*
(Informatic Practitioner Lecture Series 2026), masing-masing mengikuti workflow ML
pada Bab 6: pengumpulan data, preprocessing, pembagian data, pemilihan model, training,
evaluasi, dan inference.

## Cara menjalankan

```bash
pip install -r requirements.txt

python main.py                          # jalankan ketiga proyek
# atau satu per satu:
python proyek1_regresi_harga_rumah.py
python proyek2_klasifikasi_spam.py
python proyek3_clustering_pelanggan.py
```

Grafik hasil disimpan otomatis di folder `output/`.

## Isi proyek

| File | Jenis ML | Materi | Algoritma | Metrik |
|---|---|---|---|---|
| `proyek1_regresi_harga_rumah.py` | Supervised, regresi | Bab 4.1 B, Bab 8 | Linear Regression | MAE, R² |
| `proyek2_klasifikasi_spam.py` | Supervised, klasifikasi | Bab 4.1 A | Logistic Regression vs Decision Tree + TF-IDF | Accuracy, Precision, Recall, F1, Confusion Matrix |
| `proyek3_clustering_pelanggan.py` | Unsupervised, clustering | Bab 4.2 A | K-Means | Inertia (elbow), Silhouette |

## Konsep materi yang muncul di kode

- **Fitur vs target**: `X` (fitur) dan `y` (target) pada proyek 1 dan 2.
- **Preprocessing**: missing value (imputer median), duplikat, one-hot encoding, standardisasi.
- **Data splitting**: 80% training, 20% testing; `stratify` pada klasifikasi.
- **Cek overfitting**: skor training dibandingkan skor testing.
- **Inference**: setiap proyek diakhiri prediksi data baru.
- **Pencegahan data leakage**: semua transformasi ada di dalam `Pipeline`, jadi hanya belajar dari data training.

## Catatan penting tentang data

Semua data **sintetis** (dibuat oleh kode) agar proyek langsung jalan tanpa unduhan,
dan angka hasilnya hanya ilustrasi, bukan hasil pengujian pada kasus nyata.
Untuk data asli, ganti fungsi `kumpulkan_data()` di tiap file dengan `pd.read_csv(...)`:

| Proyek | Dataset asli yang disarankan |
|---|---|
| 1 | House Prices (Kaggle) |
| 2 | SMS Spam Collection (UCI) |
| 3 | Mall Customers (Kaggle) |

## Ide pengembangan lanjutan

1. **Proyek 1**: harga bergantung pada interaksi luas dan lokasi, jadi coba `RandomForestRegressor`
   atau `PolynomialFeatures` lalu bandingkan MAE-nya.
2. **Proyek 2**: tambahkan cross-validation (`cross_val_score`) dan validation set (Bab 6, Tahap 3).
3. **Proyek 3**: tambahkan PCA (reduksi dimensi) untuk memvisualisasikan data berfitur banyak.
4. **Deployment** (Bab 6, Tahap 7): bungkus model dengan Streamlit atau simpan dengan `joblib`.
5. **Etika** (Bab 11): jika memakai data nyata, periksa bias dan privasi data sebelum model dipakai.
