# Panduan Penggunaan Aplikasi

## Persyaratan Sistem

- Python 3.9 atau lebih baru
- pip (Python package manager)

**Library yang dibutuhkan:**

```
streamlit
numpy
pandas
matplotlib
scikit-learn
```

---

## Instalasi

**1. Clone atau ekstrak folder proyek**

https://github.com/dzikribassyril/UTS-SoftComp-Heart-Disease

**2. Install seluruh dependency**

Buka terminal, arahkan ke direktori proyek, lalu jalankan:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Dari direktori proyek, jalankan perintah berikut di terminal:

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## Panduan Penggunaan Per Halaman

Navigasi antar halaman tersedia pada **sidebar** di sebelah kiri.

---

### Halaman 1 : Prediksi Pasien

Halaman ini digunakan untuk memprediksi risiko penyakit jantung seorang pasien menggunakan ketiga sistem sekaligus.

**Langkah penggunaan:**

1. Atur nilai input menggunakan slider yang tersedia:
   - Usia —> rentang 28 hingga 77 tahun
   - Kolesterol —> rentang 100 hingga 603 mg/dL
   - Max Heart Rate —> rentang 60 hingga 202 bpm

2. Klik tombol **Prediksi**.

3. Hasil akan ditampilkan untuk ketiga sistem:
   - FIS Manual : Sistem berbasis aturan yang dirancang secara manual
   - FIS + GA : FIS dengan parameter MF hasil optimasi Genetic Algorithm
   - FIS + ANN : FIS dengan parameter MF hasil ekstraksi dari ANN

4. Setiap sistem menampilkan Risk Score (nilai 0–1), label prediksi (Disease / No Disease), dan interpretasi tingkat risiko.

---

### Halaman 2 : Visualisasi MF

Halaman ini menampilkan bentuk Membership Function dari setiap tahap dan analisis pergeserannya.

**Tiga mode tampilan :**

- Per Tahap : Menampilkan MF Tahap 1, 2, dan 3 secara terpisah beserta tabel pergeseran parameter.

- Semua Tahap : Menampilkan overlay perbandingan MF sebelum dan sesudah optimasi. Pilih variabel (age, chol, thalch) dari dropdown untuk melihat pergeseran kurva secara detail. Tabel ringkasan pergeseran titik puncak (|Delta b|) tersedia di bagian bawah.

- Sensitivity Curves ANN : Menampilkan kurva sensitivitas yang dihasilkan ANN untuk setiap fitur input, beserta kurva training loss ANN.

---

### Halaman 3 : Performa Model

Halaman ini menampilkan perbandingan performa ketiga sistem pada data uji.

**Konten yang ditampilkan:**

- Tabel metrik (Accuracy, Precision, Recall, F1 Score) dengan highlight nilai terbaik.
- Bar chart perbandingan metrik secara visual.
- Analisis threshold sweep pada FIS Manual untuk menentukan threshold optimal.

---

### Halaman 4 : Ablation Study

Halaman ini menampilkan hasil eksperimen variasi parameter GA untuk analisis konvergensi.

**Konten yang ditampilkan:**

- Grafik konvergensi dengan variasi Population Size (gen = 50 tetap).
- Grafik konvergensi dengan variasi Jumlah Generasi (pop = 30 tetap).
- Tabel ringkasan best accuracy untuk setiap konfigurasi.
- Grafik Best vs Average Fitness untuk konfigurasi default (pop=50, gen=100).