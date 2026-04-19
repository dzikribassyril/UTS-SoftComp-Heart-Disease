# Heart Disease Risk Prediction
### The Intelligence Battle: Human Expert vs. Evolutionary Tuning & Neuro-Fuzzy

> UTS Soft Computing -- Program Studi S-1 Teknik Informatika  
> Universitas Padjadjaran

---

## Tim

| NIM | Nama |
|---|---|
| 140810230008 | Robby Azwan Saputra |
| 140810230071 | Dzikri Basyril Mu'Minin |
| 140810230074 | Farhan Zia Rizky |

**Dosen Pengampu:** Dr. Ir. Intan Nurma Yulita, M.T  
**Program Studi:** S-1 Teknik Informatika FMIPA Universitas Padjadjaran

---

## Overview

Proyek ini membandingkan tiga pendekatan dalam membangun sistem prediksi risiko penyakit jantung berbasis Fuzzy Inference System (FIS):

| Tahap | Metode | Deskripsi |
|---|---|---|
| 1 | FIS Manual | MF dan Rules dirancang menggunakan intuisi pakar |
| 2 | FIS + Genetic Algorithm | Parameter MF dioptimasi menggunakan GA |
| 3 | FIS + Artificial Neural Network | Parameter MF diekstrak dari kurva sensitivitas ANN |

Dataset yang digunakan adalah Heart Disease UCI dengan fitur input: `age`, `chol`, dan `thalch`.
Sumber dataset: [Heart Disease UCI (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## Hasil Performa

| Sistem | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| FIS Manual | 57% | 0.53 | 0.81 | 0.64 |
| FIS + GA | 69%** | 0.71 | 0.67 | 0.67 |
| FIS + ANN | 63% | 0.60 | 0.90*| 0.72 |

---

## Struktur Proyek

```
heart_disease_project/
│
├── app.py                    # Aplikasi Streamlit utama
├── heart_disease_uci.csv     # Dataset
│
├── utils/
│   ├── config.py             # Konfigurasi global (MF params, rules, hyperparams)
│   ├── preprocessing.py      # Load, clean, split dataset
│   ├── evaluate.py           # Metrik evaluasi dan threshold sweep
│   └── plot_mf.py            # Visualisasi MF dan konvergensi GA
│
└── models/
    ├── fis.py                # Tahap 1 — Manual Mamdani FIS
    ├── ga.py                 # Tahap 2 — Genetic Algorithm Tuning
    └── ann.py                # Tahap 3 — ANN Sensitivity-based Tuning
```

---

## Instalasi

Panduan Penggunaan Aplikasi: [Buka file panduan](Panduan%20Penggunaan%20Aplikasi.md)