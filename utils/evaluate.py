# ============================================================
#  evaluate.py  –  untuk Metrics & model comparison utilities
#  Heart Disease Risk Prediction | Soft Computing UTS 2025/2026
#  Anggota:
#    1. 140810230008 – Robby Azwan Saputra
#    2. 140810230071 – Dzikri Basyril Mu'Minin 
#    3. 140810230074 – Farhan Zia Rizky
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# ------------------------------------------------------------
# 1. Compute single-model metrics
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Menghitung accuracy, precision, recall, F1 untuk prediksi biner.

    Parameters
    ----------
    y_true      : array-like  – label asli (0/1)
    y_pred      : array-like  – label prediksi (0/1)
    model_name  : str         – nama model (untuk display)

    Returns
    -------
    dict berisi semua metrik
    """
    metrics = {
        "model"    : model_name,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score" : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    return metrics


# ------------------------------------------------------------
# 2. Confusion matrix
# ------------------------------------------------------------
def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Mengembalikan confusion matrix 2×2 (TN, FP, FN, TP)."""
    return confusion_matrix(y_true, y_pred)


# ------------------------------------------------------------
# 3. Bandingkan beberapa model sekaligus
# ------------------------------------------------------------
def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Menerima list dict dari compute_metrics() dan mengembalikan
    DataFrame perbandingan, diurutkan dari akurasi tertinggi.

    Contoh pemakaian:
        r1 = compute_metrics(y_true, pred_fis,  "FIS Manual")
        r2 = compute_metrics(y_true, pred_ga,   "FIS + GA")
        r3 = compute_metrics(y_true, pred_ann,  "FIS + ANN")
        df = compare_models([r1, r2, r3])
    """
    df = pd.DataFrame(results)
    df.set_index("model", inplace=True)
    df.sort_values("accuracy", ascending=False, inplace=True)
    return df


# ------------------------------------------------------------
# 4. Print laporan lengkap ke console
# ------------------------------------------------------------
def print_report(y_true, y_pred, model_name: str = "Model"):
    """Cetak classification report scikit-learn ke console."""
    print(f"\n{'='*50}")
    print(f"  Evaluation Report: {model_name}")
    print(f"{'='*50}")
    print(classification_report(
        y_true, y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0
    ))
    cm = get_confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")


# ------------------------------------------------------------
# 5. Hitung performa per-threshold (untuk analisis FIS)
# ------------------------------------------------------------
def threshold_sweep(y_true, risk_scores: np.ndarray,
                    thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Sweeping nilai threshold pada risk_score FIS untuk mencari
    threshold optimal.

    Parameters
    ----------
    risk_scores : array float 0–1 (output defuzzifikasi)
    thresholds  : array threshold yang ingin dicoba

    Returns
    -------
    DataFrame dengan kolom: threshold, accuracy, f1_score
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (risk_scores >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "accuracy" : round(accuracy_score(y_true, y_pred), 4),
            "f1_score" : round(f1_score(y_true, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Data dummy
    y_true   = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred_a = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    y_pred_b = np.array([0, 1, 1, 0, 1, 0, 0, 1])

    r1 = compute_metrics(y_true, y_pred_a, "FIS Manual")
    r2 = compute_metrics(y_true, y_pred_b, "FIS + GA")
    print(compare_models([r1, r2]))

    print_report(y_true, y_pred_a, "FIS Manual")
