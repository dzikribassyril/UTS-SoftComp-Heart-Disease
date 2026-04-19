#  Alur Mamdani FIS:
#  1. Fuzzifikasi   → hitung derajat keanggotaan tiap input
#  2. Rule Firing   → evaluasi tiap rule (AND = min)
#  3. Agregasi      → clip output MF per rule, gabungkan (max)
#  4. Defuzzifikasi → centroid → risk_score (0–1)
#  5. Klasifikasi   → risk_score >= threshold → label 0/1
# ============================================================

import numpy as np
import pandas as pd

from utils.config import (
    MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL,
    RULES_MANUAL, DECISION_THRESHOLD,
    FEATURE_RANGES
)
from utils.evaluate import compute_metrics, print_report, threshold_sweep
from utils.preprocessing import load_and_prepare


# ============================================================
# BAGIAN 1 – Fuzzy Inference Engine
# ============================================================

def _trimf(x: float, abc: list) -> float:
    """Triangular MF untuk satu nilai skalar x."""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:  # b < x < c
        return (c - x) / (c - b) if c != b else 0.0


def _trimf_array(x: np.ndarray, abc: list) -> np.ndarray:
    """Triangular MF untuk array numpy (digunakan saat defuzzifikasi)."""
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    m1 = (x > a)  & (x <= b)
    m2 = (x > b)  & (x < c)
    if b != a: y[m1] = (x[m1] - a) / (b - a)
    else:      y[x == b] = 1.0
    if c != b: y[m2] = (c - x[m2]) / (c - b)
    return y


# ----------------------------------------------------------
# 1. Fuzzifikasi
# ----------------------------------------------------------
def fuzzify(sample: dict,
            mf_params: dict = None) -> dict:
    """
    Menghitung derajat keanggotaan semua variabel input.

    Parameters
    ----------
    sample    : dict {var_name: nilai_crisp}
                mis. {"age": 55, "chol": 240, "thalch": 130}
    mf_params : dict MF; default = MF_PARAMS_MANUAL

    Returns
    -------
    dict {var_name: {label: derajat_keanggotaan}}
    mis. {"age": {"young": 0.0, "middle": 0.4, "old": 0.7}, ...}
    """
    if mf_params is None:
        mf_params = MF_PARAMS_MANUAL

    result = {}
    for var, labels in mf_params.items():
        crisp_val = float(sample.get(var, 0))
        result[var] = {
            label: _trimf(crisp_val, abc)
            for label, abc in labels.items()
        }
    return result


# ----------------------------------------------------------
# 2. Rule Firing  (AND = min)
# ----------------------------------------------------------
def fire_rules(fuzz_values: dict,
               rules: list = None) -> list:
    """
    Evaluasi setiap rule dan hitung firing strength-nya.

    Parameters
    ----------
    fuzz_values : output dari fuzzify()
    rules       : list of (antecedents_dict, consequent_label)

    Returns
    -------
    list of (firing_strength: float, consequent: str)
    """
    if rules is None:
        rules = RULES_MANUAL

    fired = []
    for antecedents, consequent in rules:
        strengths = []
        for var, label in antecedents.items():
            membership = fuzz_values.get(var, {}).get(label, 0.0)
            strengths.append(membership)

        firing_strength = min(strengths) if strengths else 0.0
        fired.append((firing_strength, consequent))
    return fired


# ----------------------------------------------------------
# 3. Agregasi output MF
# ----------------------------------------------------------
def aggregate(fired_rules: list,
              output_mf: dict = None,
              resolution: int = 200) -> tuple:
    """
    Clip setiap output MF berdasarkan firing strength,
    lalu agregasikan dengan operasi max.

    Returns
    -------
    x_out  : np.ndarray  – sumbu output (0.0–1.0)
    y_agg  : np.ndarray  – MF teragregasi
    """
    if output_mf is None:
        output_mf = OUTPUT_MF_MANUAL

    x_out = np.linspace(0.0, 1.0, resolution)
    y_agg = np.zeros(resolution)

    for strength, consequent in fired_rules:
        if strength == 0.0:
            continue
        abc    = output_mf[consequent]
        y_mf   = _trimf_array(x_out, abc)
        y_clip = np.minimum(strength, y_mf)   # Mamdani: clip (min)
        y_agg  = np.maximum(y_agg, y_clip)    # Agregasi: max

    return x_out, y_agg


# ----------------------------------------------------------
# 4. Defuzzifikasi  (Centroid / Center of Gravity)
# ----------------------------------------------------------
def defuzzify(x_out: np.ndarray, y_agg: np.ndarray) -> float:
    """
    Centroid defuzzification.

    Returns
    -------
    risk_score : float 0–1
    """
    numerator   = np.sum(x_out * y_agg)
    denominator = np.sum(y_agg)

    if denominator == 0:
        return 0.0  # tidak ada rule yang aktif → risk rendah
    return float(numerator / denominator)


# ============================================================
# BAGIAN 2 – Predict (satu sampel & batch)
# ============================================================

def predict_one(sample: dict,
                mf_params: dict  = None,
                output_mf: dict  = None,
                rules: list      = None,
                threshold: float = None) -> tuple:
    """
    Prediksi satu sampel.

    Parameters
    ----------
    sample    : dict {var_name: crisp_value}
    threshold : jika None, gunakan DECISION_THRESHOLD dari config

    Returns
    -------
    (risk_score: float, label: int)
        risk_score = nilai defuzzifikasi (0–1)
        label      = 0 (no disease) / 1 (disease)
    """
    if threshold is None:
        threshold = DECISION_THRESHOLD
    if mf_params is None:
        mf_params = MF_PARAMS_MANUAL
    if output_mf is None:
        output_mf = OUTPUT_MF_MANUAL
    if rules is None:
        rules = RULES_MANUAL

    fuzz_vals  = fuzzify(sample, mf_params)
    fired      = fire_rules(fuzz_vals, rules)
    x_out, y_a = aggregate(fired, output_mf)
    score      = defuzzify(x_out, y_a)
    label      = int(score >= threshold)

    return score, label


def predict_batch(X: pd.DataFrame,
                  mf_params: dict  = None,
                  output_mf: dict  = None,
                  rules: list      = None,
                  threshold: float = None) -> tuple:
    """
    Prediksi batch DataFrame.

    Returns
    -------
    scores : np.ndarray float  – risk score tiap sampel
    labels : np.ndarray int    – prediksi biner tiap sampel
    """
    scores = []
    labels = []

    for _, row in X.iterrows():
        sample = row.to_dict()
        s, l   = predict_one(sample, mf_params, output_mf, rules, threshold)
        scores.append(s)
        labels.append(l)

    return np.array(scores), np.array(labels)


# ============================================================
# BAGIAN 3 – Evaluasi & Simpan Hasil
# ============================================================

def evaluate_stage1(X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """
    Jalankan FIS manual pada test set dan tampilkan metrik.

    Returns
    -------
    dict {
        "metrics"      : dict metrik (accuracy, precision, ...),
        "risk_scores"  : np.ndarray,
        "predictions"  : np.ndarray,
    }
    """
    print("\n[Stage 1] Menjalankan Manual FIS pada test set...")
    scores, preds = predict_batch(X_test)

    metrics = compute_metrics(y_test, preds, model_name="FIS Manual")
    print_report(y_test, preds, model_name="FIS Manual")

    # Threshold sweep → cari threshold terbaik
    print("\n[Stage 1] Threshold sweep:")
    sweep_df = threshold_sweep(y_test, scores)
    best_row  = sweep_df.loc[sweep_df["f1_score"].idxmax()]
    print(sweep_df.to_string(index=False))
    print(f"\n  → Threshold terbaik (F1): {best_row['threshold']} "
          f"(Acc={best_row['accuracy']}, F1={best_row['f1_score']})")

    return {
        "metrics"    : metrics,
        "risk_scores": scores,
        "predictions": preds,
        "sweep_df"   : sweep_df,
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, _ = load_and_prepare()

    # Evaluasi Tahap 1
    result = evaluate_stage1(X_test, y_test)

    print(f"\n[Stage 1] Accuracy  : {result['metrics']['accuracy']}")
    print(f"[Stage 1] F1 Score  : {result['metrics']['f1_score']}")
    print(f"[Stage 1] Precision : {result['metrics']['precision']}")
    print(f"[Stage 1] Recall    : {result['metrics']['recall']}")

    # Contoh prediksi satu pasien
    pasien_contoh = {"age": 60, "chol": 290, "thalch": 110}
    score, label  = predict_one(pasien_contoh)
    print(f"\n[Demo] Pasien {pasien_contoh}")
    print(f"  Risk Score : {score:.4f}")
    print(f"  Prediksi   : {'DISEASE' if label == 1 else 'NO DISEASE'}")
