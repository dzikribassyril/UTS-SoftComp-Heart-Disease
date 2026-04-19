# ============================================================
#  preprocessing.py  –  Data loading & preparation
#  Heart Disease Risk Prediction | Soft Computing UTS 2025/2026
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.config import (
    DATASET_PATH, FEATURES, TARGET_COL,
    BINARY_TARGET, TEST_SIZE, RANDOM_STATE
)


# ------------------------------------------------------------
# 1. Load raw CSV
# ------------------------------------------------------------
def load_raw(path: str = DATASET_PATH) -> pd.DataFrame:
    """Memuat file CSV mentah."""
    candidate_paths = []

    # 1) Path dari argumen/config (bisa relatif atau absolut).
    given = Path(path)
    candidate_paths.append(given)
    if not given.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate_paths.append(project_root / given)

    # 2) Fallback nama file umum pada folder data.
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths.extend([
        project_root / "data" / "heart_disease.csv",
        project_root / "data" / "heart_disease_uci.csv",
    ])

    existing_path = next((p for p in candidate_paths if p.exists()), None)
    if existing_path is None:
        checked = "\n".join(f"- {str(p)}" for p in candidate_paths)
        raise FileNotFoundError(
            "Dataset tidak ditemukan. Path yang dicek:\n"
            f"{checked}"
        )

    df = pd.read_csv(existing_path)
    return df


# ------------------------------------------------------------
# 2. Clean & select relevant columns
# ------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Pilih kolom fitur + target
    - Hapus baris dengan NaN pada fitur utama
    - Clip nilai outlier ekstrem (chol=0 tidak valid secara medis)
    """
    cols = FEATURES + [TARGET_COL]
    df = df[cols].copy()

    # Chol = 0 tidak valid → ganti dengan NaN lalu drop
    df.loc[df["chol"] == 0, "chol"] = np.nan

    # Drop baris yang ada NaN pada kolom fitur mana pun
    df.dropna(subset=FEATURES, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ------------------------------------------------------------
# 3. Binarize target
# ------------------------------------------------------------
def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi target num (0–4) menjadi label biner:
        0  → 0  (tidak ada penyakit jantung)
        >0 → 1  (ada penyakit jantung)
    """
    if BINARY_TARGET:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    return df


# ------------------------------------------------------------
# 4. Train / test split
# ------------------------------------------------------------
def split(df: pd.DataFrame):
    """
    Mengembalikan:
        X_train, X_test  → pd.DataFrame fitur
        y_train, y_test  → pd.Series target
    """
    X = df[FEATURES]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 5. Pipeline utama (one-call convenience)
# ------------------------------------------------------------
def load_and_prepare(path: str = DATASET_PATH):
    """
    Pipeline lengkap: load → clean → binarize → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, df_clean
    """
    df = load_raw(path)
    df = clean(df)
    df = binarize_target(df)

    X_train, X_test, y_train, y_test = split(df)

    print(f"[preprocessing] Total sampel bersih : {len(df)}")
    print(f"[preprocessing] Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"[preprocessing] Label distribusi (test):\n{y_test.value_counts().to_string()}")

    return X_train, X_test, y_train, y_test, df


# ------------------------------------------------------------
# 6. Helper: ambil array fitur tunggal (untuk FIS)
# ------------------------------------------------------------
def get_feature_arrays(df: pd.DataFrame):
    """
    Mengembalikan dict {nama_fitur: np.array} untuk kemudahan akses FIS.
    """
    return {feat: df[feat].values for feat in FEATURES}


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_and_prepare()
    print("\nSampel X_train (5 baris pertama):")
    print(X_train.head())
    print("\nDistribusi kelas (seluruh data):")
    print(df[TARGET_COL].value_counts())
