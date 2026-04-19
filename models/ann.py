#  Pendekatan: ANN-guided MF Optimization (Sensitivity Analysis)
#  Anggota:
#    1. 140810230008 – Robby Azwan Saputra
#    2. 140810230071 – Dzikri Basyril Mu'Minin 
#    3. 140810230074 – Farhan Zia Rizky
#
#  Langkah:
#  1. Latih MLPClassifier pada fitur [age, chol, thalch]
#  2. Untuk setiap fitur, buat kurva sensitivitas:
#       P(disease | feature=x, fitur_lain=median)
#  3. Temukan 2 titik transisi dari kurva → update batas MF
#  4. Susun kembali trimf baru dari titik transisi tersebut
#  5. Evaluasi FIS dengan MF baru vs FIS manual
#
#  Alasan pendekatan ini:
#  - MF manual dibuat dari intuisi; ANN mempelajari pola data nyata
#  - Titik transisi ANN ↔ "batas keputusan" data-driven
#  - Hasilnya adalah MF yang lebih sesuai distribusi data
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from utils.config import ( 
    MF_PARAMS_MANUAL,FEATURE_RANGES, ANN_CONFIG,
    FEATURES, RANDOM_STATE
)
from models.ga import fis_predict_vectorized
from utils.evaluate import compute_metrics, print_report


# ============================================================
# BAGIAN 1 – Build & Train ANN
# ============================================================

def build_ann_model(hidden_layers: tuple = None,
                    activation: str = None,
                    learning_rate: float = None,
                    max_iter: int = None) -> MLPClassifier:
    """
    Buat MLPClassifier.

    Parameters
    ----------
    hidden_layers  : tuple ukuran tiap hidden layer, contoh (64, 32)
    activation     : fungsi aktivasi ('relu', 'tanh', 'logistic')
    learning_rate  : laju pembelajaran awal
    max_iter       : maks epoch

    Returns
    -------
    MLPClassifier yang belum di-fit
    """
    cfg = ANN_CONFIG
    model = MLPClassifier(
        hidden_layer_sizes = hidden_layers  or cfg["hidden_layers"],
        activation         = activation     or cfg["activation"],
        max_iter           = max_iter       or cfg["max_iter"],
        learning_rate_init = learning_rate  or cfg["learning_rate"],
        solver             = "adam",
        random_state       = RANDOM_STATE,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 20,
        verbose            = False,
    )
    return model


def train_ann(X_train: pd.DataFrame,
              y_train: pd.Series,
              hidden_layers: tuple = None,
              verbose: bool = True) -> tuple:
    """
    Latih ANN dan kembalikan model + scaler.

    ANN membutuhkan fitur yang dinormalisasi (MinMaxScaler)
    agar gradient tidak dominasi oleh fitur dengan range besar.

    Returns
    -------
    (model, scaler)
        model  : MLPClassifier yang sudah di-fit
        scaler : MinMaxScaler yang di-fit pada X_train
    """
    X_arr = X_train[FEATURES].values
    y_arr = y_train.values

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_arr)

    model = build_ann_model(hidden_layers)
    model.fit(X_scaled, y_arr)

    if verbose:
        train_acc = accuracy_score(y_arr, model.predict(X_scaled))
        print(f"[ANN] Training selesai | "
              f"Epoch: {model.n_iter_} | "
              f"Train acc: {train_acc:.4f}")

    return model, scaler


# ============================================================
# BAGIAN 2 – Sensitivity Analysis
# ============================================================

def sensitivity_curve(model: MLPClassifier,
                      scaler: MinMaxScaler,
                      feature_name: str,
                      X_ref: pd.DataFrame,
                      n_points: int = 200) -> tuple:
    """
    Hitung kurva sensitivitas: P(disease=1 | feature=x, lainnya=median).

    Semua fitur lain di-fix pada nilai median dari X_ref,
    hanya feature_name yang divariasikan dari lo hingga hi.

    Parameters
    ----------
    feature_name : nama fitur yang akan disweep
    X_ref        : DataFrame referensi (gunakan X_train)
    n_points     : resolusi sweep

    Returns
    -------
    x_sweep   : np.ndarray (n_points,) – nilai feature yang disweep
    prob_curve: np.ndarray (n_points,) – P(disease=1)
    """
    lo, hi   = FEATURE_RANGES[feature_name]
    x_sweep  = np.linspace(lo, hi, n_points)

    # Buat sampel referensi: fitur lain = median
    medians = {f: float(X_ref[f].median()) for f in FEATURES}

    # Bangun matrix (n_points × n_features)
    X_sweep = np.zeros((n_points, len(FEATURES)))
    for i, feat in enumerate(FEATURES):
        if feat == feature_name:
            X_sweep[:, i] = x_sweep
        else:
            X_sweep[:, i] = medians[feat]

    X_scaled = scaler.transform(X_sweep)
    proba    = model.predict_proba(X_scaled)

    # Indeks class 1 (disease)
    class_idx = list(model.classes_).index(1)
    prob_curve = proba[:, class_idx]

    return x_sweep, prob_curve


# ============================================================
# BAGIAN 3 – Ekstraksi MF dari Kurva Sensitivitas
# ============================================================

def find_transition_points(x_sweep: np.ndarray,
                            prob_curve: np.ndarray,
                            t1: float = 0.35,
                            t2: float = 0.65) -> tuple:
    """
    Temukan 2 titik transisi dari kurva sensitivitas.

    x1 : titik pertama di mana prob_curve melewati t1
    x2 : titik pertama di mana prob_curve melewati t2

    Jika tidak ditemukan → fallback ke persentil 33 & 67.

    Returns
    -------
    (x1, x2) : float, float
    """
    lo, hi  = float(x_sweep[0]), float(x_sweep[-1])

    # Cari crossing pertama threshold t1
    x1 = None
    for i in range(len(prob_curve) - 1):
        if prob_curve[i] < t1 <= prob_curve[i + 1]:
            # Interpolasi linear
            frac = (t1 - prob_curve[i]) / (prob_curve[i+1] - prob_curve[i])
            x1   = x_sweep[i] + frac * (x_sweep[i+1] - x_sweep[i])
            break
    if x1 is None:
        x1 = lo + (hi - lo) * 0.33   # fallback persentil 33

    # Cari crossing pertama threshold t2
    x2 = None
    for i in range(len(prob_curve) - 1):
        if prob_curve[i] < t2 <= prob_curve[i + 1]:
            frac = (t2 - prob_curve[i]) / (prob_curve[i+1] - prob_curve[i])
            x2   = x_sweep[i] + frac * (x_sweep[i+1] - x_sweep[i])
            break
    if x2 is None:
        x2 = lo + (hi - lo) * 0.67   # fallback persentil 67

    # Pastikan x1 < x2
    if x1 >= x2:
        x1, x2 = lo + (hi - lo) * 0.33, lo + (hi - lo) * 0.67

    return float(x1), float(x2)


def build_mf_from_transitions(feature_name: str,
                               x1: float,
                               x2: float,
                               label_names: list) -> dict:
    """
    Susun 3 trimf (low/medium/high) dari 2 titik transisi x1 dan x2.

    Strategi (untuk 3 label berurutan):
        label[0] (low)   : peak = lo,   fade ke x1
        label[1] (medium): peak = midpoint(x1, x2)
        label[2] (high)  : peak = hi,   fade dari x2

    Parameters
    ----------
    feature_name  : nama fitur
    x1, x2        : titik transisi dari sensitivity analysis
    label_names   : list 3 nama label, mis. ['young','middle','old']

    Returns
    -------
    dict {label: [a, b, c]}
    """
    lo, hi = FEATURE_RANGES[feature_name]
    mid    = (x1 + x2) / 2.0
    margin = max((x2 - x1) * 0.15, (hi - lo) * 0.02)

    mf = {}
    mf[label_names[0]] = [lo,        lo,         x1 + margin]
    mf[label_names[1]] = [x1 - margin, mid,      x2 + margin]
    mf[label_names[2]] = [x2 - margin, hi,        hi        ]

    # Clip semua ke [lo, hi] dan pastikan a ≤ b ≤ c
    for lbl in mf:
        a, b, c = mf[lbl]
        a = max(lo, min(a, hi))
        b = max(lo, min(b, hi))
        c = max(lo, min(c, hi))
        a, b, c = sorted([a, b, c])
        mf[lbl] = [round(a, 3), round(b, 3), round(c, 3)]

    return mf


def extract_mf_from_ann(model: MLPClassifier,
                         scaler: MinMaxScaler,
                         X_ref: pd.DataFrame,
                         verbose: bool = True) -> tuple:
    """
    Pipeline utama ekstraksi MF dari ANN.

    Untuk setiap fitur:
        1. Hitung sensitivity curve
        2. Cari 2 titik transisi
        3. Bangun MF baru

    Returns
    -------
    new_mf_params : dict MF params baru
    sensitivity   : dict {feature: (x_sweep, prob_curve)} untuk plotting
    """
    new_mf_params = {}
    sensitivity   = {}

    for var in FEATURES:
        label_names = list(MF_PARAMS_MANUAL[var].keys())

        x_sweep, prob = sensitivity_curve(model, scaler, var, X_ref)
        x1, x2        = find_transition_points(x_sweep, prob)
        mf_new        = build_mf_from_transitions(var, x1, x2, label_names)

        new_mf_params[var] = mf_new
        sensitivity[var]   = (x_sweep, prob)

        if verbose:
            print(f"  [{var}] transisi: x1={x1:.1f}, x2={x2:.1f}")
            for lbl, abc in mf_new.items():
                print(f"         {lbl:>12}: {[round(v,1) for v in abc]}")

    return new_mf_params, sensitivity


# ============================================================
# BAGIAN 4 – Plot Sensitivity Curves (opsional, untuk laporan)
# ============================================================

def plot_sensitivity_curves(sensitivity: dict) -> object:
    """
    Plot kurva sensitivitas untuk semua fitur.
    Menunjukkan bagaimana ANN 'melihat' pengaruh tiap fitur.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sensitivity)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#F44336"]

    for ax, (var, (x_sweep, prob)) in zip(axes, sensitivity.items()):
        ax.plot(x_sweep, prob, color=colors[list(sensitivity).index(var)],
                linewidth=2.5, label="P(disease=1)")
        ax.axhline(0.35, color="gray", linestyle="--",
                   linewidth=1, label="t1=0.35")
        ax.axhline(0.65, color="gray", linestyle=":",
                   linewidth=1, label="t2=0.65")
        ax.fill_between(x_sweep, 0, prob, alpha=0.12,
                        color=colors[list(sensitivity).index(var)])

        ax.set_title(f"Sensitivitas – {var.upper()}", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel(var, fontsize=9)
        ax.set_ylabel("P(disease=1)", fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ANN Sensitivity Analysis – Pengaruh Tiap Fitur terhadap Risiko",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# BAGIAN 5 – Pipeline Lengkap Tahap 3
# ============================================================

def run_ann_tuning(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   hidden_layers: tuple = None,
                   verbose: bool = True) -> dict:
    """
    Pipeline lengkap Tahap 3:
        latih ANN → analisis sensitivitas → ekstrak MF → evaluasi FIS.

    Returns
    -------
    dict {
        "ann_model"       : MLPClassifier terlatih,
        "scaler"          : MinMaxScaler,
        "best_mf_params"  : dict MF hasil tuning ANN,
        "sensitivity"     : dict kurva sensitivitas,
        "ann_metrics"     : dict metrik ANN langsung (tanpa FIS),
        "loss_history"    : list[float] loss per epoch,
    }
    """
    if verbose:
        print("\n[ANN] Melatih MLP Classifier...")

    # ── Latih ANN ─────────────────────────────────────────────
    model, scaler = train_ann(X_train, y_train,
                              hidden_layers=hidden_layers,
                              verbose=verbose)

    # ── Akurasi ANN langsung (tanpa FIS) ──────────────────────
    X_arr_s   = scaler.transform(X_train[FEATURES].values)
    ann_preds = model.predict(X_arr_s)
    ann_met   = compute_metrics(y_train, ann_preds, "ANN Direct (train)")

    if verbose:
        print(f"\n[ANN] Analisis sensitivitas & ekstraksi MF...")

    # ── Ekstrak MF dari ANN ────────────────────────────────────
    new_mf, sensitivity = extract_mf_from_ann(
        model, scaler, X_train, verbose=verbose
    )

    loss_history = (model.loss_curve_
                    if hasattr(model, "loss_curve_") else [])

    if verbose:
        print(f"\n[ANN] MF baru berhasil diekstrak.")

    return {
        "ann_model"      : model,
        "scaler"         : scaler,
        "best_mf_params" : new_mf,
        "sensitivity"    : sensitivity,
        "ann_metrics"    : ann_met,
        "loss_history"   : loss_history,
    }


# ============================================================
# BAGIAN 6 – Utilitas Analisis
# ============================================================

def summarize_mf_shift(mf_before: dict, mf_after: dict) -> pd.DataFrame:
    """
    Tabel pergeseran titik puncak (b) setiap MF.
    Berguna untuk bagian 'Analisis Pergeseran Kurva' di laporan.
    """
    rows = []
    for var in FEATURES:
        for lbl in mf_before[var]:
            ab = mf_before[var][lbl]
            aa = mf_after.get(var, {}).get(lbl, ab)
            rows.append({
                "variable" : var,
                "label"    : lbl,
                "b_before" : round(ab[1], 2),
                "b_after"  : round(aa[1], 2),
                "|Δb|"     : round(abs(aa[1] - ab[1]), 2),
            })
    return pd.DataFrame(rows)


def plot_loss_curve(loss_history: list) -> object:
    """Plot ANN training loss per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(loss_history, color="#F44336", linewidth=2)
    ax.set_title("ANN Training Loss Curve", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (Cross-Entropy)")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    from utils.preprocessing import load_and_prepare
    from utils.plot_mf import plot_mf_comparison

    X_train, X_test, y_train, y_test, _ = load_and_prepare()

    print("=" * 50)
    print("  TAHAP 3 – ANN TUNING")
    print("=" * 50)

    result = run_ann_tuning(X_train, y_train, verbose=True)

    # ── Evaluasi FIS + MF baru pada test set ──────────────────
    X_test_arr = X_test[FEATURES].values.astype(np.float32)
    _, preds   = fis_predict_vectorized(X_test_arr, result["best_mf_params"])
    metrics    = compute_metrics(y_test, preds, "FIS + ANN")
    print_report(y_test, preds, "FIS + ANN")

    # ── Pergeseran MF ─────────────────────────────────────────
    shift_df = summarize_mf_shift(MF_PARAMS_MANUAL, result["best_mf_params"])
    print("\nPergeseran titik puncak MF (|Δb|):")
    print(shift_df.to_string(index=False))

    # ── ANN Direct (tanpa FIS) ────────────────────────────────
    X_test_s  = result["scaler"].transform(X_test[FEATURES].values)
    ann_preds = result["ann_model"].predict(X_test_s)
    ann_met   = compute_metrics(y_test, ann_preds, "ANN Direct (test)")
    print(f"\n[ANN Direct] Test acc: {ann_met['accuracy']} | "
          f"F1: {ann_met['f1_score']}")
