# ============================================================
#  plot_mf.py  –  Membership Function visualization
#  Heart Disease Risk Prediction | Soft Computing UTS 2025/2026
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (untuk Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.config import FEATURE_RANGES, MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL


# ------------------------------------------------------------
# Helper: hitung nilai trimf
# ------------------------------------------------------------
def trimf(x: np.ndarray, abc: list) -> np.ndarray:
    """
    Triangular membership function.
    abc = [a, b, c]  →  naik dari a ke b, turun dari b ke c.
    """
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    # naik
    mask1 = (x >= a) & (x <= b)
    if b != a:
        y[mask1] = (x[mask1] - a) / (b - a)
    else:
        y[mask1] = 1.0
    # turun
    mask2 = (x > b) & (x <= c)
    if c != b:
        y[mask2] = (c - x[mask2]) / (c - b)
    else:
        y[mask2] = 0.0
    return y


# ------------------------------------------------------------
# 1. Plot MF satu variabel
# ------------------------------------------------------------
def plot_variable_mf(ax, var_name: str, mf_params: dict,
                     title_suffix: str = ""):
    """
    Plot semua MF untuk satu variabel input/output pada ax.

    Parameters
    ----------
    var_name    : nama variabel ('age', 'chol', 'thalch', 'risk')
    mf_params   : dict {label: [a,b,c]}
    title_suffix: teks tambahan di judul (mis. 'Manual', 'After GA')
    """
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]
    label_colors = {}

    if var_name == "risk":
        lo, hi = 0.0, 1.0
    else:
        lo, hi = FEATURE_RANGES.get(var_name, (0, 1))

    x = np.linspace(lo, hi, 500)

    for i, (label, abc) in enumerate(mf_params.items()):
        color = colors[i % len(colors)]
        label_colors[label] = color
        y = trimf(x, abc)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        # Titik puncak
        peak_x = abc[1]
        ax.axvline(peak_x, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title(f"{var_name.upper()}  {title_suffix}", fontsize=11, fontweight="bold")
    ax.set_xlabel(var_name, fontsize=9)
    ax.set_ylabel("Derajat Keanggotaan", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------------
# 2. Plot semua variabel dalam satu figure (untuk satu tahap)
# ------------------------------------------------------------
def plot_all_mf(mf_input: dict, mf_output: dict,
                suptitle: str = "Membership Functions") -> plt.Figure:
    """
    Plot grid MF untuk semua input + output.

    Parameters
    ----------
    mf_input  : dict {var_name: {label: [a,b,c]}}
    mf_output : dict {label: [a,b,c]}
    suptitle  : judul figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_vars = len(mf_input) + 1  # +1 untuk output
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    # Input variables
    for ax, (var_name, mf_params) in zip(axes[:-1], mf_input.items()):
        plot_variable_mf(ax, var_name, mf_params)

    # Output variable
    plot_variable_mf(axes[-1], "risk", mf_output)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 3. Plot perbandingan MF sebelum vs sesudah optimasi (Analisis
#    Pergeseran Kurva — wajib ada di laporan)
# ------------------------------------------------------------
def plot_mf_comparison(var_name: str,
                       mf_before: dict,
                       mf_after : dict,
                       stage_label: str = "Optimasi") -> plt.Figure:
    """
    Overlay MF sebelum (manual) dan sesudah (GA/ANN) untuk satu
    variabel.  Menggunakan warna solid = sebelum, putus-putus = sesudah.

    Parameters
    ----------
    var_name    : nama variabel
    mf_before   : dict {label: [a,b,c]}  ← MF manual (Tahap 1)
    mf_after    : dict {label: [a,b,c]}  ← MF hasil optimasi
    stage_label : 'GA' atau 'ANN'
    """
    if var_name == "risk":
        lo, hi = 0.0, 1.0
    else:
        lo, hi = FEATURE_RANGES.get(var_name, (0, 1))

    x = np.linspace(lo, hi, 500)
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7, 4))
    legend_handles = []

    for i, label in enumerate(mf_before.keys()):
        color = colors[i % len(colors)]

        # Sebelum (solid)
        y_before = trimf(x, mf_before[label])
        line_b, = ax.plot(x, y_before, color=color, linewidth=2.5,
                          linestyle="-", label=f"{label} (Manual)")

        # Sesudah (dashed), jika tersedia
        if label in mf_after:
            y_after = trimf(x, mf_after[label])
            ax.plot(x, y_after, color=color, linewidth=2,
                    linestyle="--", alpha=0.8, label=f"{label} ({stage_label})")

            # Isi area pergeseran
            ax.fill_between(x, y_before, y_after,
                            color=color, alpha=0.08)

    # Legend manual (lebih rapi)
    solid_patch  = mpatches.Patch(color="gray", label="Manual (solid)")
    dashed_patch = mpatches.Patch(facecolor="white",
                                  edgecolor="gray", linestyle="--",
                                  label=f"{stage_label} (dashed)")
    ax.legend(fontsize=8, ncol=2)

    ax.set_title(f"Pergeseran MF  –  {var_name.upper()}  "
                 f"(Manual vs {stage_label})", fontsize=11, fontweight="bold")
    ax.set_xlabel(var_name, fontsize=9)
    ax.set_ylabel("Derajat Keanggotaan", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 4. Plot konvergensi GA (untuk Ablation Study)
# ------------------------------------------------------------
def plot_ga_convergence(history: dict,
                        title: str = "Konvergensi GA") -> plt.Figure:
    """
    Parameters
    ----------
    history : dict { label_eksperimen: list[float] }
              list = best_fitness per generasi
              contoh: {"pop=10": [...], "pop=50": [...]}
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors

    for i, (label, fitness_history) in enumerate(history.items()):
        gens = list(range(1, len(fitness_history) + 1))
        ax.plot(gens, fitness_history,
                color=colors[i % 10], linewidth=2, label=label)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Generasi", fontsize=10)
    ax.set_ylabel("Best Fitness (Accuracy)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Quick test / preview
# ------------------------------------------------------------
if __name__ == "__main__":
    fig = plot_all_mf(MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL,
                      suptitle="Tahap 1 – Manual FIS")
    fig.savefig("preview_mf_manual.png", dpi=120, bbox_inches="tight")
    print("Saved: preview_mf_manual.png")
