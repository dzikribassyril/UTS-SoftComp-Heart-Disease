# ========================================================================================================================
#  app.py  –  Aplikasi Streamlit Utama untuk Prediksi Risiko Penyakit Jantung serta visualisasi MF dan evaluasi model
#  Heart Disease Risk Prediction | Soft Computing UTS 2025/2026
#  Anggota:
#    1. 140810230008 – Robby Azwan Saputra
#    2. 140810230071 – Dzikri Basyril Mu'Minin 
#    3. 140810230074 – Farhan Zia Rizky
# ========================================================================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import (
    MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL,
    DECISION_THRESHOLD, FEATURE_RANGES, DATASET_PATH, FEATURES
)
from utils.preprocessing import load_and_prepare
from models.fis import predict_one, evaluate_stage1
from models.ga  import (
    run_ga, fis_predict_vectorized,
    summarize_mf_shift as ga_shift, run_ablation_study
)
from models.ann import (
    run_ann_tuning,
    summarize_mf_shift as ann_shift,
    plot_sensitivity_curves, plot_loss_curve
)
from utils.plot_mf import plot_all_mf, plot_mf_comparison, plot_ga_convergence
from utils.evaluate import compute_metrics


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title = "Heart Disease FIS | Soft Computing UTS",
    page_icon  = "heart",
    layout     = "wide",
)


# ============================================================
# Cached Data & Model Loading
# ============================================================

@st.cache_data(show_spinner="Memuat dataset...")
def get_data():
    return load_and_prepare(DATASET_PATH)


@st.cache_data(show_spinner="Mengevaluasi FIS Manual (Tahap 1)...")
def get_stage1(_X_test, _y_test):
    return evaluate_stage1(_X_test, _y_test)


@st.cache_data(show_spinner="Menjalankan GA Tuning (Tahap 2) — harap tunggu...")
def get_stage2(_X_train, _y_train):
    return run_ga(_X_train, _y_train, verbose=False)


@st.cache_data(show_spinner="Menjalankan ANN Tuning (Tahap 3) — harap tunggu...")
def get_stage3(_X_train, _y_train):
    return run_ann_tuning(_X_train, _y_train, verbose=False)


@st.cache_data(show_spinner="Menjalankan Ablation Study GA...")
def get_ablation(_X_train, _y_train):
    return run_ablation_study(_X_train, _y_train, verbose=False)


# ============================================================
# Load selruh data & model
# ============================================================

X_train, X_test, y_train, y_test, df_clean = get_data()

result_s1 = get_stage1(X_test, y_test)
result_s2 = get_stage2(X_train, y_train)
result_s3 = get_stage3(X_train, y_train)

ga_mf  = result_s2["best_mf_params"]
ann_mf = result_s3["best_mf_params"]

# Hitung metrik test set untuk semua tahap
X_test_arr = X_test[FEATURES].values.astype(np.float32)

_, preds_s1 = fis_predict_vectorized(X_test_arr, MF_PARAMS_MANUAL)
_, preds_s2 = fis_predict_vectorized(X_test_arr, ga_mf)
_, preds_s3 = fis_predict_vectorized(X_test_arr, ann_mf)

m1 = compute_metrics(y_test, preds_s1, "FIS Manual")
m2 = compute_metrics(y_test, preds_s2, "FIS + GA")
m3 = compute_metrics(y_test, preds_s3, "FIS + ANN")


# ============================================================
# Header
# ============================================================

st.title("Heart Disease Risk Prediction")
st.divider()

# ============================================================
# Sidebar Navigation
# ============================================================

st.sidebar.header("Navigasi")
pages = [
    "Prediksi Pasien",
    "Visualisasi MF",
    "Performa Model",
    "Ablation Study",
]

if "menu" not in st.session_state:
    st.session_state.menu = pages[0]

for page in pages:
    btn_type = "primary" if st.session_state.menu == page else "secondary"
    if st.sidebar.button(page, use_container_width=True, type=btn_type):
        st.session_state.menu = page

menu = st.session_state.menu

page_bg_map = {
    "Prediksi Pasien": "#eef7ff",
    "Visualisasi MF": "#f4f9f4",
    "Performa Model": "#fff8ed",
    "Ablation Study": "#f7f4ff",
}

st.markdown(
    f"""
    <style>
    .stApp [data-testid="block-container"] {{
        background-color: {page_bg_map.get(menu, "#ffffff")};
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# TAB 1 – Prediksi Pasien
# ------------------------------------------------------------
if menu == "Prediksi Pasien":
    st.header("Prediksi Risiko Penyakit Jantung")
    st.markdown(
        "Masukkan data pasien dan bandingkan prediksi "
        "dari ketiga sistem secara langsung."
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input Data Pasien")
        age    = st.slider("Usia (tahun)",
                           min_value=int(FEATURE_RANGES["age"][0]),
                           max_value=int(FEATURE_RANGES["age"][1]),
                           value=55)
        chol   = st.slider("Kolesterol (mg/dL)",
                           min_value=int(FEATURE_RANGES["chol"][0]),
                           max_value=int(FEATURE_RANGES["chol"][1]),
                           value=240)
        thalch = st.slider("Max Heart Rate",
                           min_value=int(FEATURE_RANGES["thalch"][0]),
                           max_value=int(FEATURE_RANGES["thalch"][1]),
                           value=140)

        btn = st.button("Prediksi", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Prediksi")

        if btn:
            sample     = {"age": age, "chol": chol, "thalch": thalch}
            sample_arr = np.array([[age, chol, thalch]], dtype=np.float32)

            # Prediksi dari ketiga sistem
            sc1, lb1 = predict_one(sample)
            _, lb_s2 = fis_predict_vectorized(sample_arr, ga_mf)
            sc2 = float(fis_predict_vectorized(sample_arr, ga_mf)[0][0])
            sc3 = float(fis_predict_vectorized(sample_arr, ann_mf)[0][0])
            lb2 = int(lb_s2[0])
            lb3 = int(fis_predict_vectorized(sample_arr, ann_mf)[1][0])

            def badge(label):
                return "**Disease**" if label == 1 else "**No Disease**"

            systems = [
                ("FIS Manual",  sc1,  lb1),
                ("FIS + GA",    sc2,  lb2),
                ("FIS + ANN",   sc3,  lb3),
            ]

            def interpret_risk(score):
                if score >= 0.7:
                    return "Risiko Tinggi – Disarankan pemeriksaan lanjutan"
                elif score >= 0.4:
                    return "Risiko Sedang – Perlu monitoring & gaya hidup sehat"
                else:
                    return "Risiko Rendah – Kondisi relatif aman"

            for name, score, label in systems:
                st.markdown(f"**{name}**")
                c1, c2 = st.columns(2)
                c1.metric("Risk Score", f"{score:.4f}")
                c2.markdown(f"<br>{badge(label)}", unsafe_allow_html=True)
                st.progress(float(np.clip(score, 0, 1)))
                st.caption(interpret_risk(score))
                st.divider()

        else:
            st.info("Isi data pasien dan klik Prediksi")


# ------------------------------------------------------------
# TAB 2 – Visualisasi MF
# ------------------------------------------------------------
elif menu == "Visualisasi MF":
    st.header("Visualisasi Membership Functions")
    view = st.radio("Pilih tampilan:",
                    ["Semua Tahap", "Per Tahap", "Sensitivity Curves ANN"],
                    horizontal=True)

    if view == "Per Tahap":
        sub = st.tabs(["Tahap 1 – Manual", "Tahap 2 – GA", "Tahap 3 – ANN"])

        with sub[0]:
            fig = plot_all_mf(MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL,
                              "Tahap 1 – Manual FIS")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with sub[1]:
            fig = plot_all_mf(ga_mf, OUTPUT_MF_MANUAL,
                              "Tahap 2 – FIS + GA")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.markdown("**Tabel pergeseran MF (Manual → GA):**")
            st.dataframe(ga_shift(MF_PARAMS_MANUAL, ga_mf),
                         use_container_width=True, hide_index=True)

        with sub[2]:
            fig = plot_all_mf(ann_mf, OUTPUT_MF_MANUAL,
                              "Tahap 3 – FIS + ANN")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.markdown("**Tabel pergeseran MF (Manual → ANN):**")
            st.dataframe(ann_shift(MF_PARAMS_MANUAL, ann_mf),
                         use_container_width=True, hide_index=True)

    elif view == "Semua Tahap":
        st.subheader("Pergeseran Kurva MF – Analisis Komparatif")
        selected_var = st.selectbox("Pilih variabel:", FEATURES)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Manual → GA**")
            fig_ga = plot_mf_comparison(
                selected_var,
                MF_PARAMS_MANUAL[selected_var],
                ga_mf[selected_var],
                stage_label="GA"
            )
            st.pyplot(fig_ga, use_container_width=True); plt.close(fig_ga)

        with col_b:
            st.markdown("**Manual → ANN**")
            fig_ann = plot_mf_comparison(
                selected_var,
                MF_PARAMS_MANUAL[selected_var],
                ann_mf[selected_var],
                stage_label="ANN"
            )
            st.pyplot(fig_ann, use_container_width=True); plt.close(fig_ann)

        # Tabel pergeseran gabungan
        st.divider()
        st.subheader("Tabel Pergeseran |Δb| (titik puncak MF)")
        df_ga_shift  = ga_shift(MF_PARAMS_MANUAL, ga_mf).rename(
            columns={"|Δb|": "|Δb| GA"})
        df_ann_shift = ann_shift(MF_PARAMS_MANUAL, ann_mf)[
            ["variable", "label", "|Δb|"]].rename(
            columns={"|Δb|": "|Δb| ANN"})
        df_combined  = df_ga_shift.merge(df_ann_shift, on=["variable", "label"])
        st.dataframe(df_combined, use_container_width=True, hide_index=True)

    else:  # Sensitivity Curves ANN
        st.subheader("ANN Sensitivity Analysis")
        st.markdown(
            "Kurva menunjukkan bagaimana prediksi ANN berubah "
            "saat satu fitur divariasikan (fitur lain = median). "
            "Titik perpotongan dengan **t1=0.35** dan **t2=0.65** "
            "digunakan sebagai batas transisi MF baru."
        )
        fig_sens = plot_sensitivity_curves(result_s3["sensitivity"])
        st.pyplot(fig_sens, use_container_width=True); plt.close(fig_sens)

        if result_s3["loss_history"]:
            st.subheader("ANN Training Loss Curve")
            fig_loss = plot_loss_curve(result_s3["loss_history"])
            st.pyplot(fig_loss, use_container_width=True); plt.close(fig_loss)


# ------------------------------------------------------------
# TAB 3 – Performa Model
# ------------------------------------------------------------
elif menu == "Performa Model":
    st.header("Perbandingan Performa Ketiga Sistem")

    # ── Tabel ringkasan ─────────────────────────────────────
    df_cmp = pd.DataFrame([m1, m2, m3]).set_index("model")

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: #d4edda; font-weight:bold"
                if v else "" for v in is_max]

    st.dataframe(
        df_cmp.style.apply(highlight_max, axis=0,
                           subset=["accuracy","precision","recall","f1_score"]),
        use_container_width=True
    )

    st.divider()

    # ── Bar chart perbandingan ───────────────────────────────
    st.subheader("Visualisasi Perbandingan")
    metrics_list = ["accuracy", "precision", "recall", "f1_score"]
    fig_bar, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    models = [m1["model"], m2["model"], m3["model"]]

    for ax, met in zip(axes, metrics_list):
        vals = [m1[met], m2[met], m3[met]]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(met.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02, f"{val:.2%}",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

    st.divider()

    # ── Detail per sistem ────────────────────────────────────
    st.subheader("Detail – Threshold Sweep FIS Manual")
    sweep_df = result_s1["sweep_df"]
    fig_sw, ax_sw = plt.subplots(figsize=(8, 3))
    ax_sw.plot(sweep_df["threshold"], sweep_df["accuracy"],
               label="Accuracy", marker="o", ms=4)
    ax_sw.plot(sweep_df["threshold"], sweep_df["f1_score"],
               label="F1 Score", marker="s", ms=4)
    ax_sw.axvline(DECISION_THRESHOLD, color="red", linestyle="--",
                  label=f"Threshold={DECISION_THRESHOLD}")
    ax_sw.set_xlabel("Threshold"); ax_sw.set_ylabel("Score")
    ax_sw.set_title("Akurasi & F1 vs Threshold – FIS Manual")
    ax_sw.legend(); ax_sw.grid(True, alpha=0.3)
    ax_sw.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_sw, use_container_width=True); plt.close(fig_sw)


# ------------------------------------------------------------
# TAB 4 – Ablation Study
# ------------------------------------------------------------
elif menu == "Ablation Study":
    st.header("Ablation Study – Konvergensi GA")
    st.markdown(
        "Analisis pengaruh **Population Size** dan **Jumlah Generasi** "
        "terhadap konvergensi. Konvergensi prematur terjadi ketika "
        "kurva sudah flat sebelum generasi berakhir."
    )

    ablation = get_ablation(X_train, y_train)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Variasi Population Size (gen=50)")
        fig_pop = plot_ga_convergence(
            ablation["by_popsize"],
            title="Konvergensi GA – Variasi Population Size"
        )
        st.pyplot(fig_pop, use_container_width=True); plt.close(fig_pop)

        st.markdown("**Observasi:**")
        pop_results = {k: max(v) for k, v in ablation["by_popsize"].items()}
        df_pop = pd.DataFrame(
            [{"pop_size": k, "best_accuracy": round(v, 4)}
             for k, v in pop_results.items()]
        )
        st.dataframe(df_pop, use_container_width=True, hide_index=True)
        st.caption(
            "Pop size kecil (5–10) cenderung konvergensi prematur "
            "karena keragaman genetik rendah. Pop size besar menjelajahi "
            "ruang solusi lebih luas namun lebih lambat."
        )

    with col_r:
        st.subheader("Variasi Jumlah Generasi (pop=30)")
        fig_gen = plot_ga_convergence(
            ablation["by_ngen"],
            title="Konvergensi GA – Variasi N Generations"
        )
        st.pyplot(fig_gen, use_container_width=True); plt.close(fig_gen)

        st.markdown("**Observasi:**")
        gen_results = {k: max(v) for k, v in ablation["by_ngen"].items()}
        df_gen = pd.DataFrame(
            [{"n_generations": k, "best_accuracy": round(v, 4)}
             for k, v in gen_results.items()]
        )
        st.dataframe(df_gen, use_container_width=True, hide_index=True)
        st.caption(
            "Generasi lebih banyak memberi ruang eksplorasi lebih luas, "
            "namun ada titik diminishing returns di mana peningkatan "
            "akurasi melambat signifikan."
        )

    # ── Kurva GA utama (konvergensi best vs avg) ─────────────
    st.divider()
    st.subheader("Konvergensi GA – Best vs Average Fitness (Konfigurasi Default)")
    fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
    gens = list(range(1, len(result_s2["fitness_history"]) + 1))
    ax_conv.plot(gens, result_s2["fitness_history"],
                 color="#FF9800", linewidth=2, label="Best Fitness")
    ax_conv.plot(gens, result_s2["avg_history"],
                 color="#FF9800", linewidth=1.5, linestyle="--",
                 alpha=0.6, label="Avg Fitness")
    ax_conv.fill_between(gens,
                         result_s2["avg_history"],
                         result_s2["fitness_history"],
                         alpha=0.15, color="#FF9800")
    ax_conv.set_xlabel("Generasi"); ax_conv.set_ylabel("Fitness (Accuracy)")
    ax_conv.set_title("Konvergensi GA – Default Config (pop=50, gen=100)")
    ax_conv.legend(); ax_conv.grid(True, alpha=0.3)
    ax_conv.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_conv, use_container_width=True); plt.close(fig_conv)