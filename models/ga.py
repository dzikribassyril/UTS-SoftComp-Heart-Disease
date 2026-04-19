# ============================================================
#  ga.py  –  Tahap 2: GA Tuning MF
# ============================================================
#  Anggota:
#    1. 140810230008 – Robby Azwan Saputra
#    2. 140810230071 – Dzikri Basyril Mu'Minin 
#    3. 140810230074 – Farhan Zia Rizky
#
#  Yang dioptimasi : titik-titik [a, b, c] setiap trimf INPUT
#  Rules           : TIDAK diubah (tetap dari Tahap 1)
#  Fitness         : akurasi klasifikasi pada training set
#
#  Kromosom (27 gen):
#    age   : young[a,b,c]  middle[a,b,c]  old[a,b,c]      (9)
#    chol  : normal[a,b,c] border[a,b,c]  high[a,b,c]     (9)
#    thalch: low[a,b,c]    medium[a,b,c]  high[a,b,c]     (9)
# ============================================================

import numpy as np
import pandas as pd
import time

from utils.config import (
    MF_PARAMS_MANUAL, OUTPUT_MF_MANUAL, RULES_MANUAL,
    DECISION_THRESHOLD, FEATURE_RANGES,
    GA_CONFIG, ABLATION_POP_SIZES, ABLATION_GENERATIONS,
    FEATURES
)
from utils.evaluate import compute_metrics, print_report


# ============================================================
# BAGIAN 1 – Vectorized FIS (jauh lebih cepat dari Stage 1)
# Digunakan khusus untuk fitness evaluation dalam loop GA
# ============================================================

def _trimf_vec(x_vals: np.ndarray, abc: list) -> np.ndarray:
    """Triangular MF vectorized untuk array (N,)."""
    a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
    y = np.zeros(len(x_vals), dtype=np.float32)
    if b > a:
        m = (x_vals > a) & (x_vals <= b)
        y[m] = (x_vals[m] - a) / (b - a)
    y[x_vals == b] = 1.0
    if c > b:
        m = (x_vals > b) & (x_vals < c)
        y[m] = (c - x_vals[m]) / (c - b)
    return y


def fis_predict_vectorized(X_arr: np.ndarray,
                           mf_params: dict,
                           output_mf: dict = None,
                           rules: list     = None,
                           threshold: float = None,
                           resolution: int = 80) -> tuple:
    """
    Vectorized Mamdani FIS: proses seluruh dataset sekaligus.

    Parameters
    ----------
    X_arr      : np.ndarray shape (N, 3) — kolom [age, chol, thalch]
    mf_params  : dict MF parameter input
    output_mf  : dict MF output
    rules      : list aturan fuzzy
    threshold  : threshold defuzzifikasi
    resolution : jumlah titik domain output

    Returns
    -------
    scores : np.ndarray (N,)  – risk score 0–1
    labels : np.ndarray (N,)  – prediksi biner 0/1
    """
    if output_mf is None:  output_mf = OUTPUT_MF_MANUAL
    if rules     is None:  rules     = RULES_MANUAL
    if threshold is None:  threshold = DECISION_THRESHOLD

    N        = X_arr.shape[0]
    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    # ── 1. Fuzzifikasi (batch) ─────────────────────────────
    fuzz = {}
    for var, labels in mf_params.items():
        col = X_arr[:, feat_idx[var]]
        fuzz[var] = {lbl: _trimf_vec(col, abc)
                     for lbl, abc in labels.items()}

    # ── 2. Rule firing: AND = min ──────────────────────────
    fired = []
    for antecedents, consequent in rules:
        parts    = [fuzz[v][l] for v, l in antecedents.items()]
        strength = np.min(np.stack(parts, axis=0), axis=0)   # (N,)
        fired.append((strength, consequent))

    # ── 3. Defuzzifikasi vectorized (centroid) ─────────────
    x_out = np.linspace(0.0, 1.0, resolution, dtype=np.float32)  # (R,)

    out_mf_arr = {lbl: _trimf_vec(x_out, abc)
                  for lbl, abc in output_mf.items()}

    y_agg = np.zeros((N, resolution), dtype=np.float32)
    for strength, consequent in fired:
        y_mf    = out_mf_arr[consequent]                          # (R,)
        clipped = np.minimum(strength[:, None], y_mf[None, :])   # (N,R)
        y_agg   = np.maximum(y_agg, clipped)

    denom  = y_agg.sum(axis=1)
    numer  = (x_out[None, :] * y_agg).sum(axis=1)
    scores = np.where(denom > 0, numer / denom, 0.0).astype(np.float32)
    labels = (scores >= threshold).astype(np.int8)

    return scores, labels


# ============================================================
# BAGIAN 2 – Encoding / Decoding Kromosom
# ============================================================

def encode_mf_params(mf_params: dict = None) -> np.ndarray:
    """Dict MF params → vektor flat (kromosom)."""
    if mf_params is None:
        mf_params = MF_PARAMS_MANUAL
    genes = []
    for var in FEATURES:
        for lbl, abc in mf_params[var].items():
            genes.extend([float(abc[0]), float(abc[1]), float(abc[2])])
    return np.array(genes, dtype=np.float64)


def decode_mf_params(chromosome: np.ndarray,
                     template: dict = None) -> dict:
    """
    Vektor flat → dict MF params.
    Constraint: a ≤ b ≤ c, nilai dalam FEATURE_RANGES.
    """
    if template is None:
        template = MF_PARAMS_MANUAL
    new_params = {}
    idx = 0
    for var in FEATURES:
        lo, hi = FEATURE_RANGES[var]
        new_params[var] = {}
        for lbl in template[var]:
            vals = [np.clip(chromosome[idx + k], lo, hi) for k in range(3)]
            a, b, c = sorted(vals)
            # Hindari degenerate (semua sama)
            span = (hi - lo)
            if c - a < span * 0.01:
                c = min(a + span * 0.01, hi)
            new_params[var][lbl] = [float(a), float(b), float(c)]
            idx += 3
    return new_params


def get_bounds(template: dict = None) -> tuple:
    """Kembalikan (lower_bound, upper_bound) vektor tiap gen."""
    if template is None:
        template = MF_PARAMS_MANUAL
    lb, ub = [], []
    for var in FEATURES:
        lo, hi = FEATURE_RANGES[var]
        n_genes = len(template[var]) * 3
        lb.extend([lo] * n_genes)
        ub.extend([hi] * n_genes)
    return np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64)


# ============================================================
# BAGIAN 3 – Fitness Function
# ============================================================

def fitness(chromosome: np.ndarray,
            X_arr: np.ndarray,
            y_arr: np.ndarray) -> float:
    """Fitness = akurasi pada training set (vectorized FIS)."""
    mf = decode_mf_params(chromosome)
    _, preds = fis_predict_vectorized(X_arr, mf)
    return float((preds == y_arr).mean())


# ============================================================
# BAGIAN 4 – GA Operators
# ============================================================

def initialize_population(pop_size: int,
                           lb: np.ndarray,
                           ub: np.ndarray,
                           seed: int = 42) -> np.ndarray:
    """
    Inisialisasi populasi.
    Individu ke-0 = MF manual (warm start).
    Sisanya = perturbasi ±20% dari MF manual.
    """
    np.random.seed(seed)
    n_genes = len(lb)
    base    = encode_mf_params()
    pop     = np.empty((pop_size, n_genes))

    pop[0]  = base.copy()
    spread  = (ub - lb) * 0.20
    for i in range(1, pop_size):
        noise  = np.random.uniform(-spread, spread)
        pop[i] = np.clip(base + noise, lb, ub)

    return pop


def selection_tournament(population: np.ndarray,
                          fitness_vals: np.ndarray,
                          k: int = 3) -> np.ndarray:
    """Tournament selection dengan ukuran k."""
    idx  = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax(fitness_vals[idx])]
    return population[best].copy()


def crossover_sbx(p1: np.ndarray,
                  p2: np.ndarray,
                  lb: np.ndarray,
                  ub: np.ndarray,
                  eta: float = 20.0) -> tuple:
    """
    Simulated Binary Crossover (SBX).
    eta besar → offspring dekat parent (eksploitasi).
    eta kecil → offspring menyebar (eksplorasi).
    """
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = np.random.rand()
        beta = ((2.0 * u) ** (1.0 / (eta + 1.0))
                if u <= 0.5
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))

        c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
        c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])

    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def mutation_polynomial(chromosome: np.ndarray,
                         lb: np.ndarray,
                         ub: np.ndarray,
                         mutation_rate: float = 0.1,
                         eta: float = 20.0) -> np.ndarray:
    """
    Polynomial mutation.
    Tiap gen dimutasi dengan probabilitas mutation_rate.
    """
    chrom = chromosome.copy()
    for i in range(len(chrom)):
        if np.random.rand() >= mutation_rate:
            continue
        u = np.random.rand()
        delta = ((2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                 if u < 0.5
                 else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))
        chrom[i] += delta * (ub[i] - lb[i])
        chrom[i]  = np.clip(chrom[i], lb[i], ub[i])
    return chrom


# ============================================================
# BAGIAN 5 – Main GA Loop
# ============================================================

def run_ga(X_train: pd.DataFrame,
           y_train: pd.Series,
           pop_size: int        = None,
           n_generations: int   = None,
           crossover_rate: float = None,
           mutation_rate: float  = None,
           elite_size: int       = None,
           verbose: bool         = True,
           seed: int             = 42) -> dict:
    """
    Jalankan GA untuk mengoptimasi parameter MF.

    Returns
    -------
    dict {
        "best_mf_params"  : dict MF terbaik,
        "best_fitness"    : float akurasi train terbaik,
        "fitness_history" : list[float] best fitness per generasi,
        "avg_history"     : list[float] rata-rata fitness per gen,
        "best_chromosome" : np.ndarray,
        "runtime_sec"     : float,
    }
    """
    cfg            = GA_CONFIG.copy()
    pop_size       = pop_size       or cfg["population_size"]
    n_generations  = n_generations  or cfg["n_generations"]
    crossover_rate = crossover_rate or cfg["crossover_rate"]
    mutation_rate  = mutation_rate  or cfg["mutation_rate"]
    elite_size     = elite_size     or cfg["elite_size"]

    np.random.seed(seed)

    X_arr = X_train[FEATURES].values.astype(np.float32)
    y_arr = y_train.values.astype(np.int8)

    lb, ub     = get_bounds()
    population = initialize_population(pop_size, lb, ub, seed=seed)

    fitness_history   = []
    avg_history       = []
    best_chromosome   = population[0].copy()
    best_fitness_ever = -np.inf

    if verbose:
        print(f"\n[GA] pop={pop_size} | gen={n_generations} | "
              f"pc={crossover_rate} | pm={mutation_rate}")
        print(f"{'Gen':>5}  {'Best':>8}  {'Avg':>8}  {'Time':>7}")
        print("-" * 38)

    t_start = time.time()

    for gen in range(n_generations):
        t_gen = time.time()

        # ── Evaluasi fitness ─────────────────────────────────
        fitness_vals = np.array([fitness(ind, X_arr, y_arr)
                                 for ind in population])

        best_idx = int(np.argmax(fitness_vals))
        best_fit = float(fitness_vals[best_idx])
        avg_fit  = float(fitness_vals.mean())

        fitness_history.append(best_fit)
        avg_history.append(avg_fit)

        if best_fit > best_fitness_ever:
            best_fitness_ever = best_fit
            best_chromosome   = population[best_idx].copy()

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"{gen+1:>5}  {best_fit:>8.4f}  {avg_fit:>8.4f}  "
                  f"{time.time()-t_gen:>6.2f}s")

        # ── Elitism ──────────────────────────────────────────
        sorted_idx = np.argsort(fitness_vals)[::-1]
        new_pop    = [population[i].copy()
                      for i in sorted_idx[:elite_size]]

        # ── Crossover + Mutation ─────────────────────────────
        while len(new_pop) < pop_size:
            p1 = selection_tournament(population, fitness_vals)
            p2 = selection_tournament(population, fitness_vals)

            if np.random.rand() < crossover_rate:
                c1, c2 = crossover_sbx(p1, p2, lb, ub)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutation_polynomial(c1, lb, ub, mutation_rate)
            c2 = mutation_polynomial(c2, lb, ub, mutation_rate)
            new_pop.extend([c1, c2])

        population = np.array(new_pop[:pop_size])

    runtime = time.time() - t_start

    best_mf = decode_mf_params(best_chromosome)

    if verbose:
        print(f"\n[GA] Selesai dalam {runtime:.1f}s | "
              f"Best train acc: {best_fitness_ever:.4f}")

    return {
        "best_mf_params"  : best_mf,
        "best_fitness"    : best_fitness_ever,
        "fitness_history" : fitness_history,
        "avg_history"     : avg_history,
        "best_chromosome" : best_chromosome,
        "runtime_sec"     : runtime,
    }


# ============================================================
# BAGIAN 6 – Ablation Study
# ============================================================

def run_ablation_study(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       verbose: bool = True) -> dict:
    """
    Variasikan pop_size (gen=50) dan n_gen (pop=30).
    Untuk analisis konvergensi prematur (laporan).

    Returns
    -------
    dict { "by_popsize": {...}, "by_ngen": {...} }
    """
    results_pop  = {}
    results_ngen = {}

    print("\n[Ablation] Variasi Population Size  (n_gen=50 fixed)")
    print("=" * 50)
    for ps in ABLATION_POP_SIZES:
        lbl = f"pop={ps}"
        r   = run_ga(X_train, y_train,
                     pop_size=ps, n_generations=50,
                     verbose=False, seed=0)
        results_pop[lbl] = r["fitness_history"]
        print(f"  {lbl:>10}: best={r['best_fitness']:.4f}  "
              f"t={r['runtime_sec']:.1f}s")

    print("\n[Ablation] Variasi N Generations  (pop=30 fixed)")
    print("=" * 50)
    for ng in ABLATION_GENERATIONS:
        lbl = f"gen={ng}"
        r   = run_ga(X_train, y_train,
                     pop_size=30, n_generations=ng,
                     verbose=False, seed=0)
        results_ngen[lbl] = r["fitness_history"]
        print(f"  {lbl:>8}: best={r['best_fitness']:.4f}  "
              f"t={r['runtime_sec']:.1f}s")

    return {"by_popsize": results_pop, "by_ngen": results_ngen}


# ============================================================
# BAGIAN 7 – Utilitas Analisis
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
            aa = mf_after[var][lbl]
            rows.append({
                "variable" : var,
                "label"    : lbl,
                "a_before" : round(ab[0], 2), "b_before": round(ab[1], 2),
                "c_before" : round(ab[2], 2),
                "a_after"  : round(aa[0], 2), "b_after" : round(aa[1], 2),
                "c_after"  : round(aa[2], 2),
                "|Δb|"     : round(abs(aa[1] - ab[1]), 2),
            })
    return pd.DataFrame(rows)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    from utils.preprocessing import load_and_prepare

    X_train, X_test, y_train, y_test, _ = load_and_prepare()

    print("=" * 50)
    print("  TAHAP 2 – GA TUNING")
    print("=" * 50)

    result = run_ga(X_train, y_train, verbose=True)

    # Evaluasi test set
    X_test_arr = X_test[FEATURES].values.astype(np.float32)
    _, preds   = fis_predict_vectorized(X_test_arr, result["best_mf_params"])
    metrics    = compute_metrics(y_test, preds, "FIS + GA")
    print_report(y_test, preds, "FIS + GA")

    # Pergeseran MF
    shift_df = summarize_mf_shift(MF_PARAMS_MANUAL, result["best_mf_params"])
    print("\nPergeseran titik puncak MF (|Δb|):")
    print(shift_df.to_string(index=False))

    # Ablation study
    print("\n" + "=" * 50)
    print("  ABLATION STUDY")
    print("=" * 50)
    ablation = run_ablation_study(X_train, y_train)
