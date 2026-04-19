# ============================================================
#  config.py  –  konfigurasi / rule untuk FIS, GA, ANN
#  Heart Disease Risk Prediction | Soft Computing UTS 2025/2026
#  Anggota:
#    1. 140810230008 – Robby Azwan Saputra
#    2. 140810230071 – Dzikri Basyril Mu'Minin 
#    3. 140810230074 – Farhan Zia Rizky
# ============================================================

# -----------------------------------------------------------
# Datasetnya
# -----------------------------------------------------------
DATASET_PATH = "heart_disease_uci.csv"

# Features digunakan sebagai input FIS (3 fitur numerik utama)
FEATURES       = ["age", "chol", "thalch"] # input untuk FIS yaitu umur, kolesterol, dan max heart rate
TARGET_COL     = "num"          # 0 = no disease, 1-4 = disease
BINARY_TARGET  = True           # True → binarize num: 0 vs >0
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

# -----------------------------------------------------------
# Feature ranges (dari eksplorasi data di dataset)
# -----------------------------------------------------------
FEATURE_RANGES = {
    "age"   : (28,  77),
    "chol"  : (100, 603),
    "thalch": (60,  202),
}

# -----------------------------------------------------------
# Stage 1 – Manual MF parameters  (trimf: [a, b, c])
# Ditentukan berdasarkan intuisi manusia / kelompok kami + internet (pakar)
# -----------------------------------------------------------
MF_PARAMS_MANUAL = {
    "age": {
        "young" : [28,  28,  45],   # < 45 tahun/muda
        "middle": [38,  52,  65],   # 38–65 tahun/tengah
        "old"   : [58,  77,  77],   # > 58 tahun/tua
    },
    "chol": {
        "normal"    : [100, 100, 200],  # < 200 mg/dL  → normal
        "borderline": [175, 230, 280],  # 200–239 mg/dL → batas/borderline
        "high"      : [250, 603, 603],  # ≥ 240 mg/dL   → tinggi
    },
    "thalch": {
        "low"   : [60,  60,  120],  # max heart rate rendah → buruk
        "medium": [100, 150, 170], # max heart rate sedang → sedang
        "high"  : [150, 202, 202],  # max heart rate tinggi → lebih sehat
    },
}

# Output MF (risk score 0.0 – 1.0)
OUTPUT_MF_MANUAL = {
    "low"   : [0.0, 0.0, 0.4],
    "medium": [0.3, 0.5, 0.7],
    "high"  : [0.6, 1.0, 1.0],
}

# -----------------------------------------------------------
# Rules  (kondisi → output)
# Format: (dict{variabel: kategori_MF}, output_label)
# AND semantics → firing strength = min(membership values)
# -----------------------------------------------------------
RULES_MANUAL = [
    # High risk rules
    ({"age": "old",    "chol": "high"},        "high"),
    ({"age": "old",    "thalch": "low"},       "high"),
    ({"chol": "high",  "thalch": "low"},       "high"),
    ({"age": "middle", "chol": "high"},        "high"),

    # Medium risk rules
    ({"age": "old",    "chol": "borderline"},  "medium"),
    ({"age": "middle", "chol": "borderline"},  "medium"),
    ({"age": "old",    "thalch": "medium"},    "medium"),

    # Low risk rules
    ({"age": "young",  "chol": "normal"},      "low"),
    ({"age": "young",  "chol": "borderline"},  "low"),
    ({"age": "middle", "chol": "normal"},      "low"),
    ({"age": "young",  "thalch": "high"},      "low"),
]

# Threshold defuzzifikasi → klasifikasi biner
DECISION_THRESHOLD = 0.45   # risk_score >= threshold → disease

# -----------------------------------------------------------
# Stage 2 – GA Hyperparameters (default)
# -----------------------------------------------------------
GA_CONFIG = {
    "population_size": 50,
    "n_generations"  : 100,
    "crossover_rate" : 0.8,
    "mutation_rate"  : 0.1,
    "elite_size"     : 2,
}

# -----------------------------------------------------------
# Stage 3 – ANN Hyperparameters (default)
# -----------------------------------------------------------
ANN_CONFIG = {
    "hidden_layers": (64, 32),
    "activation"   : "relu",
    "max_iter"     : 500,
    "learning_rate": 0.001,
}

# -----------------------------------------------------------
# Ablation study settings (untuk laporan)
# -----------------------------------------------------------
ABLATION_POP_SIZES   = [5, 10, 30, 50, 100]
ABLATION_GENERATIONS = [10, 30, 50, 100, 200]