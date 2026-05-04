"""
Cấu hình tham số cho các thực nghiệm so sánh thuật toán.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ============================================================
#  Cấu hình PSO
# ============================================================
PSO_CONFIG = {
    "n_particles": 50,
    "max_iter": 200,
    "w_max": 0.9,
    "w_min": 0.4,
    "c1": 2.0,
    "c2": 2.0,
    "v_max_ratio": 0.5,
    "greedy_ratio": 0.2,
    "seed": 42,
}

# ============================================================
#  Cấu hình GA
# ============================================================
GA_CONFIG = {
    "pop_size": 100,
    "generations": 300,
    "p_crossover": 0.85,
    "p_mutation": 0.05,
    "elite_size": 2,
    "tournament_k": 3,
    "greedy_ratio": 0.2,
    "seed": 42,
}

# ============================================================
#  Quy mô bài toán thực nghiệm
# ============================================================
EXPERIMENT_SCALES = [
    {"n_jobs": 6,   "n_machines": 3,  "label": "Nhỏ (6×3)",      "include_exact": True},
    {"n_jobs": 10,  "n_machines": 4,  "label": "Nhỏ-TB (10×4)",  "include_exact": True},
    {"n_jobs": 20,  "n_machines": 5,  "label": "Trung bình (20×5)", "include_exact": False},
    {"n_jobs": 50,  "n_machines": 6,  "label": "Lớn (50×6)",     "include_exact": False},
    {"n_jobs": 100, "n_machines": 8,  "label": "Rất lớn (100×8)", "include_exact": False},
]

# ============================================================
#  Cấu hình thực nghiệm
# ============================================================
N_RUNS = 10          # Số lần chạy cho mỗi thuật toán (metaheuristic)
RANDOM_SEED = 42     # Hạt giống ngẫu nhiên cho sinh dữ liệu
RESULTS_DIR = "results"

# ============================================================
#  Benchmark chuẩn để kiểm tra tính đúng đắn
# ============================================================
BENCHMARK_INSTANCES = ["ft06"]  # ft10, la01 cần thêm thời gian
