# 🧬 Job Scheduling with Evolutionary Algorithms

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Áp dụng Thuật toán Di truyền (GA) và Tối ưu hóa Bầy đàn (PSO) vào Bài toán Lập lịch Công việc**  
*So sánh toàn diện với các phương pháp tối ưu truyền thống: Greedy, Backtracking, Dynamic Programming, Branch & Bound*

---

*Môn học: Thiết kế và Đánh giá Thuật toán*  

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Bài toán](#-bài-toán-job-shop-scheduling)
- [Các thuật toán](#-các-thuật-toán-được-cài-đặt)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Giới thiệu

Bài toán **Lập lịch Công việc trên nhiều máy** (Job Shop Scheduling Problem – JSP) là một trong những bài toán tối ưu tổ hợp kinh điển và có ý nghĩa thực tiễn cao trong sản xuất, logistics và hệ thống tính toán song song. Khi số công việc và máy tăng lên, không gian nghiệm tăng theo cấp số mũ — khiến các phương pháp chính xác nhanh chóng mất khả năng áp dụng.

Dự án này nghiên cứu và so sánh **hai nhóm phương pháp**:

| Nhóm | Thuật toán | Đặc điểm |
|------|-----------|-----------|
| **Metaheuristic** | PSO, GA | Nghiệm gần tối ưu, thời gian chấp nhận được |
| **Truyền thống** | Greedy, Backtracking, DP, B&B | Tối ưu chính xác (nhỏ) hoặc heuristic nhanh |

### Mục tiêu nghiên cứu

- ✅ Cài đặt và mô hình hóa JSP bằng các lớp dữ liệu rõ ràng
- ✅ Triển khai PSO (với SPV encoding) và GA (với OX crossover, tournament selection)
- ✅ Triển khai 4 thuật toán đối chứng: Greedy (LPT/SPT), Backtracking, DP Bitmask, Branch & Bound
- ✅ So sánh theo 3 khía cạnh: thời gian thực thi, chất lượng nghiệm (makespan), khả năng mở rộng
- ✅ Kiểm định thống kê Wilcoxon để xác nhận ý nghĩa của sự khác biệt

---

## 🔧 Bài toán: Job Shop Scheduling

### Phát biểu hình thức

Cho:
- **n** công việc: $J = \{J_1, J_2, \ldots, J_n\}$
- **m** máy: $M = \{M_1, M_2, \ldots, M_m\}$
- Mỗi công việc $J_i$ có một chuỗi thao tác $(O_{i,1}, O_{i,2}, \ldots, O_{i,m})$ phải thực hiện **theo thứ tự**
- Thao tác $O_{i,k}$ cần máy $\mu_{i,k}$ trong thời gian $p_{i,k}$

**Mục tiêu:** Tối thiểu hóa *Makespan* $C_{\max} = \max_{i} C_i$

**Ràng buộc:**
- Mỗi máy chỉ xử lý tối đa một thao tác tại một thời điểm
- Các thao tác của cùng một công việc phải thực hiện theo thứ tự định trước
- Thao tác không được gián đoạn một khi đã bắt đầu

### Độ phức tạp

Bài toán JSP là **NP-Hard** ngay cả với 2 máy và 3 công việc. Không gian nghiệm có kích thước $(n!)^m$, tăng siêu cấp số mũ theo quy mô bài toán.

| Quy mô | n × m | Số thao tác | Không gian nghiệm (xấp xỉ) |
|--------|-------|-------------|---------------------------|
| Nhỏ    | 6 × 3 | 18          | $\approx 10^{6}$          |
| Trung  | 20 × 5 | 100        | $\approx 10^{94}$         |
| Lớn    | 50 × 6 | 300        | $\approx 10^{320}$        |

---

## 🤖 Các thuật toán được cài đặt

### 1. Particle Swarm Optimization (PSO)

PSO mô phỏng hành vi tìm kiếm tập thể của đàn chim/cá. Mỗi *particle* đại diện cho một nghiệm ứng viên và cập nhật vị trí theo:

$$v_{i,d}^{(t+1)} = w \cdot v_{i,d}^{(t)} + c_1 r_1 \left(pBest_{i,d} - x_{i,d}^{(t)}\right) + c_2 r_2 \left(gBest_d - x_{i,d}^{(t)}\right)$$

$$x_{i,d}^{(t+1)} = x_{i,d}^{(t)} + v_{i,d}^{(t+1)}$$

**Kỹ thuật SPV (Smallest Position Value):** Chuyển đổi vector thực liên tục sang hoán vị rời rạc, cho phép PSO giải bài toán tổ hợp.

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `n_particles` | 50 | Kích thước đàn |
| `max_iter` | 200 | Số vòng lặp tối đa |
| `w` | 0.9 → 0.4 | Trọng số quán tính (giảm tuyến tính) |
| `c1` | 2.0 | Hệ số nhận thức |
| `c2` | 2.0 | Hệ số xã hội |
| `v_max` | 0.5 × (ub - lb) | Giới hạn vận tốc |

### 2. Genetic Algorithm (GA)

GA mô phỏng tiến hóa sinh học qua các cơ chế chọn lọc, lai ghép và đột biến.

**Mã hóa:** Hoán vị công việc (permutation encoding) — mỗi nhiễm sắc thể là một chuỗi thứ tự công việc.

| Thành phần | Phương pháp | Mô tả |
|-----------|-------------|-------|
| **Selection** | Tournament (k=3) | Chọn cá thể tốt nhất trong k ứng viên ngẫu nhiên |
| **Crossover** | Order Crossover (OX) | Bảo toàn thứ tự tương đối của gen |
| **Mutation** | Swap + Inversion | Hoán đổi hoặc đảo ngược đoạn nhiễm sắc thể |
| **Elitism** | Top-2 | Giữ lại 2 cá thể tốt nhất mỗi thế hệ |

| Tham số | Giá trị |
|---------|---------|
| `pop_size` | 100 |
| `generations` | 300 |
| `p_crossover` | 0.85 |
| `p_mutation` | 0.05 |
| `elite_size` | 2 |

### 3. Greedy (LPT / SPT)

- **LPT (Longest Processing Time First):** Ưu tiên công việc dài nhất, giảm rủi ro bottleneck
- **SPT (Shortest Processing Time First):** Ưu tiên công việc ngắn nhất, tối ưu trung bình hoàn thành
- Độ phức tạp: **O(n log n)**

### 4. Backtracking

Duyệt toàn bộ không gian nghiệm theo chiều sâu với cắt tỉa nhánh. Đảm bảo tối ưu nhưng chỉ khả thi với bài toán quy mô rất nhỏ.
- Độ phức tạp: **O(n! × m)**

### 5. Dynamic Programming (DP Bitmask)

Sử dụng bitmask để mã hóa trạng thái các công việc đã hoàn thành.
- Độ phức tạp: **O(2ⁿ × n × m)**
- Giới hạn thực tế: n ≤ 20 (do bộ nhớ)

### 6. Branch & Bound (B&B)

Kết hợp duyệt cây tìm kiếm với hàm ước lượng cận dưới (lower bound) để cắt tỉa nhánh không hứa hẹn.
- Độ phức tạp: **O(n!)** (worst case), thực tế tốt hơn nhiều

---

## 📊 Kết quả thực nghiệm

### Điểm gãy (Break-point Analysis)

| Thuật toán | Quy mô tối đa khả thi | Thời gian tối đa |
|-----------|----------------------|-----------------|
| Backtracking | ≤ 8 jobs | < 10s |
| DP Bitmask | ≤ 20 jobs | < 30s |
| Branch & Bound | ≤ 15 jobs | < 60s |
| Greedy | Không giới hạn | O(n log n) |
| **PSO** | **Không giới hạn** | **~3.2s (n=50)** |
| **GA** | **Không giới hạn** | **~6.9s (n=50)** |

### So sánh PSO vs GA (n=50 jobs, m=6 machines)

| Chỉ số | PSO | GA | Nhận xét |
|--------|-----|-----|---------|
| Thời gian thực thi | **3.229s** | 6.909s | PSO nhanh hơn ~2.14× |
| Makespan trung bình | 312.4 | 315.8 | Tương đương |
| Độ lệch chuẩn | 8.3 | 11.2 | PSO ổn định hơn |
| Kiểm định Wilcoxon (p-value) | 0.3381 | — | Không có sự khác biệt thống kê đáng kể |

### Kết luận chính

> **PSO** phù hợp khi ưu tiên tốc độ và bài toán quy mô lớn (> 30 công việc).  
> **GA** phù hợp khi cần sự đa dạng nghiệm và khả năng tùy biến toán tử cao.  
> **Greedy** là lựa chọn thực dụng khi cần kết quả tức thì với chất lượng chấp nhận được.  
> **B&B/DP** chỉ nên dùng khi cần nghiệm tối ưu tuyệt đối với quy mô nhỏ.

---

## 📁 Cấu trúc dự án

```
job-scheduling-evo/
├── README.md                    # Tài liệu dự án (file này)
├── requirements.txt             # Thư viện Python cần thiết
├── setup.py                     # Cấu hình package
├── .gitignore                   # Git ignore rules
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
│
├── src/                         # Mã nguồn chính
│   ├── __init__.py
│   ├── problem/                 # Mô hình hóa bài toán
│   │   ├── __init__.py
│   │   ├── job.py               # Task, Job, Machine, JSPInstance
│   │   ├── generator.py         # Sinh dữ liệu ngẫu nhiên / benchmark
│   │   └── encoder.py           # SPV encoding, giải mã lịch trình
│   │
│   ├── algorithms/              # Các thuật toán
│   │   ├── __init__.py
│   │   ├── base.py              # Lớp cơ sở cho tất cả thuật toán
│   │   ├── pso.py               # Particle Swarm Optimization
│   │   ├── ga.py                # Genetic Algorithm
│   │   ├── greedy.py            # Greedy (LPT / SPT)
│   │   ├── backtracking.py      # Backtracking + Pruning
│   │   ├── dp.py                # Dynamic Programming (Bitmask)
│   │   └── branch_bound.py      # Branch and Bound
│   │
│   ├── evaluation/              # Đánh giá và kiểm định
│   │   ├── __init__.py
│   │   ├── metrics.py           # Các chỉ số hiệu năng
│   │   ├── benchmark.py         # Khung chạy thực nghiệm
│   │   └── statistics.py        # Kiểm định thống kê Wilcoxon
│   │
│   └── visualization/           # Trực quan hóa
│       ├── __init__.py
│       ├── gantt.py             # Biểu đồ Gantt
│       └── convergence.py       # Đường cong hội tụ
│
├── notebooks/
│   └── main_experiment.ipynb    # Notebook thực nghiệm đầy đủ
│
├── experiments/
│   ├── config.py                # Cấu hình tham số thực nghiệm
│   └── run_experiments.py       # Script chạy toàn bộ thực nghiệm
│
├── results/                     # Kết quả thực nghiệm (được sinh ra)
│   └── .gitkeep
│
├── docs/
│   ├── theory/
│   │   ├── pso.md               # Lý thuyết PSO chi tiết
│   │   └── ga.md                # Lý thuyết GA chi tiết
│   └── api.md                   # Tài liệu API
│
└── tests/
    ├── __init__.py
    ├── test_problem.py          # Unit test cho mô hình bài toán
    ├── test_pso.py              # Unit test cho PSO
    ├── test_ga.py               # Unit test cho GA
    └── test_algorithms.py       # Integration tests
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống

- Python ≥ 3.10
- pip hoặc conda

### Cài đặt

```bash
# Clone repository
git clone https://github.com/<your-username>/job-scheduling-evo.git
cd job-scheduling-evo

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Cài đặt thư viện
pip install -r requirements.txt

# Cài đặt package (development mode)
pip install -e .
```

### Chạy thực nghiệm nhanh

```bash
# Chạy toàn bộ thực nghiệm so sánh
python experiments/run_experiments.py

# Chạy chỉ PSO trên bài toán nhỏ
python -c "
from src.problem.generator import generate_random_instance
from src.algorithms.pso import PSOScheduler

instance = generate_random_instance(n_jobs=10, n_machines=3, seed=42)
pso = PSOScheduler(instance, n_particles=30, max_iter=100)
result = pso.run()
print(f'Best Makespan: {result.best_makespan}')
print(f'Time: {result.elapsed_time:.3f}s')
"
```

### Chạy Jupyter Notebook

```bash
jupyter notebook notebooks/main_experiment.ipynb
```

### Chạy tests

```bash
pytest tests/ -v
```

---

## 📖 Hướng dẫn sử dụng

### Tạo bài toán

```python
from src.problem.generator import generate_random_instance, generate_benchmark_instance

# Sinh ngẫu nhiên
instance = generate_random_instance(n_jobs=20, n_machines=5, seed=42)

# Benchmark chuẩn (ft06, ft10, la01...)
instance = generate_benchmark_instance("ft06")
```

### Chạy PSO

```python
from src.algorithms.pso import PSOScheduler, PSOConfig

config = PSOConfig(
    n_particles=50,
    max_iter=200,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
)

scheduler = PSOScheduler(instance, config=config)
result = scheduler.run()

print(f"Makespan: {result.best_makespan}")
print(f"Thời gian: {result.elapsed_time:.3f}s")
```

### Chạy GA

```python
from src.algorithms.ga import GAScheduler, GAConfig

config = GAConfig(
    pop_size=100,
    generations=300,
    p_crossover=0.85,
    p_mutation=0.05,
    elite_size=2,
    tournament_k=3,
)

scheduler = GAScheduler(instance, config=config)
result = scheduler.run()
```

### So sánh tất cả thuật toán

```python
from src.evaluation.benchmark import AlgorithmBenchmark

bench = AlgorithmBenchmark(instance)
report = bench.run_all(n_runs=10)
report.print_summary()
report.save("results/comparison.csv")
```

### Trực quan hóa

```python
from src.visualization.gantt import plot_gantt_chart
from src.visualization.convergence import plot_convergence

# Biểu đồ Gantt
plot_gantt_chart(result.schedule, title="PSO - Lịch trình tối ưu")

# Đường cong hội tụ
plot_convergence(
    {"PSO": pso_result.history, "GA": ga_result.history},
    title="So sánh tốc độ hội tụ"
)
```

---

## 📚 Tài liệu tham khảo

1. **Kennedy, J., & Eberhart, R. (1995).** Particle swarm optimization. *Proceedings of ICNN'95*, 4, 1942–1948.

2. **Holland, J. H. (1975).** *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

3. **Pinedo, M. L. (2016).** *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.

4. **Garey, M. R., & Johnson, D. S. (1979).** *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.

5. **Pezzella, F., Morganti, G., & Ciaschetti, G. (2008).** A genetic algorithm for the Flexible Job-shop Scheduling Problem. *Computers & Operations Research*, 35(10), 3202–3212.

6. **Zhang, R., & Wu, C. (2010).** A hybrid immune simulated annealing algorithm for the job shop scheduling problem. *Applied Soft Computing*, 10(1), 79–89.

7. **Błażewicz, J., Domschke, W., & Pesch, E. (1996).** The job shop scheduling problem: Conventional and new solution techniques. *European Journal of Operational Research*, 93(1), 1–33.

---

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép [MIT License](LICENSE).

---

<div align="center">

*Được thực hiện trong khuôn khổ môn học **Thiết kế và Đánh giá Thuật toán***

</div>
