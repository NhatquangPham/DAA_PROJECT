# Tài liệu API

## Mô hình hóa bài toán (`src.problem`)

### `Task`
```python
Task(job_id: int, machine_id: int, processing_time: int)
```
Đại diện một thao tác (operation). Có `start_time` và `end_time` sau khi giải mã.

### `Job`
```python
Job(job_id: int, tasks: List[Task], priority: int = 1)
```
- `.total_processing_time` → tổng thời gian xử lý
- `.num_operations` → số thao tác

### `JSPInstance`
```python
JSPInstance(jobs: List[Job], num_machines: int, name: str = "unnamed")
```
- `.lower_bound` → cận dưới lý thuyết của makespan
- `.reset()` → xóa thông tin lịch trình

### `generate_random_instance`
```python
generate_random_instance(n_jobs, n_machines, min_pt=1, max_pt=20, seed=None) → JSPInstance
```

### `generate_benchmark_instance`
```python
generate_benchmark_instance(name: str) → JSPInstance
# name ∈ {"ft06", "ft10", "la01"}
```

---

## Thuật toán (`src.algorithms`)

Mọi scheduler đều kế thừa `BaseScheduler` và có phương thức `.run() → AlgorithmResult`.

### `AlgorithmResult`
```python
@dataclass
class AlgorithmResult:
    algorithm_name: str
    best_makespan: float
    best_sequence: List[int]
    elapsed_time: float
    memory_kb: float
    history: List[float]   # Lịch sử makespan tốt nhất theo vòng lặp
    is_optimal: bool
    n_iterations: int
```

### `PSOScheduler`
```python
PSOScheduler(instance: JSPInstance, config: PSOConfig = None)
```

**PSOConfig** tham số:
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `n_particles` | 50 | Kích thước đàn |
| `max_iter` | 200 | Số vòng lặp |
| `w_max` | 0.9 | Quán tính ban đầu |
| `w_min` | 0.4 | Quán tính cuối |
| `c1` | 2.0 | Hệ số nhận thức |
| `c2` | 2.0 | Hệ số xã hội |
| `seed` | 42 | Hạt giống |

### `GAScheduler`
```python
GAScheduler(instance: JSPInstance, config: GAConfig = None)
```

**GAConfig** tham số:
| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `pop_size` | 100 | Kích thước quần thể |
| `generations` | 300 | Số thế hệ |
| `p_crossover` | 0.85 | Xác suất lai ghép |
| `p_mutation` | 0.05 | Xác suất đột biến |
| `elite_size` | 2 | Số cá thể tinh hoa |
| `tournament_k` | 3 | Kích thước giải đấu |
| `seed` | 42 | Hạt giống |

### `GreedyScheduler`
```python
GreedyScheduler(instance: JSPInstance, strategy: Literal["LPT", "SPT"] = "LPT")
```

### `BacktrackingScheduler`
```python
BacktrackingScheduler(instance: JSPInstance, time_limit: float = 30.0)
```

### `DPScheduler`
```python
DPScheduler(instance: JSPInstance, time_limit: float = 60.0)
# Giới hạn: n_jobs ≤ 20
```

### `BranchBoundScheduler`
```python
BranchBoundScheduler(instance: JSPInstance, time_limit: float = 60.0)
# Giới hạn thực tế: n_jobs ≤ 12
```

---

## Đánh giá (`src.evaluation`)

### `AlgorithmBenchmark`
```python
bench = AlgorithmBenchmark(instance, include_exact=True, pso_config=..., ga_config=...)
report = bench.run_all(n_runs=10, verbose=True)
report.print_summary()
report.save("results/comparison.csv")
```

### `wilcoxon_test`
```python
test = wilcoxon_test(samples_a, samples_b, name_a, name_b, alpha=0.05)
print(test.conclusion)
```

---

## Trực quan hóa (`src.visualization`)

### `plot_gantt_chart`
```python
plot_gantt_chart(instance, sequence, title="...", figsize=(14,6), save_path="...")
```

### `plot_convergence`
```python
plot_convergence(histories={"PSO": [...], "GA": [...]}, title="...", save_path="...")
```

### `plot_comparison_boxplot`
```python
plot_comparison_boxplot(metrics_list, title="...", save_path="...")
```

### `plot_scalability`
```python
plot_scalability(scales=[6,20,50,100], time_data={"PSO": [...], "GA": [...]}, save_path="...")
```
