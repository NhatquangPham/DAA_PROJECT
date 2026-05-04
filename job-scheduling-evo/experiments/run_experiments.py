"""
Script chạy toàn bộ thực nghiệm so sánh thuật toán.

Sử dụng:
    python experiments/run_experiments.py

Kết quả được lưu vào thư mục results/.
"""
import os
import sys
import time

# Thêm root vào Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Không hiển thị cửa sổ, chỉ lưu file

from src.problem.generator import generate_random_instance, generate_benchmark_instance
from src.algorithms.pso import PSOScheduler, PSOConfig
from src.algorithms.ga import GAScheduler, GAConfig
from src.algorithms.greedy import GreedyScheduler
from src.evaluation.benchmark import AlgorithmBenchmark
from src.evaluation.statistics import wilcoxon_test, StatisticalSummary
from src.visualization.gantt import plot_gantt_chart
from src.visualization.convergence import (
    plot_convergence,
    plot_comparison_boxplot,
    plot_scalability,
)
from experiments.config import (
    PSO_CONFIG, GA_CONFIG, EXPERIMENT_SCALES,
    N_RUNS, RANDOM_SEED, RESULTS_DIR, BENCHMARK_INSTANCES,
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_correctness_check():
    """
    Kiểm tra tính đúng đắn: so sánh với optimal đã biết
    trên benchmark chuẩn ft06 (optimal = 55).
    """
    print("\n" + "=" * 60)
    print("BƯỚC 1: KIỂM TRA TÍNH ĐÚNG ĐẮN (Benchmark ft06)")
    print("=" * 60)

    instance = generate_benchmark_instance("ft06")
    print(f"Instance: {instance} — Optimal makespan = 55")

    pso_cfg = PSOConfig(**{**PSO_CONFIG, "max_iter": 300, "n_particles": 80})
    ga_cfg = GAConfig(**{**GA_CONFIG, "generations": 400, "pop_size": 150})

    pso = PSOScheduler(instance, pso_cfg)
    ga = GAScheduler(instance, ga_cfg)
    greedy = GreedyScheduler(instance)

    results = {}
    for name, scheduler in [("PSO", pso), ("GA", ga), ("Greedy-LPT", greedy)]:
        r = scheduler.run()
        results[name] = r
        gap = (r.best_makespan - 55) / 55 * 100
        print(f"  {name:<15}: Makespan = {r.best_makespan:.0f} | Gap từ optimal = {gap:.1f}%")

    return results


def run_scalability_analysis():
    """
    Phân tích điểm gãy (break-point): thời gian thực thi theo quy mô.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 2: PHÂN TÍCH ĐIỂM GÃY (Scalability)")
    print("=" * 60)

    scales = [s["n_jobs"] for s in EXPERIMENT_SCALES]
    time_data = {
        "PSO": [],
        "GA": [],
        "Greedy-LPT": [],
        "DP-Bitmask": [],
        "Branch&Bound": [],
        "Backtracking": [],
    }

    for scale_cfg in EXPERIMENT_SCALES:
        n_jobs = scale_cfg["n_jobs"]
        n_machines = scale_cfg["n_machines"]
        label = scale_cfg["label"]
        include_exact = scale_cfg["include_exact"]

        print(f"\n  Quy mô: {label}")
        instance = generate_random_instance(n_jobs, n_machines, seed=RANDOM_SEED)

        pso_cfg = PSOConfig(**PSO_CONFIG)
        ga_cfg = GAConfig(**GA_CONFIG)

        bench = AlgorithmBenchmark(instance, include_exact=include_exact, pso_config=pso_cfg, ga_config=ga_cfg)
        report = bench.run_all(n_runs=3, verbose=True)

        for alg_name in time_data:
            if alg_name in report.metrics:
                time_data[alg_name].append(report.metrics[alg_name].mean_time)
            else:
                time_data[alg_name].append(None)

    fig = plot_scalability(
        scales=scales,
        time_data={k: v for k, v in time_data.items()},
        title="Phân tích Điểm Gãy: Thời gian thực thi theo Quy mô bài toán",
        save_path=os.path.join(RESULTS_DIR, "scalability.png"),
    )
    print(f"\n  Đã lưu biểu đồ scalability: {RESULTS_DIR}/scalability.png")

    return time_data


def run_pso_vs_ga_comparison():
    """
    So sánh chi tiết PSO vs GA: makespan, thời gian, hội tụ, kiểm định thống kê.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 3: SO SÁNH PSO vs GA (n=50, m=6, 10 lần chạy)")
    print("=" * 60)

    instance = generate_random_instance(50, 6, seed=RANDOM_SEED)
    pso_cfg = PSOConfig(**PSO_CONFIG)
    ga_cfg = GAConfig(**GA_CONFIG)

    pso_results = []
    ga_results = []
    pso_histories = []
    ga_histories = []

    for i in range(N_RUNS):
        pso_cfg.seed = 42 + i
        ga_cfg.seed = 42 + i

        pr = PSOScheduler(instance, pso_cfg).run()
        gr = GAScheduler(instance, ga_cfg).run()

        pso_results.append(pr)
        ga_results.append(gr)
        pso_histories.append(pr.history)
        ga_histories.append(gr.history)

        print(f"  Run {i+1:2d}: PSO={pr.best_makespan:.1f} ({pr.elapsed_time:.3f}s) | "
              f"GA={gr.best_makespan:.1f} ({gr.elapsed_time:.3f}s)")

    # Thống kê tổng hợp
    pso_makespans = [r.best_makespan for r in pso_results]
    ga_makespans = [r.best_makespan for r in ga_results]

    print(f"\n  PSO: mean={sum(pso_makespans)/len(pso_makespans):.1f} | "
          f"min={min(pso_makespans):.1f} | "
          f"time={sum(r.elapsed_time for r in pso_results)/N_RUNS:.3f}s")
    print(f"  GA : mean={sum(ga_makespans)/len(ga_makespans):.1f} | "
          f"min={min(ga_makespans):.1f} | "
          f"time={sum(r.elapsed_time for r in ga_results)/N_RUNS:.3f}s")

    # Kiểm định Wilcoxon
    test = wilcoxon_test(pso_makespans, ga_makespans, "PSO", "GA")
    StatisticalSummary([test]).print_report()

    # Vẽ đường cong hội tụ trung bình
    min_len = min(len(h) for h in pso_histories + ga_histories)
    pso_avg = [sum(h[i] for h in pso_histories) / N_RUNS for i in range(min_len)]
    ga_avg = [sum(h[i] for h in ga_histories) / N_RUNS for i in range(min_len)]

    plot_convergence(
        histories={"PSO (trung bình)": pso_avg, "GA (trung bình)": ga_avg},
        title="Đường cong hội tụ PSO vs GA (n=50, m=6)",
        save_path=os.path.join(RESULTS_DIR, "convergence_pso_ga.png"),
    )

    return pso_results, ga_results, test


def run_full_comparison():
    """
    So sánh tất cả thuật toán trên ba mức quy mô.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 4: SO SÁNH ĐẦY ĐỦ TẤT CẢ THUẬT TOÁN")
    print("=" * 60)

    configs = [
        (6, 3, True, "small"),
        (20, 5, False, "medium"),
        (50, 6, False, "large"),
    ]

    for n_jobs, n_machines, include_exact, size_tag in configs:
        instance = generate_random_instance(n_jobs, n_machines, seed=RANDOM_SEED)
        pso_cfg = PSOConfig(**PSO_CONFIG)
        ga_cfg = GAConfig(**GA_CONFIG)

        bench = AlgorithmBenchmark(instance, include_exact=include_exact,
                                   pso_config=pso_cfg, ga_config=ga_cfg)
        report = bench.run_all(n_runs=N_RUNS, verbose=True)
        report.print_summary()
        report.save(os.path.join(RESULTS_DIR, f"comparison_{size_tag}.csv"))

        # Boxplot so sánh makespan
        metrics_for_plot = [m for m in report.metrics.values() if len(m.makespans) > 1]
        if metrics_for_plot:
            plot_comparison_boxplot(
                metrics_for_plot,
                title=f"So sánh Makespan — {n_jobs} jobs × {n_machines} machines",
                save_path=os.path.join(RESULTS_DIR, f"boxplot_{size_tag}.png"),
            )


def run_gantt_visualization():
    """
    Tạo biểu đồ Gantt cho nghiệm tốt nhất của PSO và GA.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 5: TRỰC QUAN HÓA BIỂU ĐỒ GANTT")
    print("=" * 60)

    instance = generate_random_instance(8, 3, seed=RANDOM_SEED)
    pso_cfg = PSOConfig(**{**PSO_CONFIG, "max_iter": 300})
    ga_cfg = GAConfig(**{**GA_CONFIG, "generations": 400})

    for name, scheduler, file_tag in [
        ("PSO", PSOScheduler(instance, pso_cfg), "pso"),
        ("GA", GAScheduler(instance, ga_cfg), "ga"),
        ("Greedy-LPT", GreedyScheduler(instance), "greedy"),
    ]:
        result = scheduler.run()
        path = os.path.join(RESULTS_DIR, f"gantt_{file_tag}.png")
        plot_gantt_chart(
            instance,
            result.best_sequence,
            title=f"Biểu đồ Gantt — {name} (Makespan={result.best_makespan:.0f})",
            save_path=path,
        )
        print(f"  {name}: Makespan={result.best_makespan:.0f} → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("THỰC NGHIỆM: GA & PSO cho bài toán Lập lịch Công việc")
    print("So sánh với: Greedy, Backtracking, DP, Branch & Bound")
    print("=" * 60)

    t_total = time.perf_counter()

    run_correctness_check()
    run_scalability_analysis()
    run_pso_vs_ga_comparison()
    run_full_comparison()
    run_gantt_visualization()

    elapsed = time.perf_counter() - t_total
    print(f"\n✅ Hoàn thành toàn bộ thực nghiệm trong {elapsed:.1f}s")
    print(f"📁 Kết quả lưu tại: {os.path.abspath(RESULTS_DIR)}/")
