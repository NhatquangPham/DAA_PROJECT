"""
Khung chạy thực nghiệm (benchmarking framework).
So sánh nhiều thuật toán trên cùng một instance với nhiều lần chạy.
"""
from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.problem.job import JSPInstance
from src.algorithms.base import AlgorithmResult, BaseScheduler
from src.algorithms.pso import PSOScheduler, PSOConfig
from src.algorithms.ga import GAScheduler, GAConfig
from src.algorithms.greedy import GreedyScheduler
from src.algorithms.backtracking import BacktrackingScheduler
from src.algorithms.dp import DPScheduler
from src.algorithms.branch_bound import BranchBoundScheduler
from .metrics import PerformanceMetrics


@dataclass
class BenchmarkReport:
    """
    Báo cáo tổng hợp kết quả thực nghiệm.

    Attributes:
        instance_name   : Tên instance
        n_jobs          : Số công việc
        n_machines      : Số máy
        metrics         : Dict[tên thuật toán → PerformanceMetrics]
        all_results     : Tất cả kết quả thô
    """
    instance_name: str
    n_jobs: int
    n_machines: int
    metrics: Dict[str, PerformanceMetrics]
    all_results: Dict[str, List[AlgorithmResult]]

    def print_summary(self) -> None:
        """In bảng tổng kết kết quả."""
        print("\n" + "=" * 90)
        print(f"KẾT QUẢ THỰC NGHIỆM: {self.instance_name} ({self.n_jobs} jobs × {self.n_machines} machines)")
        print("=" * 90)
        header = f"{'Thuật toán':<22} {'Makespan TB':>12} {'Std':>8} {'Min':>8} {'Max':>8} {'Time (s)':>10} {'Mem (KB)':>10}"
        print(header)
        print("-" * 90)
        for name, m in sorted(self.metrics.items(), key=lambda x: x[1].mean_makespan):
            row = (
                f"{name:<22} "
                f"{m.mean_makespan:>12.1f} "
                f"{m.std_makespan:>8.1f} "
                f"{m.min_makespan:>8.1f} "
                f"{m.max_makespan:>8.1f} "
                f"{m.mean_time:>10.4f} "
                f"{m.mean_memory_kb:>10.1f}"
            )
            print(row)
        print("=" * 90)

    def save(self, filepath: str) -> None:
        """Lưu kết quả ra file CSV."""
        rows = [m.as_dict() for m in self.metrics.values()]
        if not rows:
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Đã lưu kết quả vào: {filepath}")


class AlgorithmBenchmark:
    """
    Khung so sánh tất cả thuật toán trên một instance JSP.

    Chạy mỗi thuật toán n_runs lần và tổng hợp kết quả
    thành BenchmarkReport.
    """

    def __init__(
        self,
        instance: JSPInstance,
        include_exact: bool = True,
        pso_config: Optional[PSOConfig] = None,
        ga_config: Optional[GAConfig] = None,
    ) -> None:
        self.instance = instance
        self.include_exact = include_exact
        self.pso_config = pso_config or PSOConfig()
        self.ga_config = ga_config or GAConfig()

    def run_all(self, n_runs: int = 10, verbose: bool = True) -> BenchmarkReport:
        """
        Chạy tất cả thuật toán và trả về báo cáo.

        Args:
            n_runs  : Số lần chạy cho mỗi thuật toán (metaheuristic)
            verbose : In tiến trình

        Returns:
            BenchmarkReport chứa đầy đủ kết quả
        """
        all_results: Dict[str, List[AlgorithmResult]] = {}

        algorithms: Dict[str, BaseScheduler] = {
            "PSO": PSOScheduler(self.instance, self.pso_config),
            "GA": GAScheduler(self.instance, self.ga_config),
            "Greedy-LPT": GreedyScheduler(self.instance, strategy="LPT"),
            "Greedy-SPT": GreedyScheduler(self.instance, strategy="SPT"),
        }

        if self.include_exact:
            algorithms.update({
                "Backtracking": BacktrackingScheduler(self.instance),
                "DP-Bitmask": DPScheduler(self.instance),
                "Branch&Bound": BranchBoundScheduler(self.instance),
            })

        for name, scheduler in algorithms.items():
            if verbose:
                print(f"Chạy {name}...", end="", flush=True)

            results = []
            # Greedy, B&B, DP chỉ chạy 1 lần (deterministic)
            is_deterministic = name in ("Greedy-LPT", "Greedy-SPT", "DP-Bitmask", "Branch&Bound", "Backtracking")
            runs = 1 if is_deterministic else n_runs

            for i in range(runs):
                # Thay seed khác nhau cho mỗi lần chạy (metaheuristic)
                if hasattr(scheduler, "config") and hasattr(scheduler.config, "seed"):
                    scheduler.config.seed = 42 + i
                result = scheduler.run()
                results.append(result)

            all_results[name] = results

            if verbose:
                best = min(r.best_makespan for r in results)
                t = sum(r.elapsed_time for r in results) / len(results)
                print(f" ✓  Makespan={best:.1f}, Time={t:.3f}s")

        metrics = {
            name: PerformanceMetrics.from_results(results)
            for name, results in all_results.items()
        }

        return BenchmarkReport(
            instance_name=self.instance.name,
            n_jobs=self.instance.num_jobs,
            n_machines=self.instance.num_machines,
            metrics=metrics,
            all_results=all_results,
        )
