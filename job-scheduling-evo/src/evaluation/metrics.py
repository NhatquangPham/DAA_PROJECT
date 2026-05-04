"""
Các chỉ số đánh giá hiệu năng thuật toán.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.algorithms.base import AlgorithmResult


@dataclass
class PerformanceMetrics:
    """
    Tổng hợp chỉ số hiệu năng của một thuật toán qua nhiều lần chạy.

    Attributes:
        algorithm_name  : Tên thuật toán
        n_runs          : Số lần chạy
        makespans       : Danh sách makespan từng lần chạy
        times           : Danh sách thời gian thực thi từng lần chạy
        mean_makespan   : Makespan trung bình
        std_makespan    : Độ lệch chuẩn makespan
        min_makespan    : Makespan tốt nhất
        max_makespan    : Makespan xấu nhất
        mean_time       : Thời gian thực thi trung bình
        mean_memory_kb  : Bộ nhớ trung bình (KB)
    """
    algorithm_name: str
    n_runs: int
    makespans: List[float]
    times: List[float]
    memories: List[float]

    @property
    def mean_makespan(self) -> float:
        return sum(self.makespans) / len(self.makespans) if self.makespans else 0.0

    @property
    def std_makespan(self) -> float:
        if len(self.makespans) < 2:
            return 0.0
        mean = self.mean_makespan
        variance = sum((x - mean) ** 2 for x in self.makespans) / (len(self.makespans) - 1)
        return variance ** 0.5

    @property
    def min_makespan(self) -> float:
        return min(self.makespans) if self.makespans else 0.0

    @property
    def max_makespan(self) -> float:
        return max(self.makespans) if self.makespans else 0.0

    @property
    def mean_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def mean_memory_kb(self) -> float:
        return sum(self.memories) / len(self.memories) if self.memories else 0.0

    @classmethod
    def from_results(cls, results: List[AlgorithmResult]) -> "PerformanceMetrics":
        """Tạo PerformanceMetrics từ danh sách AlgorithmResult."""
        if not results:
            raise ValueError("Danh sách kết quả không được rỗng.")
        return cls(
            algorithm_name=results[0].algorithm_name,
            n_runs=len(results),
            makespans=[r.best_makespan for r in results],
            times=[r.elapsed_time for r in results],
            memories=[r.memory_kb for r in results],
        )

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm_name,
            "n_runs": self.n_runs,
            "mean_makespan": round(self.mean_makespan, 2),
            "std_makespan": round(self.std_makespan, 2),
            "min_makespan": round(self.min_makespan, 2),
            "max_makespan": round(self.max_makespan, 2),
            "mean_time_s": round(self.mean_time, 4),
            "mean_memory_kb": round(self.mean_memory_kb, 1),
        }

    def __repr__(self) -> str:
        return (
            f"[{self.algorithm_name}] "
            f"Makespan={self.mean_makespan:.1f}±{self.std_makespan:.1f} "
            f"(min={self.min_makespan:.1f}, max={self.max_makespan:.1f}) | "
            f"Time={self.mean_time:.4f}s | Mem={self.mean_memory_kb:.1f}KB"
        )
