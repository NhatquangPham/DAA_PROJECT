"""
Lớp cơ sở và cấu trúc kết quả cho tất cả thuật toán.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from src.problem.job import JSPInstance


@dataclass
class AlgorithmResult:
    """
    Kết quả trả về từ bất kỳ thuật toán nào.

    Attributes:
        algorithm_name  : Tên thuật toán
        best_makespan   : Makespan tốt nhất tìm được
        best_sequence   : Chuỗi công việc ứng với makespan tốt nhất
        elapsed_time    : Thời gian thực thi (giây)
        memory_kb       : Bộ nhớ sử dụng (KB)
        history         : Lịch sử makespan tốt nhất qua các vòng lặp (cho metaheuristic)
        is_optimal      : True nếu nghiệm được đảm bảo tối ưu
        n_iterations    : Số vòng lặp đã thực hiện
    """
    algorithm_name: str
    best_makespan: float
    best_sequence: List[int]
    elapsed_time: float
    memory_kb: float = 0.0
    history: List[float] = field(default_factory=list)
    is_optimal: bool = False
    n_iterations: int = 0

    def __repr__(self) -> str:
        opt_tag = " [OPTIMAL]" if self.is_optimal else ""
        return (
            f"[{self.algorithm_name}]{opt_tag} "
            f"Makespan={self.best_makespan} | "
            f"Time={self.elapsed_time:.4f}s | "
            f"Mem={self.memory_kb:.1f}KB"
        )


class BaseScheduler(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả thuật toán lập lịch.

    Mọi thuật toán cần triển khai phương thức `run()`.
    """

    def __init__(self, instance: JSPInstance) -> None:
        self.instance = instance

    @abstractmethod
    def run(self) -> AlgorithmResult:
        """Chạy thuật toán và trả về kết quả."""
        ...

    def _measure(self, func, *args, **kwargs):
        """
        Wrapper đo thời gian và bộ nhớ.

        Returns:
            (result, elapsed_time_seconds, memory_kb)
        """
        import time
        import tracemalloc

        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_kb = peak / 1024
        return result, elapsed, memory_kb
