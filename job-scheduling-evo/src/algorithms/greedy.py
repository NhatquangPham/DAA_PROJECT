"""
Thuật toán Tham lam (Greedy) cho bài toán Job Shop Scheduling.

Cài đặt hai biến thể:
- LPT (Longest Processing Time First): ưu tiên công việc dài nhất
- SPT (Shortest Processing Time First): ưu tiên công việc ngắn nhất

Độ phức tạp: O(n log n) — sắp xếp + gán máy tuyến tính
"""
from __future__ import annotations
import time
import tracemalloc
from typing import List, Literal

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan
from .base import BaseScheduler, AlgorithmResult


class GreedyScheduler(BaseScheduler):
    """
    Bộ giải Greedy cho JSP.

    Chiến lược:
    - LPT: sắp xếp công việc theo tổng thời gian xử lý giảm dần
    - SPT: sắp xếp công việc theo tổng thời gian xử lý tăng dần

    Sau khi sắp xếp, các công việc được lặp lại num_machines lần
    để tạo chuỗi đầy đủ cho bộ giải mã.
    """

    def __init__(
        self,
        instance: JSPInstance,
        strategy: Literal["LPT", "SPT"] = "LPT",
    ) -> None:
        super().__init__(instance)
        if strategy not in ("LPT", "SPT"):
            raise ValueError(f"Chiến lược '{strategy}' không hợp lệ. Chọn 'LPT' hoặc 'SPT'.")
        self.strategy = strategy

    def run(self) -> AlgorithmResult:
        """Chạy Greedy và trả về kết quả."""
        tracemalloc.start()
        t_start = time.perf_counter()

        sequence = self._build_sequence()
        self.instance.reset()
        makespan = decode_sequence_to_makespan(sequence, self.instance)

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return AlgorithmResult(
            algorithm_name=f"Greedy-{self.strategy}",
            best_makespan=float(makespan),
            best_sequence=sequence,
            elapsed_time=elapsed,
            memory_kb=peak / 1024,
            history=[float(makespan)],
            is_optimal=False,
            n_iterations=1,
        )

    def _build_sequence(self) -> List[int]:
        """
        Xây dựng chuỗi công việc theo chiến lược Greedy.

        Returns:
            Chuỗi công việc có độ dài (n_jobs × n_machines)
        """
        inst = self.instance
        jobs_sorted = sorted(
            range(inst.num_jobs),
            key=lambda j: inst.jobs[j].total_processing_time,
            reverse=(self.strategy == "LPT"),
        )
        return jobs_sorted * inst.num_machines
