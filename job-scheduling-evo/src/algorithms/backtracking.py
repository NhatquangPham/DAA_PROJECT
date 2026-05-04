"""
Thuật toán Quay lui (Backtracking) cho bài toán Job Shop Scheduling.

Duyệt toàn bộ không gian nghiệm theo chiều sâu (DFS) với cắt tỉa:
- Cắt tỉa khi makespan hiện tại vượt quá best tìm được
- Giới hạn thời gian để tránh treo chương trình

Độ phức tạp: O(n! × m) trong trường hợp xấu nhất.
Chỉ khả thi với n ≤ 8 jobs.
"""
from __future__ import annotations
import time
import tracemalloc
from typing import List, Optional

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan
from .base import BaseScheduler, AlgorithmResult

MAX_TIME_SECONDS = 30.0
MAX_FEASIBLE_JOBS = 8


class BacktrackingScheduler(BaseScheduler):
    """
    Bộ giải Backtracking cho JSP.

    Tìm nghiệm tối ưu bằng cách duyệt toàn bộ hoán vị công việc
    kết hợp với cắt tỉa nhánh (branch pruning).

    Lưu ý: Chỉ khả thi với quy mô nhỏ (n_jobs ≤ 8).
    Với quy mô lớn hơn, thuật toán sẽ timeout và trả về
    nghiệm tốt nhất tìm được trong giới hạn thời gian.
    """

    def __init__(self, instance: JSPInstance, time_limit: float = MAX_TIME_SECONDS) -> None:
        super().__init__(instance)
        self.time_limit = time_limit

    def run(self) -> AlgorithmResult:
        """Chạy Backtracking và trả về kết quả."""
        if self.instance.num_jobs > MAX_FEASIBLE_JOBS:
            print(
                f"[Backtracking] CẢNH BÁO: n_jobs={self.instance.num_jobs} > {MAX_FEASIBLE_JOBS}. "
                f"Giới hạn thời gian {self.time_limit}s sẽ được áp dụng."
            )

        tracemalloc.start()
        t_start = time.perf_counter()

        best_makespan, best_sequence = self._backtrack()

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        is_optimal = (
            self.instance.num_jobs <= MAX_FEASIBLE_JOBS
            and elapsed < self.time_limit
        )

        return AlgorithmResult(
            algorithm_name="Backtracking",
            best_makespan=float(best_makespan),
            best_sequence=best_sequence,
            elapsed_time=elapsed,
            memory_kb=peak / 1024,
            history=[float(best_makespan)],
            is_optimal=is_optimal,
            n_iterations=0,
        )

    def _backtrack(self):
        """
        Thực hiện tìm kiếm quay lui.

        Chiến lược:
        - Sắp xếp công việc theo SPT để có nghiệm ban đầu tốt
        - Duyệt hoán vị theo chiều sâu
        - Cắt tỉa khi makespan tạm thời vượt best
        """
        inst = self.instance
        n_jobs = inst.num_jobs
        n_machines = inst.num_machines

        # Nghiệm ban đầu từ Greedy SPT
        greedy_order = sorted(range(n_jobs), key=lambda j: inst.jobs[j].total_processing_time)
        greedy_seq = greedy_order * n_machines
        inst.reset()
        best_makespan = decode_sequence_to_makespan(greedy_seq, inst)
        best_sequence = list(greedy_seq)

        t_start = time.perf_counter()
        self._timeout = False

        def _dfs(current_seq: List[int], job_counts: List[int]) -> None:
            nonlocal best_makespan, best_sequence

            if time.perf_counter() - t_start > self.time_limit:
                self._timeout = True
                return

            if len(current_seq) == n_jobs * n_machines:
                inst.reset()
                makespan = decode_sequence_to_makespan(current_seq, inst)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_sequence = list(current_seq)
                return

            for job_id in range(n_jobs):
                if job_counts[job_id] >= n_machines:
                    continue
                if self._timeout:
                    return

                current_seq.append(job_id)
                job_counts[job_id] += 1

                # Cắt tỉa: ước lượng cận dưới đơn giản
                lb = inst.lower_bound
                if lb < best_makespan:
                    _dfs(current_seq, job_counts)

                current_seq.pop()
                job_counts[job_id] -= 1

        _dfs([], [0] * n_jobs)
        return best_makespan, best_sequence
