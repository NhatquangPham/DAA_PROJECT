"""
Branch and Bound (B&B) cho bài toán Job Shop Scheduling.

Kết hợp:
- Duyệt cây tìm kiếm (tương tự backtracking)
- Hàm ước lượng cận dưới (lower bound) để cắt tỉa hiệu quả hơn
- Best-First Search với priority queue

Cận dưới sử dụng:
1. Cận dưới theo machine: max tải trọng tối thiểu trên từng máy
2. Cận dưới theo job: max thời gian còn lại của từng job

Độ phức tạp: O(n!) worst case, thực tế tốt hơn nhiều với pruning tốt.
"""
from __future__ import annotations
import time
import tracemalloc
import heapq
from typing import List, Tuple, Optional

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan
from .base import BaseScheduler, AlgorithmResult

MAX_FEASIBLE_JOBS_BB = 12


class BranchBoundScheduler(BaseScheduler):
    """
    Bộ giải Branch and Bound cho JSP.

    Sử dụng Best-First Search với min-heap dựa trên cận dưới.
    Tích hợp cắt tỉa dựa trên upper bound hiện tại.
    """

    def __init__(self, instance: JSPInstance, time_limit: float = 60.0) -> None:
        super().__init__(instance)
        self.time_limit = time_limit

    def run(self) -> AlgorithmResult:
        """Chạy Branch and Bound và trả về kết quả."""
        n_jobs = self.instance.num_jobs
        if n_jobs > MAX_FEASIBLE_JOBS_BB:
            print(
                f"[B&B] CẢNH BÁO: n_jobs={n_jobs} > {MAX_FEASIBLE_JOBS_BB}. "
                "B&B sẽ không hoàn thành trong thời gian cho phép. "
                "Áp dụng giới hạn thời gian và trả về nghiệm tốt nhất."
            )

        tracemalloc.start()
        t_start = time.perf_counter()

        best_makespan, best_sequence = self._branch_and_bound(t_start)

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        is_optimal = n_jobs <= MAX_FEASIBLE_JOBS_BB and elapsed < self.time_limit

        return AlgorithmResult(
            algorithm_name="Branch&Bound",
            best_makespan=float(best_makespan),
            best_sequence=best_sequence,
            elapsed_time=elapsed,
            memory_kb=peak / 1024,
            history=[float(best_makespan)],
            is_optimal=is_optimal,
            n_iterations=0,
        )

    def _branch_and_bound(self, t_start: float) -> Tuple[float, List[int]]:
        """
        Best-First B&B với priority queue.

        Node = (lower_bound, current_sequence, job_counts)
        """
        inst = self.instance
        n_jobs = inst.num_jobs
        n_machines = inst.num_machines

        # Nghiệm ban đầu (upper bound) từ Greedy LPT
        greedy_order = sorted(range(n_jobs), key=lambda j: -inst.jobs[j].total_processing_time)
        greedy_seq = greedy_order * n_machines
        inst.reset()
        upper_bound = float(decode_sequence_to_makespan(greedy_seq, inst))
        best_sequence = list(greedy_seq)

        # Priority queue: (lower_bound, sequence, job_counts)
        root_lb = self._compute_lower_bound([], [0] * n_jobs, inst)
        heap = [(root_lb, [], [0] * n_jobs)]
        heapq.heapify(heap)

        nodes_explored = 0
        max_nodes = 50000

        while heap:
            if time.perf_counter() - t_start > self.time_limit:
                break
            if nodes_explored > max_nodes:
                break

            lb, seq, counts = heapq.heappop(heap)
            nodes_explored += 1

            # Cắt tỉa: cận dưới không tốt hơn upper bound
            if lb >= upper_bound:
                continue

            # Nút lá — nghiệm hoàn chỉnh
            if len(seq) == n_jobs * n_machines:
                inst.reset()
                makespan = float(decode_sequence_to_makespan(seq, inst))
                if makespan < upper_bound:
                    upper_bound = makespan
                    best_sequence = list(seq)
                continue

            # Phân nhánh: thêm từng job chưa hoàn thành
            for job_id in range(n_jobs):
                if counts[job_id] >= n_machines:
                    continue

                new_seq = seq + [job_id]
                new_counts = list(counts)
                new_counts[job_id] += 1

                child_lb = self._compute_lower_bound(new_seq, new_counts, inst)
                if child_lb < upper_bound:
                    heapq.heappush(heap, (child_lb, new_seq, new_counts))

        return upper_bound, best_sequence

    @staticmethod
    def _compute_lower_bound(
        partial_seq: List[int],
        job_counts: List[int],
        instance: JSPInstance,
    ) -> float:
        """
        Ước lượng cận dưới cho trạng thái hiện tại.

        Cận dưới = max(
            cận dưới theo machine load,
            cận dưới theo job remaining time
        )

        Đây là lower bound laxo (lax lower bound) — không vi phạm
        tính đúng đắn của B&B vì luôn ≤ makespan thực tế.
        """
        inst = instance
        n_jobs = inst.num_jobs
        n_machines = inst.num_machines

        # Giải mã partial sequence để lấy thời gian hiện tại
        if partial_seq:
            inst.reset()
            _ = decode_sequence_to_makespan(partial_seq, inst)

        machine_available = [0] * n_machines
        job_available = [0] * n_jobs

        for task in [t for job in inst.jobs for t in job.tasks if t.start_time is not None]:
            machine_available[task.machine_id] = max(
                machine_available[task.machine_id], task.end_time or 0
            )
            job_available[task.job_id] = max(
                job_available[task.job_id], task.end_time or 0
            )

        # Cận dưới theo remaining job time
        job_lb = 0
        for j in range(n_jobs):
            op_idx = job_counts[j]
            remaining = sum(
                t.processing_time for t in inst.jobs[j].tasks[op_idx:]
            )
            job_lb = max(job_lb, job_available[j] + remaining)

        # Cận dưới theo machine load
        machine_remaining = [0] * n_machines
        for j in range(n_jobs):
            op_idx = job_counts[j]
            for t in inst.jobs[j].tasks[op_idx:]:
                machine_remaining[t.machine_id] += t.processing_time

        machine_lb = max(
            machine_available[m] + machine_remaining[m] for m in range(n_machines)
        )

        return float(max(job_lb, machine_lb, inst.lower_bound))
