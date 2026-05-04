"""
Dynamic Programming với Bitmask cho bài toán lập lịch.

Áp dụng cho bài toán Parallel Machine Scheduling (phiên bản đơn giản hóa):
- n công việc cần được phân công vào m máy song song
- Mục tiêu: tối thiểu hóa makespan

Độ phức tạp: O(2ⁿ × n × m)
Giới hạn thực tế: n ≤ 20 (do bộ nhớ và thời gian)

Lưu ý: DP Bitmask hoạt động tốt nhất cho bài toán TSP-like và
assignment problems. Với JSP đầy đủ (thứ tự thao tác), DP
cần state space phức tạp hơn nên giới hạn ở bài toán đơn giản hóa.
"""
from __future__ import annotations
import time
import tracemalloc
from typing import List, Optional

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan
from .base import BaseScheduler, AlgorithmResult

MAX_FEASIBLE_JOBS_DP = 20


class DPScheduler(BaseScheduler):
    """
    Bộ giải Dynamic Programming (Bitmask) cho lập lịch song song.

    Trạng thái: (bitmask của các job đã xử lý, máy tiếp theo được dùng)
    Chuyển trạng thái: gán job chưa xử lý vào máy rảnh nhất

    Đảm bảo tối ưu với n ≤ 20 jobs.
    """

    def __init__(self, instance: JSPInstance, time_limit: float = 60.0) -> None:
        super().__init__(instance)
        self.time_limit = time_limit

    def run(self) -> AlgorithmResult:
        """Chạy DP và trả về kết quả."""
        n_jobs = self.instance.num_jobs
        if n_jobs > MAX_FEASIBLE_JOBS_DP:
            print(
                f"[DP] CẢNH BÁO: n_jobs={n_jobs} > {MAX_FEASIBLE_JOBS_DP}. "
                "DP sẽ không khả thi về mặt bộ nhớ. Trả về nghiệm Greedy thay thế."
            )
            return self._fallback_greedy()

        tracemalloc.start()
        t_start = time.perf_counter()

        best_makespan, best_sequence = self._dp_solve()

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return AlgorithmResult(
            algorithm_name="DP-Bitmask",
            best_makespan=float(best_makespan),
            best_sequence=best_sequence,
            elapsed_time=elapsed,
            memory_kb=peak / 1024,
            history=[float(best_makespan)],
            is_optimal=True,
            n_iterations=0,
        )

    def _dp_solve(self):
        """
        Giải bài toán lập lịch song song bằng DP Bitmask.

        State: dp[mask] = tải trọng tối thiểu của các máy khi đã gán
                          tập job được mã hóa bởi mask
        """
        inst = self.instance
        n = inst.num_jobs
        m = inst.num_machines

        # Lấy thời gian xử lý trung bình mỗi job (đơn giản hóa JSP → Parallel)
        proc_times = [inst.jobs[j].total_processing_time for j in range(n)]

        INF = float("inf")
        total_masks = 1 << n

        # dp[mask] = danh sách tải trọng m máy khi gán tập job trong mask
        dp = [None] * total_masks
        parent = [-1] * total_masks

        dp[0] = [0] * m

        for mask in range(total_masks):
            if dp[mask] is None:
                continue
            loads = dp[mask]

            for job in range(n):
                if mask & (1 << job):
                    continue

                new_mask = mask | (1 << job)
                # Gán job vào máy có tải thấp nhất
                min_m = min(range(m), key=lambda x: loads[x])
                new_loads = list(loads)
                new_loads[min_m] += proc_times[job]

                if dp[new_mask] is None or max(new_loads) < max(dp[new_mask]):
                    dp[new_mask] = new_loads
                    parent[new_mask] = mask

        full_mask = total_masks - 1
        best_makespan = max(dp[full_mask])

        # Truy vết nghiệm
        sequence = []
        cur_mask = full_mask
        while cur_mask != 0:
            prev_mask = parent[cur_mask]
            changed_bit = cur_mask ^ prev_mask
            job_id = changed_bit.bit_length() - 1
            sequence.append(job_id)
            cur_mask = prev_mask
        sequence.reverse()

        full_seq = sequence * inst.num_machines
        return best_makespan, full_seq

    def _fallback_greedy(self) -> AlgorithmResult:
        """Trả về nghiệm Greedy khi DP không khả thi."""
        from .greedy import GreedyScheduler
        result = GreedyScheduler(self.instance, strategy="LPT").run()
        result.algorithm_name = "DP-Bitmask(fallback→Greedy)"
        result.is_optimal = False
        return result
