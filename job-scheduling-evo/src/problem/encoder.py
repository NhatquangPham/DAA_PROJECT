"""
Kỹ thuật mã hóa và giải mã nghiệm cho bài toán JSP.

Hỗ trợ:
- SPV (Smallest Position Value): vector thực → hoán vị
- Giải mã chuỗi công việc → Makespan
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import List
from .job import JSPInstance


class Encoder:
    """
    Bộ mã hóa / giải mã nghiệm cho JSP.

    Kỹ thuật SPV (Smallest Position Value):
    ----------------------------------------
    Cho vector thực x = (x₁, x₂, ..., xₙ), ta xây dựng hoán vị π
    bằng cách sắp thứ tự các phần tử theo giá trị tăng dần:
        π(1) ≤ π(2) ≤ ... ≤ π(n)

    Trong ngữ cảnh JSP với n_jobs công việc và n_machines máy:
    - Kích thước vector: D = n_jobs × n_machines
    - Mỗi nhóm n_machines vị trí ứng với 1 lần xuất hiện của từng job
    - job_id = index % n_jobs
    """

    @staticmethod
    def spv_decode(position: np.ndarray, n_jobs: int, n_machines: int) -> List[int]:
        """
        Chuyển đổi vector vị trí PSO → chuỗi công việc (hoán vị).

        Args:
            position    : Vector thực kích thước (n_jobs × n_machines,)
            n_jobs      : Số công việc
            n_machines  : Số máy

        Returns:
            Chuỗi công việc có độ dài (n_jobs × n_machines)
        """
        indices = np.argsort(position)
        return [int(idx % n_jobs) for idx in indices]

    @staticmethod
    def random_permutation(n_jobs: int, n_machines: int, rng: np.random.Generator | None = None) -> List[int]:
        """
        Sinh ngẫu nhiên một chuỗi công việc hợp lệ.
        Mỗi job xuất hiện đúng n_machines lần.

        Args:
            n_jobs      : Số công việc
            n_machines  : Số máy
            rng         : Generator ngẫu nhiên (tùy chọn)

        Returns:
            Chuỗi hoán vị ngẫu nhiên
        """
        if rng is None:
            rng = np.random.default_rng()
        seq = list(range(n_jobs)) * n_machines
        rng.shuffle(seq)
        return seq

    @staticmethod
    def greedy_permutation(instance: JSPInstance) -> List[int]:
        """
        Xây dựng chuỗi công việc theo heuristic SPT (Shortest Processing Time).
        Dùng để khởi tạo quần thể ban đầu (greedy seeding).

        Args:
            instance: Instance JSP

        Returns:
            Chuỗi công việc theo thứ tự SPT
        """
        job_pts = [(j, instance.jobs[j].total_processing_time) for j in range(instance.num_jobs)]
        sorted_jobs = [j for j, _ in sorted(job_pts, key=lambda x: x[1])]
        return sorted_jobs * instance.num_machines


def decode_sequence_to_makespan(sequence: List[int], instance: JSPInstance) -> int:
    """
    Giải mã chuỗi công việc → Makespan (hàm mục tiêu).

    Thuật toán:
    1. Duyệt qua từng job_id trong sequence
    2. Lấy thao tác tiếp theo cần thực hiện của job đó
    3. Thời điểm bắt đầu = max(thời điểm máy rảnh, thời điểm job rảnh)
    4. Cập nhật thời điểm rảnh của máy và job

    Độ phức tạp: O(n × m) với n là số jobs, m là số máy

    Args:
        sequence    : Chuỗi công việc (có thể lặp)
        instance    : Instance JSP

    Returns:
        Makespan (thời điểm hoàn thành muộn nhất)
    """
    n_jobs = instance.num_jobs
    n_machines = instance.num_machines

    job_op_idx = [0] * n_jobs
    machine_available = [0] * n_machines
    job_available = [0] * n_jobs

    for job_id in sequence:
        op_idx = job_op_idx[job_id]
        if op_idx >= instance.jobs[job_id].num_operations:
            continue

        task = instance.jobs[job_id].tasks[op_idx]
        machine_id = task.machine_id
        pt = task.processing_time

        start = max(machine_available[machine_id], job_available[job_id])
        end = start + pt

        task.start_time = start
        task.end_time = end

        machine_available[machine_id] = end
        job_available[job_id] = end
        job_op_idx[job_id] += 1

    return max(machine_available)
