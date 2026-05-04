"""
Sinh dữ liệu bài toán JSP: ngẫu nhiên và benchmark chuẩn.
"""
from __future__ import annotations
import random
from typing import Optional
from .job import Task, Job, JSPInstance


def generate_random_instance(
    n_jobs: int,
    n_machines: int,
    min_pt: int = 1,
    max_pt: int = 20,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> JSPInstance:
    """
    Sinh ngẫu nhiên một instance JSP.

    Mỗi công việc gồm n_machines thao tác, máy được gán ngẫu nhiên
    (đảm bảo mỗi job dùng mỗi máy đúng 1 lần).

    Args:
        n_jobs      : Số công việc
        n_machines  : Số máy
        min_pt      : Thời gian xử lý tối thiểu
        max_pt      : Thời gian xử lý tối đa
        seed        : Hạt giống ngẫu nhiên (để tái hiện kết quả)
        name        : Tên instance

    Returns:
        JSPInstance hoàn chỉnh
    """
    if seed is not None:
        random.seed(seed)

    if name is None:
        name = f"random_{n_jobs}x{n_machines}"

    jobs = []
    for j in range(n_jobs):
        machine_order = list(range(n_machines))
        random.shuffle(machine_order)
        tasks = [
            Task(
                job_id=j,
                machine_id=m,
                processing_time=random.randint(min_pt, max_pt),
            )
            for m in machine_order
        ]
        jobs.append(Job(job_id=j, tasks=tasks))

    return JSPInstance(jobs=jobs, num_machines=n_machines, name=name)


def generate_benchmark_instance(name: str) -> JSPInstance:
    """
    Tạo instance benchmark chuẩn.

    Hỗ trợ các benchmark nổi tiếng:
    - 'ft06' : Fisher & Thompson 6×6 (1963), optimal = 55
    - 'ft10' : Fisher & Thompson 10×10 (1963), optimal = 930
    - 'la01' : Lawrence 10×5 (1984), optimal = 666

    Args:
        name: Tên benchmark ('ft06', 'ft10', 'la01')

    Returns:
        JSPInstance tương ứng

    Raises:
        ValueError: Nếu tên benchmark không được hỗ trợ
    """
    benchmarks = {
        "ft06": _ft06(),
        "ft10": _ft10(),
        "la01": _la01(),
    }
    if name not in benchmarks:
        supported = ", ".join(f"'{k}'" for k in benchmarks)
        raise ValueError(f"Benchmark '{name}' không được hỗ trợ. Chọn một trong: {supported}")
    return benchmarks[name]


# ------------------------------------------------------------------
# Dữ liệu benchmark chuẩn
# Mỗi job được mã hóa là: [(machine_id, processing_time), ...]
# ------------------------------------------------------------------

def _build_instance(data: list[list[tuple[int, int]]], name: str) -> JSPInstance:
    n_machines = max(m for job in data for m, _ in job) + 1
    jobs = []
    for j, ops in enumerate(data):
        tasks = [Task(job_id=j, machine_id=m, processing_time=pt) for m, pt in ops]
        jobs.append(Job(job_id=j, tasks=tasks))
    return JSPInstance(jobs=jobs, num_machines=n_machines, name=name)


def _ft06() -> JSPInstance:
    """Fisher & Thompson 6×6 — Optimal makespan = 55."""
    data = [
        [(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],
        [(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
        [(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],
        [(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
        [(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],
        [(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)],
    ]
    return _build_instance(data, "ft06")


def _ft10() -> JSPInstance:
    """Fisher & Thompson 10×10 — Optimal makespan = 930."""
    data = [
        [(0,29),(1,78),(2,9),(3,36),(4,49),(5,11),(6,62),(7,56),(8,44),(9,21)],
        [(0,43),(2,90),(4,75),(6,11),(8,69),(9,28),(1,46),(3,46),(5,72),(7,30)],
        [(1,91),(0,85),(3,39),(2,74),(6,90),(5,10),(7,12),(8,89),(9,45),(4,33)],
        [(1,81),(2,95),(0,71),(4,99),(6,9),(8,52),(7,85),(3,98),(9,22),(5,43)],
        [(2,14),(0,6),(1,22),(5,61),(3,26),(4,69),(8,21),(7,49),(9,72),(6,53)],
        [(2,84),(1,2),(5,52),(3,95),(8,48),(9,72),(0,47),(6,65),(4,6),(7,25)],
        [(1,46),(0,37),(3,61),(2,13),(6,32),(5,21),(9,32),(8,89),(7,30),(4,55)],
        [(2,31),(0,86),(1,46),(5,74),(4,32),(6,88),(8,19),(9,48),(7,36),(3,79)],
        [(0,76),(1,69),(3,76),(5,51),(2,85),(9,11),(6,40),(7,89),(4,26),(8,74)],
        [(1,85),(0,13),(2,61),(6,7),(8,64),(9,76),(5,47),(3,52),(4,90),(7,45)],
    ]
    return _build_instance(data, "ft10")


def _la01() -> JSPInstance:
    """Lawrence la01 — 10 jobs × 5 machines, optimal = 666."""
    data = [
        [(1,21),(0,53),(4,95),(3,55),(2,34)],
        [(0,21),(3,52),(4,16),(2,26),(1,71)],
        [(3,39),(4,98),(1,42),(2,31),(0,12)],
        [(1,77),(0,55),(4,79),(2,66),(3,77)],
        [(0,83),(3,34),(2,64),(1,19),(4,37)],
        [(1,54),(2,43),(4,79),(0,92),(3,62)],
        [(3,69),(4,77),(1,87),(2,87),(0,93)],
        [(2,38),(0,60),(1,41),(3,24),(4,83)],
        [(3,17),(1,49),(4,25),(0,44),(2,98)],
        [(4,77),(3,79),(2,43),(1,75),(0,96)],
    ]
    return _build_instance(data, "la01")
