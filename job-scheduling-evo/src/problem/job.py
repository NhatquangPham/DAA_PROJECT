"""
Mô hình hóa bài toán Job Shop Scheduling.
Bao gồm các lớp: Task, Job, Machine, JSPInstance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    """
    Đại diện cho một thao tác (operation) của công việc trên máy cụ thể.

    Attributes:
        job_id          : ID công việc chứa thao tác này
        machine_id      : ID máy yêu cầu
        processing_time : Thời gian xử lý (đơn vị thời gian)
        start_time      : Thời điểm bắt đầu (được gán sau khi giải mã)
        end_time        : Thời điểm kết thúc (được gán sau khi giải mã)
    """
    job_id: int
    machine_id: int
    processing_time: int
    start_time: Optional[int] = field(default=None, compare=False)
    end_time: Optional[int] = field(default=None, compare=False)

    def reset(self) -> None:
        """Xóa thông tin lịch trình đã gán."""
        self.start_time = None
        self.end_time = None

    def __repr__(self) -> str:
        timing = ""
        if self.start_time is not None:
            timing = f", [{self.start_time}-{self.end_time}]"
        return f"Task(J{self.job_id}, M{self.machine_id}, pt={self.processing_time}{timing})"


class Job:
    """
    Đại diện cho một công việc gồm nhiều thao tác cần thực hiện theo thứ tự.

    Ràng buộc thứ tự: thao tác tasks[k] phải hoàn thành trước khi
    tasks[k+1] có thể bắt đầu.

    Attributes:
        job_id          : Định danh duy nhất của công việc
        tasks           : Danh sách thao tác theo thứ tự cố định
        priority        : Trọng số ưu tiên (mặc định = 1)
    """

    def __init__(self, job_id: int, tasks: List[Task], priority: int = 1) -> None:
        self.job_id = job_id
        self.tasks = tasks
        self.priority = priority

    @property
    def num_operations(self) -> int:
        """Số thao tác của công việc."""
        return len(self.tasks)

    @property
    def total_processing_time(self) -> int:
        """Tổng thời gian xử lý của tất cả thao tác."""
        return sum(t.processing_time for t in self.tasks)

    def reset(self) -> None:
        """Xóa toàn bộ thông tin lịch trình."""
        for task in self.tasks:
            task.reset()

    def __repr__(self) -> str:
        return (
            f"Job(id={self.job_id}, ops={self.num_operations}, "
            f"total_pt={self.total_processing_time})"
        )


class Machine:
    """
    Đại diện cho một máy trong hệ thống lập lịch.

    Attributes:
        machine_id      : Định danh duy nhất của máy
        schedule        : Danh sách thao tác đã được phân công
        available_time  : Thời điểm máy rảnh tiếp theo
    """

    def __init__(self, machine_id: int) -> None:
        self.machine_id = machine_id
        self.schedule: List[Task] = []
        self.available_time: int = 0

    def reset(self) -> None:
        """Đặt lại trạng thái máy."""
        self.schedule = []
        self.available_time = 0

    @property
    def total_load(self) -> int:
        """Tổng tải trọng (tổng thời gian xử lý các thao tác được gán)."""
        return sum(t.processing_time for t in self.schedule)

    def __repr__(self) -> str:
        return f"Machine(id={self.machine_id}, load={self.total_load}, tasks={len(self.schedule)})"


class JSPInstance:
    """
    Lớp bao chứa toàn bộ instance bài toán Job Shop Scheduling.

    Một instance gồm n công việc và m máy. Mỗi công việc có đúng m thao tác,
    mỗi thao tác được xử lý bởi một máy xác định trong thời gian xác định.

    Attributes:
        jobs        : Danh sách công việc
        num_jobs    : Số lượng công việc (n)
        num_machines: Số lượng máy (m)
        machines    : Danh sách máy
        name        : Tên của instance (để nhận diện)
    """

    def __init__(
        self,
        jobs: List[Job],
        num_machines: int,
        name: str = "unnamed",
    ) -> None:
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = num_machines
        self.machines = [Machine(m) for m in range(num_machines)]
        self.name = name

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái: máy và các thao tác."""
        for m in self.machines:
            m.reset()
        for j in self.jobs:
            j.reset()

    @property
    def lower_bound(self) -> int:
        """
        Cận dưới lý thuyết của makespan:
        max(tổng PT của job dài nhất, tổng PT trên máy bận nhất).
        """
        job_lb = max(j.total_processing_time for j in self.jobs)
        machine_loads = [0] * self.num_machines
        for job in self.jobs:
            for task in job.tasks:
                machine_loads[task.machine_id] += task.processing_time
        machine_lb = max(machine_loads)
        return max(job_lb, machine_lb)

    def __repr__(self) -> str:
        return (
            f"JSPInstance(name='{self.name}', "
            f"n_jobs={self.num_jobs}, n_machines={self.num_machines})"
        )
