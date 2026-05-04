"""
Trực quan hóa lịch trình bằng biểu đồ Gantt.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from typing import List, Optional

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan


def plot_gantt_chart(
    instance: JSPInstance,
    sequence: List[int],
    title: str = "Biểu đồ Gantt - Lịch trình công việc",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ Gantt cho lịch trình công việc.

    Mỗi thanh ngang đại diện cho một thao tác (Task):
    - Trục X: thời gian
    - Trục Y: máy
    - Màu sắc: phân biệt công việc

    Args:
        instance    : Instance JSP (đã được giải mã)
        sequence    : Chuỗi công việc
        title       : Tiêu đề biểu đồ
        figsize     : Kích thước figure
        save_path   : Đường dẫn lưu file (None = chỉ hiển thị)

    Returns:
        matplotlib Figure
    """
    # Giải mã lịch trình
    instance.reset()
    makespan = decode_sequence_to_makespan(sequence, instance)

    fig, ax = plt.subplots(figsize=figsize)

    n_jobs = instance.num_jobs
    n_machines = instance.num_machines

    # Tạo bảng màu cho từng job
    cmap = cm.get_cmap("tab20", n_jobs)
    colors = [cmap(j) for j in range(n_jobs)]

    # Vẽ các thanh Gantt
    for job in instance.jobs:
        for task in job.tasks:
            if task.start_time is None:
                continue
            ax.barh(
                y=task.machine_id,
                width=task.processing_time,
                left=task.start_time,
                color=colors[job.job_id],
                edgecolor="white",
                linewidth=0.8,
                height=0.6,
                alpha=0.85,
            )
            # Label thời gian nếu đủ rộng
            if task.processing_time >= 2:
                ax.text(
                    x=task.start_time + task.processing_time / 2,
                    y=task.machine_id,
                    s=f"J{job.job_id}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )

    # Trục và nhãn
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Máy {m}" for m in range(n_machines)], fontsize=10)
    ax.set_xlabel("Thời gian", fontsize=11)
    ax.set_title(f"{title}\nMakespan = {makespan}", fontsize=12, fontweight="bold")

    # Đường dọc makespan
    ax.axvline(x=makespan, color="red", linestyle="--", linewidth=1.5, label=f"Makespan = {makespan}")

    # Legend cho jobs
    legend_patches = [
        mpatches.Patch(color=colors[j], label=f"Job {j}")
        for j in range(min(n_jobs, 10))
    ]
    if n_jobs > 10:
        legend_patches.append(mpatches.Patch(color="gray", label=f"...+{n_jobs-10} jobs"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ Gantt: {save_path}")

    return fig
