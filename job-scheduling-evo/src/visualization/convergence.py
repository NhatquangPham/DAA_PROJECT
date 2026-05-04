"""
Trực quan hóa đường cong hội tụ và so sánh thuật toán.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Optional

from src.evaluation.metrics import PerformanceMetrics


def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Đường cong hội tụ",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ đường cong hội tụ (convergence curve) cho các thuật toán metaheuristic.

    Args:
        histories   : Dict[tên thuật toán → danh sách makespan theo vòng lặp]
        title       : Tiêu đề biểu đồ
        figsize     : Kích thước figure
        save_path   : Đường dẫn lưu file

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = cm.get_cmap("tab10", len(histories))
    for i, (name, history) in enumerate(histories.items()):
        iterations = list(range(len(history)))
        ax.plot(iterations, history, label=name, color=cmap(i), linewidth=2, alpha=0.85)

    ax.set_xlabel("Vòng lặp / Thế hệ", fontsize=11)
    ax.set_ylabel("Makespan tốt nhất", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison_boxplot(
    metrics_list: List[PerformanceMetrics],
    title: str = "So sánh Makespan (Boxplot)",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ boxplot so sánh phân phối makespan của các thuật toán.

    Args:
        metrics_list: Danh sách PerformanceMetrics của từng thuật toán
        title       : Tiêu đề
        figsize     : Kích thước figure
        save_path   : Đường dẫn lưu file

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = [m.algorithm_name for m in metrics_list]
    data = [m.makespans for m in metrics_list]

    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=False)

    cmap = cm.get_cmap("tab10", len(names))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.7)

    ax.set_xlabel("Thuật toán", fontsize=11)
    ax.set_ylabel("Makespan", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scalability(
    scales: List[int],
    time_data: Dict[str, List[float]],
    title: str = "Phân tích khả năng mở rộng (Scalability)",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ thời gian thực thi theo quy mô bài toán.

    Args:
        scales      : Danh sách số công việc (trục X)
        time_data   : Dict[tên thuật toán → thời gian theo từng quy mô]
        title       : Tiêu đề
        figsize     : Kích thước figure
        save_path   : Đường dẫn lưu file

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = cm.get_cmap("tab10", len(time_data))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, (name, times) in enumerate(time_data.items()):
        valid_scales = [s for s, t in zip(scales, times) if t is not None and t < 120]
        valid_times = [t for t in times if t is not None and t < 120]

        if valid_scales:
            ax.semilogy(
                valid_scales,
                valid_times,
                label=name,
                color=cmap(i),
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=7,
            )

    ax.set_xlabel("Số công việc (n_jobs)", fontsize=11)
    ax.set_ylabel("Thời gian thực thi (giây, log scale)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
